"""Submit and record a bounded determinant comparison on archived fields."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import subprocess
import sys
import time

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'src'))
from policy import memory_reason
from registry import artifact, atomic, environment_fingerprint, event, job_metadata, sha
from submit_probe_profile import assert_idle, registered

WALL_SECONDS = 3600


def load(path):
    return json.loads(path.read_text())


def submit(run, parent, failed):
    assert_idle(P)
    for name in ('plan.json', 'summary.json'):
        registered(P, parent, parent / name)
    source, summary = (load(parent / name) for name in ('plan.json', 'summary.json'))
    assert summary['complete'] and summary['proof_benchmark_passed']
    profiles = sorted((parent / 'worker-profiles').glob('*.json'))
    assert len(profiles) == 8, 'Expected both fields across four profiled shards'
    determinant_seconds = profile_seconds = 0.
    for profile in profiles:
        registered(P, parent, profile)
        for row in load(profile)['functions']:
            profile_seconds += row['exclusive_seconds']
            if row['function'] == 'det' and Path(row['file']).name == '_linalg.py':
                determinant_seconds += row['exclusive_seconds']
    assert determinant_seconds > .5 * profile_seconds, 'Determinant bottleneck not established'
    registered(P, failed, failed / 'plan.json')
    prior = load(failed / 'plan.json')
    state = load(failed / 'state.json')
    log = failed / 'known_probe/attempt-001/command.log'
    registered(P, failed, log)
    assert state['status'] == 'blocked' and state['phase'] == 'known_probe' and state['terminal_rejection']
    assert state['generation_seconds'] == 0 and prior['worker_policy']['workers'] == 8
    assert 'Regularized Newton wall budget exhausted' in log.read_text()
    assert prior['worker_policy']['benchmark_summary_sha256'] == sha(parent / 'summary.json')
    elapsed = state['cumulative_seconds']
    assert summary['cumulative_seconds'] <= elapsed
    assert elapsed + WALL_SECONDS <= prior['cumulative_pipeline_budget_hours'] * 3600
    for mapping in ('parent_files_sha256', 'scientific_sources', 'diagnostic_sources'):
        for name, digest in source[mapping].items():
            assert sha(P / name) == digest, 'Changed inherited source: ' + name
    audit = P / state['platform_output']
    assert prior['inherited_evidence_sha256'][str(audit.relative_to(P))] == sha(audit)
    platform = load(audit)
    snapshot = load(P / source['snapshot'])
    assert platform['passed'] and platform['audit_complete']
    count = snapshot['accepted_steps']
    assert platform['checked_steps'] == count
    hashes = {}
    for index in range(count - 3, count):
        row = platform['records'][index]
        assert row['step'] == index
        hashes[str(index)] = row['field_sha256']
    frozen = dict(source['parent_files_sha256'])
    for path in [Path(__file__).resolve(), P / 'hpc/cofactor_proof.py', audit, log,
                 failed / 'state.json', failed / 'plan.json', parent / 'summary.json', parent / 'plan.json',
                 *profiles]:
        frozen[str(path.relative_to(P))] = sha(path)
    frozen.update(source['scientific_sources'])
    frozen.update(source['diagnostic_sources'])
    resources = ('queue', 'cpus', 'memory_gb', 'software', 'native_prefix', 'host_free_reserve_gib',
                 'snapshot', 'snapshot_sha256', 'environment_sha256')
    plan = dict({key: source[key] for key in resources}, account=source.get('account'),
                run_id=run.name, workers=8, fixed_steps=[count-2, count-1],
                field_hashes=hashes, frozen=frozen, cumulative_seconds_before=elapsed,
                attributed_seconds=dict(determinant=determinant_seconds, total=profile_seconds),
                outer_seconds=WALL_SECONDS, physical_states_to_generate=0,
                reference_continuation_allowed=False,
                scope='Archived endpoint and stress proof comparison. Candidate bulk cofactor path '
                      'with legacy fallback near the margin and unchanged subdivision.')
    run.mkdir()
    atomic(run / 'plan.json', plan)
    atomic(run / 'state.json', dict(status='submitted', phase='cofactor_benchmark', generation_seconds=0,
                                   cumulative_seconds=elapsed, blockers=[]))
    artifact(P, run, run / 'plan.json', 'Immutable cofactor benchmark protocol')
    command = [str(P / '.tools/bin/micromamba'), 'run', '-p', str(P / plan['native_prefix']),
               'python', 'hpc/benchmark_cofactor.py', 'execute', '--run-id', run.name]
    script = run / 'job.pbs'
    script.write_text('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(P)) +
                      '\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '
                      'NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""\nexport MAMBA_ROOT_PREFIX=' +
                      shlex.quote(str(P / '.hpc/mamba')) + '\nexec ' + shlex.join(command) + '\n')
    args = ['qsub', '-N', 'lv_cofactor', '-q', plan['queue'], '-l',
            f"select=1:ncpus={plan['cpus']}:mem={plan['memory_gb']}gb",
            '-l', 'walltime=01:00:00', '-l', 'place=excl', '-l', 'software=' + plan['software'],
            '-j', 'oe', '-o', str(run / 'pbs.log')]
    if plan.get('account'):
        args += ['-A', plan['account']]
    args.append(str(script))
    reply = subprocess.run(args, cwd=P, capture_output=True, text=True)
    event(run, 'job_submission', argv=args, return_code=reply.returncode,
          stdout=reply.stdout.strip(), stderr=reply.stderr.strip())
    if reply.returncode:
        state = load(run / 'state.json')
        state.update(status='submission_failed', blockers=[reply.stderr.strip()])
        atomic(run / 'state.json', state)
        raise SystemExit(reply.stderr)
    print('Proof comparison job:', reply.stdout.strip())
    print('Results:', run.relative_to(P) / 'summary.json')


def check_resources(plan):
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    reason = memory_reason(available, rss, plan['memory_gb'], plan['host_free_reserve_gib'])
    assert reason is None, reason
    assert shutil.disk_usage(P).free >= 100 * 2**30, '100 GiB free-disk reserve'
    return rss


def execute(run):
    assert os.environ.get('PBS_JOBID'), 'Use PBS for full-size proof benchmarking'
    with (run / 'execution.lock').open('a') as lock, (P / '.hpc/profile-execution.lock').open('a') as shared:
        for handle in (lock, shared):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registered(P, run, run / 'plan.json')
        plan, state = (load(run / name) for name in ('plan.json', 'state.json'))
        assert state['status'] == 'submitted' and not (run / 'summary.json').exists()
        state.update(status='running')
        atomic(run / 'state.json', state)
        event(run, 'job_started', **job_metadata())
        started = time.monotonic()
        result = dict(complete=False, proof_benchmark_passed=False, reference_continuation_allowed=False,
                      physical_new_state_generated=False, records=[], stop_reason=None)
        try:
            assert len(job_metadata().get('allocated_hosts', [])) == 1
            for name, digest in plan['frozen'].items():
                assert sha(P / name) == digest, 'Changed bound input: ' + name
            for path, digest in plan['environment_sha256'].items():
                assert environment_fingerprint(P / path) == digest, 'Changed environment: ' + path
            compare(run, plan, result)
            result.update(complete=True, proof_benchmark_passed=True)
        except Exception as exc:
            result['stop_reason'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            result['wall_seconds'] = time.monotonic() - started
            result['cumulative_seconds'] = plan['cumulative_seconds_before'] + result['wall_seconds']
            result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            atomic(run / 'summary.json', result)
            state.update(status='diagnostic_complete' if result['complete'] else 'diagnostic_incomplete',
                         cumulative_seconds=result['cumulative_seconds'],
                         blockers=['Native known-pair qualification is pending'])
            atomic(run / 'state.json', state)
            artifact(P, run, run / 'summary.json', 'Archived proof equivalence and measured timings')
            event(run, 'job_finished', status=state['status'], pbs_job_id=os.environ['PBS_JOBID'])


def compare(run, plan, result):
    import numpy as np
    from cofactor_proof import CofactorBernsteinAudit
    from lv_surrogate.physics.threaded_bernstein import ThreadedBernsteinAudit
    from proof_benchmark import same_proof

    snapshot_path = P / plan['snapshot']
    assert sha(snapshot_path) == plan['snapshot_sha256']
    snapshot = load(snapshot_path)
    check_resources(plan)
    for path, digest in snapshot['files'].items():
        assert sha(P / path) == digest, 'Changed archived input: ' + path
    root = P / 'data/processed' / snapshot['reference']
    with np.load(root / 'inputs.npz') as inputs:
        arrays = {name: inputs[name] for name in ('coords', 'cells', 'canonical_cell_dofs')}
    fields = np.load(root / 'displacement_coefficients.npy', mmap_mode='r')
    archived = load(root / 'positivity.json')
    for step, order in zip(plan['fixed_steps'], [('legacy', 'cofactor'), ('cofactor', 'legacy')], strict=True):
        u, old = np.array(fields[step]), np.array(fields[step-1])
        for index, field in [(step, u), (step-1, old)]:
            assert hashlib.sha256(field.tobytes()).hexdigest() == plan['field_hashes'][str(index)]
        proofs, times = {}, {}
        for name in order:
            check_resources(plan)
            cls = ThreadedBernsteinAudit if name == 'legacy' else CofactorBernsteinAudit
            audit = cls(arrays, workers=plan['workers'], chunk=8192, margin=1e-9, max_depth=12)
            print('PROOF COMPARISON BEGIN', step, name, flush=True)
            start = time.perf_counter()
            try:
                proofs[name] = audit.pair(u, old, alpha_f=.4)
                times[name] = time.perf_counter() - start
            finally:
                audit.close()
            output = run / f'proof-{step}-{name}.json'
            atomic(output, proofs[name])
            artifact(P, run, output, 'Fixed-field proof output')
            print('PROOF COMPARISON SAVED', step, name, times[name], flush=True)
        same_proof(archived[step], proofs['legacy'])
        same_proof(proofs['legacy'], proofs['cofactor'])
        result['records'].append(dict(step=step, pair_seconds=times, proof_agreement=True,
                                      speedup=times['legacy']/times['cofactor']))
        atomic(run / 'progress.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['submit', 'execute'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--parent-run', default='cmbbe-proof-01')
    parser.add_argument('--failed-run', default='cmbbe-worker8-01')
    args = parser.parse_args()
    for name in (args.run_id, args.parent_run, args.failed_run):
        assert re.fullmatch('[A-Za-z0-9_-]+', name), 'Invalid run ID'
    directory = P / 'experiments/hpc'
    if args.action == 'submit':
        with (P / '.hpc/profile-submission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            submit(directory / args.run_id, directory / args.parent_run, directory / args.failed_run)
    else:
        execute(directory / args.run_id)
