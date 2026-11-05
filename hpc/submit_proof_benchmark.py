"""Register one bounded archived-field proof benchmark after completed profiling."""
import argparse
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'hpc'))
from registry import atomic, artifact, environment_fingerprint, event, job_metadata, sha
from submit_probe_profile import assert_idle, registered
from policy import memory_reason
from manuscript import render

OUTER_SECONDS = 7200
CASE_SECONDS = 900


def evidence(root, parent):
    plan = json.loads((parent / 'plan.json').read_text())
    registered(root, parent, parent / 'plan.json')
    registered(root, parent, parent / 'summary.json')
    summary = json.loads((parent / 'summary.json').read_text())
    assert summary['diagnostic_complete'] and summary['stop_reason'] is None
    assert summary['rank_reports_received'] == summary['expected_ranks'] == plan['ranks']
    assert summary['recorded_custom_iterations'] > 0 and not summary['known_pair_qualified']
    registered(root, parent, parent / 'rank-000.json')
    report = json.loads((parent / 'rank-000.json').read_text())
    assert report['status'] == 'solver_timeout' and not report['histories_advanced']
    pairs = [r['inclusive_seconds'] for r in report['functions']
             if r['function'] == 'pair' and Path(r['file']).name == 'bernstein.py']
    assert pairs and max(pairs) > summary['recorded_linear_seconds'], 'Require measured proof bottleneck'
    elapsed = summary['cumulative_seconds']
    assert elapsed >= plan['cumulative_seconds_before']
    assert elapsed + OUTER_SECONDS + 120 < plan['cumulative_pipeline_budget_hours'] * 3600
    for mapping in ['parent_files_sha256', 'scientific_sources', 'diagnostic_sources']:
        for path, digest in plan[mapping].items():
            assert sha(root / path) == digest, 'Bound parent evidence/source changed: ' + path
    original = root / 'experiments/hpc' / plan['parent_run']
    state = json.loads((original / 'state.json').read_text())
    audit = json.loads((root / state['platform_output']).read_text())
    snapshot = json.loads((root / plan['snapshot']).read_text())
    assert audit['passed'] and audit['audit_complete'] and audit['checked_steps'] == snapshot['accepted_steps']
    step = snapshot['accepted_steps'] - 2
    case = dict(step=step, previous_step=step - 1,
                field_sha256=audit['records'][step]['field_sha256'],
                previous_field_sha256=audit['records'][step - 1]['field_sha256'])
    assert audit['records'][step]['step'] == step and audit['records'][step - 1]['step'] == step - 1
    hashes = dict(plan['parent_files_sha256'])
    for name in ['plan.json', 'summary.json', 'state.json', 'events.jsonl', 'rank-000.json']:
        path = parent / name
        hashes[str(path.relative_to(root))] = sha(path)
    return plan, elapsed, case, hashes


def submit(args):
    parent = P / 'experiments/hpc' / args.parent_run
    source, elapsed, case, hashes = evidence(P, parent)
    idle = assert_idle(P)
    run = P / 'experiments/hpc' / args.run_id
    run.mkdir()  # Never retry or overwrite any output ID.
    files = dict(source['diagnostic_sources'])
    for name in ['hpc/proof_benchmark.py', 'hpc/submit_proof_benchmark.py', 'hpc/submit_probe_profile.py']:
        files[name] = sha(P / name)
    plan = dict(source, run_id=args.run_id, parent_run=args.parent_run,
                ranks=1, source_mpi_ranks=source['ranks'],
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                parent_files_sha256=hashes, diagnostic_sources=files, fixed_case=case,
                cumulative_seconds_before=elapsed, outer_seconds=OUTER_SECONDS,
                case_seconds=CASE_SECONDS, worker_order=[4, 1, 2, 8, 16, 16, 8, 2, 1, 4],
                reference_continuation_allowed=False, physical_states_to_generate=0,
                scope='Fixed archived first-known endpoint and its exact prior field; '
                      'unchanged alpha_f=.4, chunk8192, margin1e-9, max_depth12, float64 proofs. '
                      'Counterbalanced worker comparison with two repetitions/count and separate worker profiling. '
                      'Attribution-only cProfile serializes shards; excluded from concurrent timing statistics. '
                      'Original proof topology must agree; bounds within1e-12 roundoff, not a relaxed positivity margin. '
                      'No solver/history updates, full-reference restart, ML or automatic worker-policy change.')
    atomic(run / 'plan.json', plan)
    atomic(run / 'state.json', dict(status='submitted', phase='proof_benchmark', blockers=[],
           completed_phases=[], generation_seconds=0, cumulative_seconds=elapsed))
    event(run, 'diagnostic_prepared', scheduler_output=idle, carried_cumulative_seconds=elapsed,
          parent_run=args.parent_run, fixed_case=case)
    artifact(P, run, run / 'plan.json', 'Immutable fixed-field proof benchmark plan')
    script = run / 'job.pbs'
    script.write_text('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(P)) +
                      '\nexec .venv/bin/python hpc/submit_proof_benchmark.py execute --run-id ' +
                      shlex.quote(args.run_id) + '\n')
    command = ['qsub', '-N', 'lv_proof_bench', '-q', plan['queue'], '-l',
               f"select=1:ncpus={plan['cpus']}:mem={plan['memory_gb']}gb",
               '-l', 'walltime=03:00:00', '-l', 'software=' + plan['software'],
               '-l', 'place=excl', '-j', 'oe', '-o', str(run / 'pbs.log')]
    if plan.get('account'):
        command += ['-A', plan['account']]
    command.append(str(script))
    result = subprocess.run(command, cwd=P, capture_output=True, text=True)
    event(run, 'job_submission', argv=command, return_code=result.returncode,
          stdout=result.stdout.strip(), stderr=result.stderr.strip())
    if result.returncode:
        state = json.loads((run / 'state.json').read_text())
        state.update(status='submission_failed', blockers=[result.stderr.strip()])
        atomic(run / 'state.json', state)
        raise SystemExit(result.stderr)
    print('Proof benchmark job:', result.stdout.strip())
    print('One diagnostic; no reference restart. Results:', run.relative_to(P) / 'summary.json')


def execute(args):
    assert os.environ.get('PBS_JOBID'), 'Use PBS; never run this benchmark on the gateway or laptop'
    run = P / 'experiments/hpc' / args.run_id
    locks = [(run / 'execution.lock').open('a'), (P / '.hpc/profile-execution.lock').open('a')]
    for lock in locks:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registered(P, run, run / 'plan.json')
    plan = json.loads((run / 'plan.json').read_text())
    state = json.loads((run / 'state.json').read_text())
    assert state['status'] == 'submitted' and not (run / 'summary.json').exists()
    state.update(status='running')
    atomic(run / 'state.json', state)
    event(run, 'job_started', **job_metadata())
    started, child, reason, code, peak = time.monotonic(), None, None, None, 0
    interrupted = []
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1]:
        signal.signal(sig, lambda s, f: interrupted.append(s))
    try:
        assert len(job_metadata().get('allocated_hosts', [])) == 1
        for key in ['parent_files_sha256', 'scientific_sources', 'diagnostic_sources']:
            for path, digest in plan[key].items():
                assert sha(P / path) == digest, 'Bound evidence/source changed: ' + path
        for path, digest in plan['environment_sha256'].items():
            assert environment_fingerprint(P / path) == digest, 'Runtime changed: ' + path
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
                   MAMBA_ROOT_PREFIX=str(P / '.hpc/mamba'))
        command = [str(P / '.tools/bin/micromamba'), 'run', '-p', str(P / plan['native_prefix']),
                   'python', 'hpc/proof_benchmark.py', '--spec', str((run / 'plan.json').relative_to(P)),
                   '--directory', str(run.relative_to(P))]
        event(run, 'stage_started', phase='proof_benchmark', argv=command)
        with (run / 'command.log').open('x') as log:
            child = subprocess.Popen(command, cwd=P, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                rows = [tuple(map(int, line.split())) for line in subprocess.check_output(
                    ['ps', '-axo', 'pid=,ppid=,rss='], text=True).splitlines() if line.strip()]
                tree = {child.pid}
                while True:
                    found = {pid for pid, parent, rss in rows if parent in tree}
                    if found <= tree:
                        break
                    tree.update(found)
                rss = sum(rss * 1024 for pid, parent, rss in rows if pid in tree)
                peak = max(peak, rss)
                mem = {line.split(':')[0]: int(line.split()[1]) * 1024
                       for line in Path('/proc/meminfo').read_text().splitlines()}
                reason = memory_reason(mem.get('MemAvailable', mem.get('MemFree', 0)),
                                       rss, plan['memory_gb'], plan['host_free_reserve_gib'])
                if interrupted:
                    reason = 'Scheduler/user interruption'
                if shutil.disk_usage(P).free < 100 * 2**30:
                    reason = '100 GiB free-disk reserve'
                if time.monotonic() - started >= plan['outer_seconds']:
                    reason = 'Declared two-hour benchmark budget'
                if (run / 'result.json').exists():
                    active = json.loads((run / 'result.json').read_text()).get('active_case')
                    if active and time.monotonic() - active['started_monotonic'] > plan['case_seconds']:
                        reason = 'Declared 900-second single-case budget'
                event(run, 'resource_sample', rss_bytes=rss, host_available_bytes=mem.get('MemAvailable'),
                      elapsed_seconds=time.monotonic() - started)
                if reason:
                    break
                time.sleep(5)
    except BaseException as exc:
        reason = f'{type(exc).__name__}: {exc}'
    finally:
        if child is not None:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
            code = child.wait()
        result = json.loads((run / 'result.json').read_text()) if (run / 'result.json').exists() else {}
        complete = code == 0 and reason is None and result.get('complete') and result.get('passed')
        elapsed = time.monotonic() - started
        summary = dict(complete=bool(complete), proof_benchmark_passed=bool(complete),
                       reference_continuation_allowed=False, physical_new_state_generated=False,
                       stop_reason=reason or result.get('failure'), return_code=code,
                       wall_seconds=elapsed, sampled_peak_RSS_bytes=peak,
                       cumulative_seconds=plan['cumulative_seconds_before'] + elapsed,
                       worker_summary=result.get('worker_summary', []),
                       worker_profile=result.get('worker_profile'),
                       scope=plan['scope'], next_gate='Review timings and exact proof outcomes before a separately '
                            'qualified solver change; no automatic continuation or full-cycle validation implied')
        atomic(run / 'summary.json', summary)
        state.update(status='diagnostic_complete' if complete else 'diagnostic_incomplete',
                     cumulative_seconds=summary['cumulative_seconds'],
                     blockers=['Full reference and known-field solve qualification remain blocked'])
        atomic(run / 'state.json', state)
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix in ['.json', '.prof', '.log'] and path.name not in ['pbs.log', 'manuscript-state.json']:
                artifact(P, run, path, 'Proof benchmark evidence; no new physical state or solver qualification')
        event(run, 'job_finished', status=state['status'], pbs_job_id=os.environ['PBS_JOBID'])
        render(P, run)
    return 0 if complete else 2


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['submit', 'execute'])
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--parent-run', default='cmbbe-profile-01')
    args = ap.parse_args()
    for value in [args.run_id, args.parent_run]:
        assert re.fullmatch('[A-Za-z0-9_-]+', value), 'Invalid run ID'
    if args.action == 'submit':
        with (P / '.hpc/profile-submission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            submit(args)
    else:
        raise SystemExit(execute(args))
