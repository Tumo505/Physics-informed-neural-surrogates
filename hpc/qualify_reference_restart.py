"""Qualify production continuation from an admitted checkpoint in two PBS jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time

import numpy as np

from policy import memory_reason
from registry import artifact, atomic, environment_fingerprint, event, job_metadata, sha
from submit_probe_profile import assert_idle, registered

ROOT = Path(__file__).resolve().parents[1]
HPC = ROOT / 'experiments/hpc'
START = 551
PBS_SECONDS = 3600
WORK_SECONDS = 3000
TIMEOUT = 'Regularized Newton wall budget exhausted'
FAILURE_FILES = ('failure.json', 'UNACCEPTED_iterate.npz')


def load(path):
    return json.loads(path.read_text())


def check_checkpoint(target, count):
    manifest = load(target / 'manifest.json')
    assert manifest['completed_steps'] == len(manifest['steps']) == count
    assert manifest['n_steps'] == 1600 and not manifest['complete']
    restart = target / 'restart.npz'
    journal = load(target / 'step_journal' / f'{count - 1:06d}.json')
    digest = sha(restart)
    assert digest == journal['accepted_checkpoint_sha256']
    assert manifest['steps'][-1] == journal['record']
    with np.load(restart) as checkpoint:
        assert int(checkpoint['completed_steps']) == count
    return manifest, digest


def failure_evidence(parent, target, state):
    phase = state['phase']
    assert phase == 'reference_000547' and state['attempts'][phase] == 1
    log = parent / phase / 'attempt-001/command.log'
    registered(ROOT, parent, log)
    assert f'RuntimeError: {TIMEOUT}' in log.read_text(), 'Expected registered Newton timeout'
    marker = target / 'failure.json'
    if marker.exists():
        assert TIMEOUT in load(marker)['error'], 'A different failure marker requires review'
    return log, {name: sha(target / name) for name in FAILURE_FILES if (target / name).exists()}


def submit(run, recovery, fresh=False):
    with (ROOT / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(ROOT)
        prepare_and_submit(run, recovery, fresh)


def prepare_and_submit(run, recovery, fresh=False):
    evidence = [recovery / name for name in ('plan.json', 'summary.json', 'independent-audit.json')]
    for path in evidence:
        registered(ROOT, recovery, path)
    source, summary, audit = map(load, evidence)
    assert summary['complete'] and summary['passed'] and audit['passed'] and audit['history_passed']
    parent = HPC / source['parent_run']
    registered(ROOT, parent, parent / 'plan.json')
    parent_state = load(parent / 'state.json')
    assert parent_state['status'] == 'blocked' and parent_state['terminal_rejection']
    target = ROOT / 'data/processed' / source['target_name']
    start = source['step'] + 1 if fresh else START
    if fresh:
        assert start in (552, 553), 'Fresh qualification requires admitted state 552 or 553'
    assert source['step'] == summary['step'] == audit['step'] == start - 1
    manifest, digest = check_checkpoint(target, start)
    admission = manifest['steps'][-1]['recovery_admission']
    assert admission == dict(run=str(recovery.relative_to(ROOT)),
                             candidate_sha256=audit['candidate_sha256'],
                             independent_audit_sha256=sha(evidence[-1]))
    protocol_path = ROOT / parent_state['stream_protocol']
    protocol = load(protocol_path)
    assert protocol['worker_policy']['implementation'] == 'cofactor'
    assert protocol['config'] == source['config']
    assert protocol['per_step_wall_seconds'] == 1800
    assert protocol['parallel_ranks'] == source['ranks'] == 12
    assert {start + 1, start + 2} <= set(protocol['allowed_qualification_stops'])
    generation = max(parent_state['generation_seconds'], manifest['cumulative_native_wall_seconds'])
    if fresh:
        if start == 552:
            assert source['warm_restart_552'] and source['warm_followup_parent']
            failed = HPC / source['failed_restart_run']
        else:
            assert source['checkpointed_recovery_553'] and source['recovery_parent'] and source['saved_iterate']
            failed = HPC / source['failed_qualification']
            failed_state = load(failed / 'state.json')
            assert failed_state['status'] == 'blocked' and failed_state['terminal_rejection']
            assert failed_state['phase'] == 'state_553' and failed_state['completed_steps'] == 552
        assert not source['reuse_jacobian'] and not source['reuse_gamg']
        assert source['regularization_policy'] == 'two_good_steps' and source['late_forcing_floor'] == 1e-5
        assert summary['proof_agreement'] and 0 <= summary['native_force_N'] < 1e-9
        assert summary['native_updates'] == 0 and summary['SNES_reason'] > 0
        assert 0 <= audit['native_force_N'] < 1e-8
        failure_log = failed / f'command-{start}.log'
        registered(ROOT, failed, failure_log)
        assert f'RuntimeError: {TIMEOUT}' in failure_log.read_text()
        marker = target / 'failure.json'
        if marker.exists():
            assert TIMEOUT in load(marker)['error'], 'A different failure marker requires review'
        archived_failure = {name: sha(target / name) for name in FAILURE_FILES if (target / name).exists()}
        # The successor binds the newly admitted checkpoint and this launcher version.
        replaced = {str(path.relative_to(ROOT)) for path in
                    (target / 'manifest.json', target / 'restart.npz', Path(__file__).resolve())}
        for name in FAILURE_FILES:
            key = str((target / name).relative_to(ROOT))
            if key in source['frozen']:
                assert archived_failure.get(name) == source['frozen'][key]
                replaced.add(key)
        frozen = {key: value for key, value in source['frozen'].items() if key not in replaced}
        for path in ('hpc/fresh_reference_native.py',
                     'src/lv_surrogate/physics/admissible_newton_gentle_shrink.py',
                     'src/lv_surrogate/physics/regularization_recovery_policy.py'):
            frozen[path] = sha(ROOT / path)
        if start == 553:
            entry = 'hpc/production_fresh_native.py'
            frozen[entry] = sha(ROOT / entry)
        generation = max(generation, source['generation_seconds_before'] + summary['wall_seconds'])
    else:
        assert source['per_step_seconds'] == 1800
        failure_log, archived_failure = failure_evidence(parent, target, parent_state)
        frozen = dict(source['scientific_sources'], **source['inherited_evidence_sha256'])
    assert sha(protocol_path) == source['frozen'][str(protocol_path.relative_to(ROOT))]
    for path in (*evidence, failure_log, protocol_path, ROOT / source['config'], target / 'inputs.npz',
                 parent / 'state.json', Path(__file__).resolve()):
        frozen[str(path.relative_to(ROOT))] = sha(path)
    assert manifest['config_sha256'] == frozen[source['config']]
    assert manifest['source_inputs_sha256'] == frozen[str((target / 'inputs.npz').relative_to(ROOT))]
    elapsed = summary['cumulative_seconds'] + audit['wall_seconds']
    assert elapsed + 2 * PBS_SECONDS <= source['cumulative_pipeline_budget_hours'] * 3600
    assert generation + 2 * PBS_SECONDS <= source['native_generation_budget_hours'] * 3600

    run.mkdir()
    plan = dict(source, frozen=frozen, run_id=run.name, start=start, stop=start + 2, fresh_jacobian=fresh,
                parent_run=parent.name, recovery_run=recovery.name, queue='nvidia',
                stream_protocol=str(protocol_path.relative_to(ROOT)), start_restart_sha256=digest,
                start_manifest_sha256=sha(target / 'manifest.json'),
                archived_failure=archived_failure,
                generation_entry='hpc/production_fresh_native.py' if start == 553
                else 'hpc/fresh_reference_native.py' if fresh else 'hpc/worker8_native.py',
                scope='Two production states through separate PBS jobs. The inherited terminal run '
                      'is retained. Production beyond this restart and timing qualification requires review.')
    atomic(run / 'plan.json', plan)
    artifact(ROOT, run, run / 'plan.json', 'Restart qualification plan')
    atomic(run / 'state.json', dict(status='submitted', phase='restart_qualification',
                                  completed_steps=start, cumulative_seconds=elapsed,
                                  generation_seconds=generation, measurements=[], blockers=[]))
    dependency = None
    for count in range(start, start + 2):
        script = run / f'state-{count + 1}.pbs'
        invocation = ['.venv/bin/python', 'hpc/qualify_reference_restart.py', 'execute',
                      '--run-id', run.name, '--start', str(count)]
        script.write_text('#!/bin/bash\nset -eu\ncd ' + shlex.quote(str(ROOT)) + '\n'
                          'exec ' + shlex.join(invocation) + '\n')
        cmd = ['qsub', '-N', 'lv_restart', '-q', plan['queue'], '-l',
               f"select=1:ncpus={plan['cpus']}:mem={plan['memory_gb']}gb",
               '-l', 'walltime=01:00:00', '-l', 'place=excl', '-l', 'software=' + plan['software'],
               '-j', 'oe', '-o', str(run / f'pbs-{count + 1}.log')]
        if plan.get('account'):
            cmd += ['-A', plan['account']]
        if dependency:
            cmd += ['-W', 'depend=afterok:' + dependency]
        reply = subprocess.run(cmd + [str(script)], cwd=ROOT, capture_output=True, text=True)
        event(run, 'job_submission', argv=cmd + [str(script)], return_code=reply.returncode,
              stdout=reply.stdout.strip(), stderr=reply.stderr.strip())
        if reply.returncode:
            raise SystemExit('Submission stopped. Preserve submitted jobs and inspect the ledger: ' + reply.stderr)
        dependency = reply.stdout.strip()
        assert re.fullmatch(r'[\w.-]+', dependency)
        print(f'State {count + 1}: {dependency}', flush=True)


def check_resources(plan, pid):
    rows = [tuple(map(int, row.split())) for row in subprocess.check_output(
        ['ps', '-axo', 'pid=,ppid=,rss='], text=True, env={'PATH': '/usr/bin:/bin'}).splitlines()]
    tree = {pid}
    while True:
        descendants = tree | {child for child, parent, _ in rows if parent in tree}
        if descendants == tree:
            break
        tree = descendants
    rss = 1024 * sum(size for child, _, size in rows if child in tree)
    available = next(int(row.split()[1]) * 1024 for row in Path('/proc/meminfo').read_text().splitlines()
                     if row.startswith('MemAvailable:'))
    reason = memory_reason(available, rss, plan['memory_gb'], plan['host_free_reserve_gib'])
    assert reason is None, reason
    assert shutil.disk_usage(ROOT).free >= 100 * 2**30, '100 GiB free-disk reserve'
    return rss


def advance(run, plan, count, started):
    prefix = ROOT / plan['native_prefix']
    env = os.environ.copy()
    for name in ('CONDARC', 'MAMBARC', 'CONDA_PREFIX', 'CONDA_DEFAULT_ENV', 'CONDA_SHLVL',
                 'BASH_ENV', 'ENV', 'FI_TCP_IFACE', 'PKG_CONFIG_SYSROOT_DIR'):
        env.pop(name, None)
    pkgconfig = f'{prefix}/lib/pkgconfig:{prefix}/share/pkgconfig'
    env.update(PATH=f'{prefix}/bin:{ROOT}/.hpc/tools/bin:/usr/bin:/bin',
               PKG_CONFIG=str(prefix / 'bin/pkg-config'), PKG_CONFIG_PATH=pkgconfig,
               PKG_CONFIG_LIBDIR=pkgconfig, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', HYDRA_LAUNCHER='fork',
               FI_PROVIDER='tcp', CUDA_VISIBLE_DEVICES='', XDG_CACHE_HOME=str(ROOT / '.hpc/cache'),
               MPLCONFIGDIR=str(ROOT / '.hpc/cache/matplotlib'),
               DIJITSO_CACHE_DIR=str(ROOT / '.hpc/fenics-cache'))
    protocol = plan['stream_protocol']
    entry = plan.get('generation_entry',
                     'hpc/fresh_reference_native.py' if plan.get('fresh_jacobian') else 'hpc/worker8_native.py')
    cmd = [str(ROOT / '.tools/bin/micromamba'), '--no-rc', 'run', '-r', str(ROOT / '.hpc/mamba'),
           '-p', str(prefix), 'mpiexec', '-n', str(plan['ranks']), 'python',
           entry, 'generate', '--policy', protocol, '--protocol', protocol,
           '--qualification-stop', str(count + 1)]
    log = run / f'command-{count + 1}.log'
    event(run, 'stage_started', argv=cmd, **job_metadata())
    interrupted = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda signum, _: interrupted.append(signum))
    peak = 0
    try:
        with log.open('x') as output:
            with subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=output,
                                  stderr=subprocess.STDOUT, start_new_session=True) as child:
                try:
                    while child.poll() is None:
                        assert not interrupted, 'PBS or user interruption'
                        assert time.monotonic() - started < WORK_SECONDS, 'Qualification wall budget exhausted'
                        peak = max(peak, check_resources(plan, child.pid))
                        time.sleep(2)
                    assert child.returncode == 0, f'Generator exited {child.returncode}. Inspect {log}'
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
    finally:
        if log.exists():
            artifact(ROOT, run, log, 'Production generation command output')
    return peak


def execute(run, count):
    assert os.environ.get('PBS_JOBID'), 'Use PBS compute jobs'
    started = time.monotonic()
    registered(ROOT, run, run / 'plan.json')
    plan = load(run / 'plan.json')
    start = plan['start']
    parent = HPC / plan['parent_run']
    target = ROOT / 'data/processed' / plan['target_name']
    with (run / 'execution.lock').open('a') as lock, (parent / 'execution.lock').open('a') as inherited:
        for handle in (lock, inherited):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = run / 'state.json'
        state = load(state_path)
        assert count in (start, start + 1)
        assert state['status'] == ('submitted' if count == start else 'awaiting_restart')
        assert state['completed_steps'] == count
        state.update(status='running', phase=f'state_{count + 1}')
        atomic(state_path, state)
        generation_started = None
        try:
            assert state['cumulative_seconds'] + PBS_SECONDS <= plan['cumulative_pipeline_budget_hours'] * 3600
            assert state['generation_seconds'] + PBS_SECONDS <= plan['native_generation_budget_hours'] * 3600
            assert len(job_metadata().get('allocated_hosts', [])) == 1
            for path, digest in plan['frozen'].items():
                assert sha(ROOT / path) == digest, 'Bound input changed: ' + path
            for prefix, digest in plan['environment_sha256'].items():
                assert environment_fingerprint(ROOT / prefix) == digest, 'Environment changed: ' + prefix
            check_resources(plan, os.getpid())
            _, before = check_checkpoint(target, count)
            if count == start:
                assert before == plan['start_restart_sha256']
                assert sha(target / 'manifest.json') == plan['start_manifest_sha256']
                archive_name = 'pre-fresh-jacobian-qualification' if plan.get('fresh_jacobian') else 'pre-restart-qualification'
                if start == 553:
                    archive_name += '-553'
                archive = target / archive_name
                with (target / 'generation.lock').open('a') as generation_lock:
                    fcntl.flock(generation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    assert {name for name in FAILURE_FILES if (target / name).exists()} == set(plan['archived_failure']), (
                        'Terminal output inventory changed after submission')
                    for name, digest in plan['archived_failure'].items():
                        assert sha(target / name) == digest
                    if plan['archived_failure']:
                        archive.mkdir()
                    for name in plan['archived_failure']:
                        original = target / name
                        original.rename(archive / name)
                        artifact(ROOT, run, archive / name, f'Retained terminal output before state {count + 1} qualification')
            else:
                previous = state['measurements'][-1]
                assert before == previous['output_checkpoint_sha256']
                assert sha(target / 'manifest.json') == previous['output_manifest_sha256']
                assert state['measurements'][-1]['job_id'] != os.environ['PBS_JOBID']
            generation_started = time.monotonic()
            peak = advance(run, plan, count, started)
            manifest, after = check_checkpoint(target, count + 1)
            journal = load(target / 'step_journal' / f'{count:06d}.json')
            assert journal['input_checkpoint_sha256'] == before
            record = journal['record']
            assert record['SNES_reason'] > 0 and record['nonlinear_iterations'] == 0
            assert 0 <= record['residual_l2_N'] < 1e-8 and record['whole_element_positivity_passed']
            if plan.get('fresh_jacobian'):
                trace = target / f'custom_trace_{count:06d}.json'
                assert record['custom_trace_sha256'] == sha(trace)
                trace_result = load(trace)
                assert trace_result['solver_policy'] == 'fresh_jacobian_two_good_steps'
                assert 0 <= trace_result['custom_force_N'] < 1e-9
                artifact(ROOT, run, trace, 'Fresh-Jacobian qualification trace')
            artifact(ROOT, run, target / 'step_journal' / f'{count:06d}.json', 'Durable state acceptance')
            elapsed = time.monotonic() - started
            state['measurements'].append(dict(state=count + 1, job_id=os.environ['PBS_JOBID'],
                                              input_checkpoint_sha256=before, output_checkpoint_sha256=after,
                                              output_manifest_sha256=sha(target / 'manifest.json'),
                                              wall_seconds=elapsed, solve_seconds=record['nonlinear_solve_seconds'],
                                              peak_rss_bytes=peak))
            generation = state['generation_seconds'] + elapsed
            state.update(completed_steps=count + 1, generation_seconds=generation,
                         status='awaiting_restart' if count == start else 'qualification_complete')
            if count == start + 1:
                remaining = manifest['n_steps'] - state['completed_steps']
                generation_left = plan['native_generation_budget_hours'] * 3600 - generation
                pipeline_left = plan['cumulative_pipeline_budget_hours'] * 3600 - state['cumulative_seconds'] - elapsed
                allowance = min(generation_left, pipeline_left) / remaining
                observed = sum(row['wall_seconds'] for row in state['measurements']) / 2
                state.update(restart_verified=True, remaining_states=remaining,
                             allowed_average_seconds=allowance, measured_average_seconds=observed,
                             measured_pace_within_budget=observed <= allowance,
                             production_submitted=False)
        except Exception as exc:
            if generation_started is not None and state['completed_steps'] == count:
                state['generation_seconds'] += time.monotonic() - generation_started
            state.update(status='blocked', blockers=[str(exc)], terminal_rejection=True)
            raise
        finally:
            state['cumulative_seconds'] += time.monotonic() - started
            atomic(state_path, state)
            event(run, 'job_finished', status=state['status'], pbs_job_id=os.environ['PBS_JOBID'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'execute'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--recovery-run')
    parser.add_argument('--fresh-jacobian', action='store_true')
    parser.add_argument('--start', type=int)
    args = parser.parse_args()
    if args.recovery_run is None:
        args.recovery_run = 'cmbbe-warm-552-02' if args.fresh_jacobian else 'cmbbe-recovery-551-resume-01'
    for value in (args.run_id, args.recovery_run):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('Invalid run ID')
    run = HPC / args.run_id
    if args.action == 'submit':
        submit(run, HPC / args.recovery_run, args.fresh_jacobian)
    else:
        execute(run, args.start)
