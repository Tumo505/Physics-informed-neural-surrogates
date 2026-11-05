"""One distinct, bounded known-equation diagnostic after the preserved PBS timeout.

This does not resume the failed pipeline or reset its cumulative budget. No
setup job, physical continuation, ML, or dependent jobs are submitted.
"""
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
from registry import atomic, artifact, environment_fingerprint, event, job_metadata, read, sha
from policy import memory_reason
from manuscript import render

OUTER_SECONDS = 3600


def registered(root, parent, path):
    matches = [r for r in read(parent) if r['kind'] == 'artifact'
               and r['path'] == str(path.relative_to(root))]
    assert matches and matches[-1]['sha256'] == sha(path), 'Unregistered or changed artifact: ' + str(path)


def parent_evidence(root, parent):
    root, parent = Path(root), Path(parent)
    plan = json.loads((parent / 'plan.json').read_text())
    state = json.loads((parent / 'state.json').read_text())
    registered(root, parent, parent / 'plan.json')
    assert state['status'] == 'blocked' and state['terminal_rejection']
    assert state['phase'] == 'known_probe' and state['generation_seconds'] == 0
    assert {'platform', 'audit_resume_control'} <= set(state['completed_phases'])
    assert state['attempts']['known_probe'] == 1
    assert any('known_probe exited' in b for b in state['blockers'])
    audit_path = root / state['platform_output']
    registered(root, parent, audit_path)
    audit = json.loads(audit_path.read_text())
    snapshot = json.loads((root / plan['snapshot']).read_text())
    assert sha(root / plan['snapshot']) == plan['snapshot_sha256']
    assert audit['passed'] and audit['complete'] and audit['audit_complete']
    assert audit['checked_steps'] == audit['total_steps'] == snapshot['accepted_steps']
    for key in ['known_start_checkpoint', 'canonical_checkpoint']:
        registered(root, parent, root / plan[key])
    failed = parent / 'known_probe/attempt-001'
    assert 'Regularized Newton wall budget exhausted' in (failed / 'command.log').read_text()
    trace = json.loads((failed / 'result.json').read_text())
    assert not trace['passed'] and not trace['complete'] and not trace['records']
    assert trace['active_trace']['custom_newton_steps'], 'Require measured iteration evidence'
    elapsed = state['cumulative_seconds']
    assert 0 <= elapsed and elapsed + OUTER_SECONDS < plan['cumulative_pipeline_budget_hours'] * 3600
    paths = [parent / 'state.json', parent / 'plan.json', parent / 'events.jsonl',
             audit_path, failed / 'result.json', failed / 'command.log',
             root / plan['known_start_checkpoint'], root / plan['canonical_checkpoint']]
    return plan, state, {str(p.relative_to(root)): sha(p) for p in paths}


def assert_idle(root):
    result = subprocess.run(['qstat', '-u', os.environ['USER']], cwd=root,
                            capture_output=True, text=True, check=True)
    assert not any(re.match(r'^\s*\d+[.\[]', line) for line in result.stdout.splitlines()), \
        'An existing PBS job is present; do not duplicate work'
    return result.stdout


def submit(args):
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id)
    assert re.fullmatch('[A-Za-z0-9_-]+', args.parent_run)
    parent = P / 'experiments/hpc' / args.parent_run
    source, state, hashes = parent_evidence(P, parent)
    scheduler = assert_idle(P)
    for path, digest in source['scientific_sources'].items():
        assert sha(P / path) == digest, 'Frozen parent source changed: ' + path
    run = P / 'experiments/hpc' / args.run_id
    run.mkdir()  # Duplicate IDs, including failed submissions, are never reused.
    files = {str(p.relative_to(P)): sha(p) for p in [
        P / 'hpc/profile_probe_native.py', P / 'hpc/profile_report.py',
        P / 'hpc/submit_probe_profile.py', P / 'hpc/manuscript.py',
        P / 'hpc/registry.py', P / 'hpc/policy.py']}
    plan = dict(source, run_id=args.run_id, parent_run=args.parent_run,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=P, text=True).strip(),
                diagnostic_only=True, parent_files_sha256=hashes,
                diagnostic_sources=files, cumulative_seconds_before=state['cumulative_seconds'],
                outer_seconds=OUTER_SECONDS, per_equation_seconds=1800,
                reference_continuation_allowed=False, physical_states_to_generate=0,
                scope='Profile only the first known equation through the unchanged frozen solver. '
                      'No original run edits, history advances, qualification waivers or automatic continuation. '
                      'Inclusive timing rows and rank times overlap; profile overhead is included.')
    atomic(run / 'plan.json', plan)
    atomic(run / 'state.json', dict(status='submitted', phase='known_probe_profile',
           completed_phases=[], generation_seconds=0, blockers=[],
           cumulative_seconds=state['cumulative_seconds']))
    event(run, 'diagnostic_prepared', scheduler_output=scheduler, parent_run=args.parent_run,
          carried_cumulative_seconds=state['cumulative_seconds'])
    artifact(P, run, run / 'plan.json', 'Immutable separate diagnostic plan')
    script = run / 'job.pbs'
    script.write_text('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(P)) +
                      '\nexport PATH="$PWD/.hpc/tools/bin:$PATH"\nexec .venv/bin/python '
                      'hpc/submit_probe_profile.py execute --run-id ' + shlex.quote(args.run_id) + '\n')
    command = ['qsub', '-N', 'lv_probe_profile', '-q', source['queue'], '-l',
               f"select=1:ncpus={source['cpus']}:mem={source['memory_gb']}gb",
               '-l', 'walltime=02:00:00', '-l', 'software=' + source['software'],
               '-l', 'place=excl', '-j', 'oe', '-o', str(run / 'pbs.log')]
    if source.get('account'):
        command += ['-A', source['account']]
    result = subprocess.run(command + [str(script)], cwd=P, capture_output=True, text=True)
    event(run, 'job_submission', argv=command + [str(script)], return_code=result.returncode,
          stdout=result.stdout.strip(), stderr=result.stderr.strip())
    if result.returncode:
        atomic(run / 'state.json', dict(status='submission_failed', phase='known_probe_profile',
              blockers=[result.stderr.strip()], cumulative_seconds=state['cumulative_seconds']))
        raise SystemExit(result.stderr)
    print('Diagnostic job:', result.stdout.strip())
    print('One diagnostic only; no pipeline restart. Results:', run.relative_to(P) / 'summary.json')


def execute(args):
    assert os.environ.get('PBS_JOBID'), 'Compute-node PBS execution required'
    run = P / 'experiments/hpc' / args.run_id
    lock = (run / 'execution.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    host_lock = (P / '.hpc/profile-execution.lock').open('a')
    fcntl.flock(host_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((run / 'plan.json').read_text())
    registered(P, run, run / 'plan.json')
    assert not (run / 'summary.json').exists(), 'Never retry a diagnostic output'
    state = json.loads((run / 'state.json').read_text())
    assert state['status'] == 'submitted', 'Previously started diagnostic cannot be retried'
    started = time.monotonic()
    state.update(status='running')
    atomic(run / 'state.json', state)
    event(run, 'job_started', **job_metadata())
    child = None
    reason = None
    code = None
    peak = 0
    interrupted = []
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1]:
        signal.signal(sig, lambda s, f: interrupted.append(s))
    try:
        assert len(job_metadata().get('allocated_hosts', [])) == 1
        for mapping in ['parent_files_sha256', 'scientific_sources', 'diagnostic_sources']:
            for path, digest in plan[mapping].items():
                assert sha(P / path) == digest, 'Bound input/source changed: ' + path
        for path, digest in plan['environment_sha256'].items():
            assert environment_fingerprint(P / path) == digest, 'Runtime changed: ' + path
        native = P / plan['native_prefix']
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   HYDRA_LAUNCHER='fork', FI_PROVIDER='tcp', CUDA_VISIBLE_DEVICES='',
                   MAMBA_ROOT_PREFIX=str(P / '.hpc/mamba'), XDG_CACHE_HOME=str(P / '.hpc/cache'),
                   DIJITSO_CACHE_DIR=str(P / '.hpc/fenics-cache'),
                   MPLCONFIGDIR=str(P / '.hpc/cache/matplotlib'))
        env.pop('FI_TCP_IFACE', None)
        command = [str(P / '.tools/bin/micromamba'), 'run', '-p', str(native),
                   'mpiexec', '-n', str(plan['ranks']), 'python', 'hpc/profile_probe_native.py',
                   '--spec', str((run / 'plan.json').relative_to(P)),
                   '--directory', str(run.relative_to(P))]
        event(run, 'stage_started', phase='known_probe_profile', argv=command)
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
                    reason = 'Declared diagnostic outer budget'
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
        elapsed = time.monotonic() - started
        ranks = []
        for path in sorted(run.glob('rank-*.json')):
            report = json.loads(path.read_text())
            ranks.append(dict(rank=report['rank'], status=report['status'],
                              wall_seconds=report['wall_seconds'], error=report['error'],
                              top_functions=report['functions'][:25]))
        observed = len(ranks) == plan['ranks'] and not reason and all(
            r['status'] in ['solver_timeout', 'first_known_equation_checked'] for r in ranks)
        trace_path = run / 'trace.json'
        trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
        steps = trace.get('active_trace', {}).get('custom_newton_steps', [])
        last_accepted = [trial['native_force_N'] for step in steps
                         if step.get('accepted_custom_update')
                         for trial in step['trials'][-1:] if trial.get('whole_element_admissible')]
        summary = dict(diagnostic_complete=observed, passed=False,
                       reference_continuation_allowed=False, known_pair_qualified=False,
                       physical_new_state_generated=False, return_code=code, stop_reason=reason,
                       rank_reports_received=len(ranks), expected_ranks=plan['ranks'],
                       wall_seconds=elapsed, sampled_peak_RSS_bytes=peak,
                       cumulative_seconds=plan['cumulative_seconds_before'] + elapsed,
                       recorded_custom_iterations=len(steps),
                       recorded_linear_seconds=sum(s.get('linear_seconds', 0) for s in steps),
                       last_recorded_accepted_force_N=last_accepted[-1] if last_accepted else None,
                       terminal_force_N=None,
                       force_scope='Last persisted accepted force is not the terminal residual; '
                                   'no extra failure-state force assembly is performed.',
                       ranks=ranks, next_gate='Review per-rank measured bottleneck; no automatic retry or budget increase')
        atomic(run / 'summary.json', summary)
        state.update(status='diagnostic_complete' if observed else 'diagnostic_incomplete',
                     cumulative_seconds=summary['cumulative_seconds'],
                     blockers=['Diagnostic only; known-pair and independent qualification remain blocked'])
        atomic(run / 'state.json', state)
        for path in sorted(run.iterdir()):
            if path.suffix in ['.json', '.npz', '.prof', '.log'] and path.name not in ['manuscript-state.json', 'pbs.log']:
                artifact(P, run, path, 'Preserved diagnostic evidence; no reference qualification implied')
        event(run, 'job_finished', status=state['status'], pbs_job_id=os.environ['PBS_JOBID'])
        render(P, run)
    return 0 if observed else 2


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['submit', 'execute'])
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--parent-run', default='cmbbe-pbs-05')
    args = ap.parse_args()
    assert re.fullmatch('[A-Za-z0-9_-]+', args.run_id), 'Invalid diagnostic run ID'
    if args.action == 'submit':
        with (P / '.hpc/profile-submission.lock').open('a') as submission_lock:
            fcntl.flock(submission_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            submit(args)
    else:
        raise SystemExit(execute(args))
