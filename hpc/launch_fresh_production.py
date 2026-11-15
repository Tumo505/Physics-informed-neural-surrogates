"""Submit one checkpointed PBS chain after fresh-Jacobian restart qualification."""
import argparse
import fcntl
import math
from pathlib import Path
import re
import shlex
import subprocess

from qualify_reference_restart import ROOT, HPC, check_checkpoint, load
from registry import artifact, atomic, event, read, sha
from submit_probe_profile import assert_idle, registered

DRIVER = 'hpc/run_worker8.py'
SLICE_HOURS = 24
SLICES = 8
FINAL_RESERVE_SECONDS = 34 * 3600


def throughput_limit(plan, state, remaining):
    seconds = min(plan['native_generation_budget_hours'] * 3600 - state['generation_seconds'],
                  plan['cumulative_pipeline_budget_hours'] * 3600 - state['cumulative_seconds']
                  - FINAL_RESERVE_SECONDS)
    assert seconds > 0, 'Insufficient budget after the final audit and mesh comparison reserve'
    return seconds / remaining


def prepare(run, qualification):
    registered(ROOT, qualification, qualification / 'plan.json')
    qualified = load(qualification / 'plan.json')
    state_path = qualification / 'state.json'
    state = load(state_path)
    assert qualified['fresh_jacobian'] and (qualified['start'], qualified['stop']) == (552, 554)
    assert state['status'] == 'qualification_complete' and state['restart_verified']
    assert state['completed_steps'] == 554 and not state['blockers']
    first, second = state['measurements']
    assert (first['state'], second['state']) == (553, 554)
    assert first['job_id'] != second['job_id']
    assert first['input_checkpoint_sha256'] == qualified['start_restart_sha256']
    assert first['output_checkpoint_sha256'] == second['input_checkpoint_sha256']
    accepted_jobs = {item['pbs_job_id'] for item in read(qualification)
                     if item['kind'] == 'job_finished' and item['status'] in
                     ('awaiting_restart', 'qualification_complete')}
    assert {first['job_id'], second['job_id']} <= accepted_jobs
    target = ROOT / 'data/processed' / qualified['target_name']
    manifest, digest = check_checkpoint(target, state['completed_steps'])
    assert digest == second['output_checkpoint_sha256']
    assert sha(target / 'manifest.json') == second['output_manifest_sha256']
    assert not any((target / name).exists() for name in ('failure.json', 'UNACCEPTED_iterate.npz'))
    for index in (552, 553):
        for name in (f'step_journal/{index:06d}.json', f'custom_trace_{index:06d}.json'):
            registered(ROOT, qualification, target / name)
        trace = load(target / f'custom_trace_{index:06d}.json')
        assert trace['solver_policy'] == 'fresh_jacobian_two_good_steps'
        assert 0 <= trace['custom_force_N'] < 1e-9
    parent = HPC / qualified['parent_run']
    registered(ROOT, parent, parent / 'plan.json')
    inherited, prior_state = load(parent / 'plan.json'), load(parent / 'state.json')
    assert inherited['target_name'] == qualified['target_name']
    assert prior_state['stream_protocol'] == qualified['stream_protocol']
    phases = ['native_preflight', 'audit_resume_control', 'platform',
              'known_probe', 'known_probe_independent', 'seed', 'seed_audit']
    assert set(phases) <= set(prior_state['completed_phases'])
    remaining = manifest['n_steps'] - state['completed_steps']
    allowed = throughput_limit(qualified, state, remaining)
    times = [row['wall_seconds'] for row in (first, second)]
    assert all(math.isfinite(value) and value > 0 for value in times)
    measured = sum(times) / len(times)
    assert state['measured_pace_within_budget'] and measured <= allowed, (
        f'Qualification averaged {measured:.1f} s/state. Production allows {allowed:.1f} s/state '
        'after reserving 34 hours for final checks. No jobs submitted.')

    frozen = dict(qualified['frozen'])
    # This successor uses a new driver. All other inherited inputs retain their hashes.
    frozen.pop(DRIVER, None)
    for path, digest in frozen.items():
        assert sha(ROOT / path) == digest, 'Inherited qualification input changed: ' + path
    for path in (qualification / 'plan.json', state_path, qualification / 'events.jsonl',
                 ROOT / DRIVER, Path(__file__).resolve(),
                 ROOT / 'hpc/production_fresh_native.py',
                 ROOT / 'scripts/finalize_parallel_reference.py',
                 ROOT / 'scripts/compare_quadratic_references.py'):
        frozen[str(path.relative_to(ROOT))] = sha(path)
    plan = dict(inherited, run_id=run.name, queue='nvidia', fresh_production=True,
                parent_run=parent.name, qualification_run=qualification.name,
                final_reserve_seconds=FINAL_RESERVE_SECONDS,
                scientific_sources={DRIVER: frozen[DRIVER]}, inherited_evidence_sha256=frozen,
                scope='Checkpointed fresh-Jacobian reference production and final numerical audits.')
    state = dict(status='submitted', phase='reference', completed_phases=phases, attempts={},
                 checkpoints_verified=True, checkpoint_sha256=second['output_checkpoint_sha256'],
                 manifest_sha256=second['output_manifest_sha256'], completed_steps=554,
                 n_steps=manifest['n_steps'], stream_protocol=qualified['stream_protocol'],
                 cumulative_seconds=state['cumulative_seconds'], generation_seconds=state['generation_seconds'],
                 measured_seconds_per_state=measured, allowed_seconds_per_state=allowed, blockers=[])
    return plan, state


def submit(run, plan, state):
    run.mkdir()
    atomic(run / 'plan.json', plan)
    artifact(ROOT, run, run / 'plan.json', 'Production reference plan')
    atomic(run / 'state.json', state)
    script = run / 'job.pbs'
    argv = ['.venv/bin/python', DRIVER, '--run', str(run.relative_to(ROOT)),
            '--slice-seconds', str(SLICE_HOURS * 3600)]
    script.write_text('#!/bin/bash\nset -eu\ncd ' + shlex.quote(str(ROOT)) + '\nexec ' + shlex.join(argv) + '\n')
    dependency = None
    for index in range(1, SLICES + 1):
        cmd = ['qsub', '-N', 'lv_fresh_ref', '-q', plan['queue'], '-l',
               f"select=1:ncpus={plan['cpus']}:mem={plan['memory_gb']}gb",
               '-l', f'walltime={SLICE_HOURS}:00:00', '-l', 'place=excl',
               '-l', 'software=' + plan['software'], '-j', 'oe',
               '-o', str(run / f'pbs-{index:03d}.log')]
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
        print(f'Slice {index}: {dependency}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', default='cmbbe-fresh-restart-552-01')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    for value in (args.qualification, args.run_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('Invalid run ID')
    with (ROOT / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(ROOT)
        run = HPC / args.run_id
        submit(run, *prepare(run, HPC / args.qualification))
