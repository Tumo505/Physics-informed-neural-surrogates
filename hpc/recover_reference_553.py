"""Recover the first state of a timed-out fresh-Jacobian qualification."""
import argparse
import fcntl
from pathlib import Path
import re
import zipfile

import numpy as np

from probe_regularization_recovery import P, checkpointed_recovery, load, queue_recovery
from qualify_reference_restart import check_checkpoint, TIMEOUT
from recover_reference_552 import recovery_plan
from registry import sha
from submit_probe_profile import assert_idle, registered

HPC = P / 'experiments/hpc'


def prepare(run, qualification):
    plan_path = qualification / 'plan.json'
    state_path = qualification / 'state.json'
    registered(P, qualification, plan_path)
    previous, state = load(plan_path), load(state_path)
    step = previous['start']
    log = qualification / f'command-{step + 1}.log'
    registered(P, qualification, log)
    evidence = [plan_path, state_path, log]
    assert previous['fresh_jacobian'] and 552 <= step <= 1598 and previous['stop'] == step + 2
    assert state['status'] == 'blocked' and state['terminal_rejection'] and not state['measurements']
    assert state['phase'] == f'state_{step + 1}' and state['completed_steps'] == step
    assert f'RuntimeError: {TIMEOUT}' in log.read_text()
    target = P / 'data/processed' / previous['target_name']
    manifest, digest = check_checkpoint(target, step)
    assert digest == previous['start_restart_sha256']
    assert sha(target / 'manifest.json') == previous['start_manifest_sha256']
    trace_path = target / f'custom_trace_{step:06d}.json'
    trace = load(trace_path)
    assert trace['step'] == step and trace['solver_policy'] == 'fresh_jacobian_two_good_steps'
    rows = [row for row in trace['custom_newton_steps'] if row.get('accepted_custom_update')]
    assert len(rows) >= 2
    forces = [row['trials'][-1]['native_force_N'] for row in rows[-2:]]
    assert forces[0] > forces[1] > 1e-9

    iterate = target / 'UNACCEPTED_iterate.npz'
    if iterate.exists():
        # A legacy abort may leave an incomplete ZIP or a complete file without shift metadata.
        if zipfile.is_zipfile(iterate):
            with zipfile.ZipFile(iterate) as archive:
                assert 'u.npy' not in archive.namelist(), 'Saved displacement exists and requires review'
        evidence.append(iterate)
    frozen = dict(previous['frozen'])
    # The production supervisor changed after qualification and is outside this recovery path.
    frozen.pop('hpc/run_worker8.py', None)
    recovery_driver = P / 'hpc/probe_regularization_recovery.py'
    updated_sources = {recovery_driver, Path(__file__).resolve()}
    for path, expected in frozen.items():
        if P / path not in updated_sources:
            assert sha(P / path) == expected, 'Changed inherited input: ' + path
    for path in (*evidence, trace_path, target / 'manifest.json', target / 'restart.npz',
                 target / f'step_journal/{step - 2:06d}.json', target / f'step_journal/{step - 1:06d}.json',
                 recovery_driver, Path(__file__).resolve()):
        frozen[str(path.relative_to(P))] = sha(path)
    elapsed, generation = state['cumulative_seconds'], state['generation_seconds']
    assert elapsed + 3600 <= previous['cumulative_pipeline_budget_hours'] * 3600
    assert generation + 3600 <= previous['native_generation_budget_hours'] * 3600
    return recovery_plan(previous, run_id=run.name, step=step, frozen=frozen,
                         queue='nvidia', checkpointed_recovery=True, checkpointed_recovery_553=step == 552,
                         failed_qualification=qualification.name, full_solve=True, log_phases=True,
                         solver_seconds=1800, regularization_policy='two_good_steps',
                         reuse_jacobian=False, reuse_gamg=False, late_forcing_floor=1e-5,
                         cumulative_seconds_before=elapsed, generation_seconds_before=generation,
                         scope=f'One state-{step + 1} recovery from accepted state {step}. Atomically preserve '
                               'each accepted Newton iterate. Independently audit a converged candidate.')


def prepare_resume(run, qualification, recovery):
    plan = prepare(run, qualification)
    step = plan['step']
    evidence = [recovery / name for name in ('plan.json', 'summary.json', 'last-newton.npz')]
    for path in evidence:
        registered(P, recovery, path)
    previous, summary = map(load, evidence[:2])
    assert checkpointed_recovery(previous) and not previous.get('saved_iterate')
    assert previous['failed_qualification'] == qualification.name
    assert previous['step'] == summary['step'] == step
    assert summary['complete'] and not summary['passed'] and TIMEOUT in summary['stop_reason']
    assert not summary['physical_new_state_generated'] and not summary['reference_continuation_allowed']
    for name in ('manifest.json', 'restart.npz'):
        path = f"data/processed/{plan['target_name']}/{name}"
        assert previous['frozen'][path] == plan['frozen'][path]
    saved = evidence[-1]
    assert sha(saved) == summary['last_newton_iterate_sha256']
    with np.load(saved, allow_pickle=False) as data:
        assert int(data['step']) == step and not bool(data['accepted'])
        force, shift, iteration = float(data['force_N']), float(data['next_shift']), int(data['iteration'])
        assert np.isfinite(data['u']).all()
    assert np.isfinite(force) and 1e-9 < force == summary['last_accepted_custom_force_N']
    options = load(P / plan['stream_protocol'])['admissible_newton']
    assert options['minimum_regularization'] <= shift <= options['maximum_regularization']
    trace = recovery / 'trace.json'
    record = load(trace)
    assert record['step'] == step
    rows = [row for row in record['custom_newton_steps'] if row.get('accepted_custom_update')]
    assert len(rows) == summary['accepted_custom_iterations'] and len(rows) >= 5
    assert rows[-1]['iteration'] == iteration and rows[-1]['next_regularization'] == shift
    trials = [row['trials'][-1] for row in rows[-5:]]
    assert all(trial['whole_element_admissible'] and trial['armijo_passed'] for trial in trials)
    forces = [trial['native_force_N'] for trial in trials]
    assert all(np.isfinite(value) for value in forces) and forces[-1] == force
    assert all(left > right for left, right in zip(forces, forces[1:]))
    elapsed, wall = summary['cumulative_seconds'], summary['wall_seconds']
    assert np.isfinite(wall) and wall > 0 and elapsed > plan['cumulative_seconds_before']
    generation = previous['generation_seconds_before'] + wall
    assert elapsed + 3600 <= plan['cumulative_pipeline_budget_hours'] * 3600
    assert generation + 3600 <= plan['native_generation_budget_hours'] * 3600
    for path in (*evidence, trace):
        plan['frozen'][str(path.relative_to(P))] = sha(path)
    return dict(plan, cumulative_seconds_before=elapsed, generation_seconds_before=generation,
                saved_iterate=str(saved.relative_to(P)), saved_iterate_force_N=force,
                warm_shift=shift, saved_iterate_iteration=iteration, recovery_parent=recovery.name,
                prior_final_force_trend_N=forces,
                scope=f'Resume the verified state-{step + 1} Newton iterate with fresh Jacobians. '
                      'Independently audit a converged candidate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', default='cmbbe-fresh-restart-552-01')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--resume-after')
    args = parser.parse_args()
    for value in filter(None, (args.qualification, args.run_id, args.resume_after)):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('Invalid run ID')
    with (P / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(P)
        run = HPC / args.run_id
        qualification = HPC / args.qualification
        plan = (prepare_resume(run, qualification, HPC / args.resume_after)
                if args.resume_after else prepare(run, qualification))
        queue_recovery(run, plan)
