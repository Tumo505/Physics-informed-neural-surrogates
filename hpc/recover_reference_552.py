"""Register one bounded state-552 recovery from the unchanged admitted checkpoint."""
import argparse
import fcntl
from pathlib import Path
import re

import numpy as np

from probe_regularization_recovery import P, load, queue_recovery
from qualify_reference_restart import check_checkpoint, TIMEOUT
from registry import sha
from submit_probe_profile import assert_idle, registered

HPC = P / 'experiments/hpc'
STEP = 551
SHARED_FIELDS = (
    'account', 'config', 'cpus', 'environment_sha256', 'host_free_reserve_gib',
    'memory_gb', 'native_prefix', 'parent_run', 'ranks', 'reference_config_sha256',
    'snapshot', 'software', 'stream_protocol', 'target_name',
    'cumulative_pipeline_budget_hours', 'native_generation_budget_hours',
)


def recovery_plan(pilot, **fields):
    shared = {name: pilot[name] for name in SHARED_FIELDS if name != 'account'}
    return dict(shared, account=pilot.get('account'), **fields)


def prepare(run, failed, startup_failure=None):
    plan_path = failed / 'plan.json'
    log = failed / 'command-552.log'
    for path in (plan_path, log):
        registered(P, failed, path)
    pilot = load(plan_path)
    state_path = failed / 'state.json'
    state = load(state_path)
    assert state['status'] == 'blocked' and state['terminal_rejection']
    assert state['phase'] == 'state_552' and state['completed_steps'] == STEP
    assert pilot['start'] == STEP and pilot['stop'] == 553 and not state['measurements']
    assert f'RuntimeError: {TIMEOUT}' in log.read_text()
    target = P / 'data/processed' / pilot['target_name']
    manifest, checkpoint_hash = check_checkpoint(target, STEP)
    assert checkpoint_hash == pilot['start_restart_sha256']
    assert sha(target / 'manifest.json') == pilot['start_manifest_sha256']
    assert manifest['steps'][-1]['recovery_admission']['run'] == str(
        (HPC / pilot['recovery_run']).relative_to(P))
    trace_path = target / 'custom_trace_000551.json'
    trace = load(trace_path)
    assert trace['step'] == STEP
    assert any(row.get('accepted_custom_update') for row in trace['custom_newton_steps'])
    protocol = load(P / pilot['stream_protocol'])
    assert protocol['per_step_wall_seconds'] == 1800
    assert protocol['admissible_newton']['custom_absolute_tolerance_N'] == 1e-9

    frozen = dict(pilot['frozen'])
    for path, digest in frozen.items():
        assert sha(P / path) == digest, 'Changed inherited input: ' + path
    for path in (plan_path, state_path, log, trace_path, target / 'manifest.json',
                 target / 'restart.npz', target / 'step_journal/000549.json',
                 target / 'step_journal/000550.json', Path(__file__).resolve(),
                 P / 'hpc/probe_regularization_recovery.py',
                 P / 'src/lv_surrogate/physics/admissible_newton_gentle_shrink.py',
                 P / 'src/lv_surrogate/physics/regularization_recovery_policy.py'):
        frozen[str(path.relative_to(P))] = sha(path)
    elapsed, generation = state['cumulative_seconds'], state['generation_seconds']
    if startup_failure is not None:
        for name in ('plan.json', 'summary.json'):
            path = startup_failure / name
            registered(P, startup_failure, path)
            frozen[str(path.relative_to(P))] = sha(path)
        previous = load(startup_failure / 'plan.json')
        summary = load(startup_failure / 'summary.json')
        assert previous['failed_restart_run'] == failed.name and previous['step'] == STEP
        assert previous['frozen'][str(plan_path.relative_to(P))] == sha(plan_path)
        assert previous['frozen'][str((target / 'restart.npz').relative_to(P))] == checkpoint_hash
        assert previous['screen_and_solve'] and previous['saved_iterate']
        assert summary['complete'] and not summary['passed'] and summary['step'] == STEP
        assert not summary['physical_new_state_generated'] and not summary['reference_continuation_allowed']
        assert summary['accepted_custom_iterations'] == 0 and summary['regularization_policy'] is None
        assert summary['gamg_reuse_attempts'] == 0 and summary['selected_predictor'] is None
        assert 'Mismatched elements' in summary['stop_reason']
        assert summary['cumulative_seconds'] > elapsed
        generation += summary['cumulative_seconds'] - elapsed
        elapsed = summary['cumulative_seconds']
    assert elapsed + 3600 <= pilot['cumulative_pipeline_budget_hours'] * 3600
    assert generation + 3600 <= pilot['native_generation_budget_hours'] * 3600
    return recovery_plan(pilot, run_id=run.name, step=STEP, frozen=frozen, queue='nvidia',
                failed_restart_run=failed.name,
                startup_failure_run=startup_failure.name if startup_failure else None,
                cumulative_seconds_before=elapsed,
                generation_seconds_before=generation, solver_seconds=1800,
                screen_and_solve=True, full_solve=True, log_phases=True,
                reuse_jacobian=True, reuse_gamg=True, regularization_policy='two_good_steps',
                late_forcing_floor=1e-5,
                scope='Screen seven predictors, then solve one state within the original Newton limit. '
                      'A converged candidate receives an independent audit. Reference admission is separate.')


def prepare_damped_screen(run, failed, recovery):
    plan = prepare(run, failed)
    for name in ('plan.json', 'summary.json'):
        registered(P, recovery, recovery / name)
        path = recovery / name
        plan['frozen'][str(path.relative_to(P))] = sha(path)
    previous = load(recovery / 'plan.json')
    summary = load(recovery / 'summary.json')
    assert previous['step'] == summary['step'] == STEP
    assert previous['screen_and_solve'] and previous['failed_restart_run'] == failed.name
    target = P / 'data/processed' / plan['target_name']
    assert previous['target_name'] == plan['target_name']
    for name in ('manifest.json', 'restart.npz'):
        key = str((target / name).relative_to(P))
        assert previous['frozen'][key] == plan['frozen'][key]
    assert summary['complete'] and not summary['passed']
    assert not summary['physical_new_state_generated'] and not summary['reference_continuation_allowed']
    assert TIMEOUT in summary['stop_reason'] and summary['accepted_custom_iterations'] > 0
    assert summary['selected_predictor'] == 'stress extrapolation'
    baseline = next(row for row in summary['predictors'] if row['predictor'] == 'stress extrapolation')
    assert baseline['admissible'] and 0 < baseline['force_N'] == summary['initial_force_N']
    elapsed = summary['cumulative_seconds']
    assert elapsed > plan['cumulative_seconds_before']
    generation = previous['generation_seconds_before'] + summary['wall_seconds']
    assert elapsed + 3600 <= plan['cumulative_pipeline_budget_hours'] * 3600
    return dict(plan, cumulative_seconds_before=elapsed, generation_seconds_before=generation,
                screen_and_solve=False, full_solve=False, screen_predictors=True,
                damped_predictor_screen=True, reuse_gamg=False, reuse_jacobian=False,
                late_forcing_floor=None, expected_baseline_force_N=baseline['force_N'],
                damped_screen_parent=recovery.name,
                scope='Measure the state-552 force and whole-element endpoint and stress positivity '
                      'for fixed damped increments and the stress extrapolation baseline.')


def saved_iterate_parameters(path, summary, options):
    assert sha(path) == summary['last_newton_iterate_sha256']
    with np.load(path, allow_pickle=False) as saved:
        assert int(saved['step']) == STEP and not bool(saved['accepted'])
        force, shift = float(saved['force_N']), float(saved['next_shift'])
        assert np.isfinite(force) and 1e-9 < force == summary['last_accepted_custom_force_N']
        assert np.isfinite(shift) and options['minimum_regularization'] <= shift <= options['maximum_regularization']
        return force, shift, int(saved['iteration'])


def prepare_warm_restart(run, failed, screen):
    for name in ('plan.json', 'summary.json'):
        registered(P, screen, screen / name)
    screen_plan = load(screen / 'plan.json')
    screen_result = load(screen / 'summary.json')
    assert screen_plan['damped_predictor_screen'] and screen_plan['failed_restart_run'] == failed.name
    assert screen_plan['step'] == screen_result['step'] == STEP
    assert screen_result['complete'] and screen_result['predictor_screen_complete']
    assert not screen_result['reference_continuation_allowed']
    recovery = HPC / screen_plan['damped_screen_parent']
    plan = prepare_damped_screen(run, failed, recovery)
    assert screen_plan['cumulative_seconds_before'] == plan['cumulative_seconds_before']
    assert screen_plan['generation_seconds_before'] == plan['generation_seconds_before']
    saved = recovery / 'last-newton.npz'
    registered(P, recovery, saved)
    summary = load(recovery / 'summary.json')
    options = load(P / plan['stream_protocol'])['admissible_newton']
    force, shift, iteration = saved_iterate_parameters(saved, summary, options)
    for path in (screen / 'plan.json', screen / 'summary.json', saved):
        plan['frozen'][str(path.relative_to(P))] = sha(path)
    elapsed = screen_result['cumulative_seconds']
    assert elapsed > plan['cumulative_seconds_before']
    assert elapsed + 3600 <= plan['cumulative_pipeline_budget_hours'] * 3600
    assert plan['generation_seconds_before'] + 3600 <= plan['native_generation_budget_hours'] * 3600
    return dict(plan, cumulative_seconds_before=elapsed, damped_predictor_screen=False,
                screen_predictors=False, full_solve=True, warm_restart_552=True,
                warm_restart_parent=recovery.name, warm_restart_screen=screen.name,
                saved_iterate=str(saved.relative_to(P)), saved_iterate_force_N=force,
                saved_iterate_iteration=iteration, warm_shift=shift, late_forcing_floor=1e-5,
                scope='Bounded state-552 solve from the verified unapplied iterate. Assemble a fresh '
                      'Jacobian for each update and independently audit any converged candidate.')


def verified_final_trend(trace, summary, force, shift, iteration):
    assert trace['step'] == STEP
    rows = [row for row in trace['custom_newton_steps'] if row.get('accepted_custom_update')]
    assert len(rows) == summary['accepted_custom_iterations'] and len(rows) >= 5
    last = rows[-1]
    assert last['iteration'] == iteration and last['next_regularization'] == shift
    trials = [row['trials'][-1] for row in rows[-5:]]
    assert all(trial['whole_element_admissible'] and trial['armijo_passed'] for trial in trials)
    forces = [trial['native_force_N'] for trial in trials]
    assert forces[-1] == force
    assert all(np.isfinite(value) and value > 0 for value in forces)
    assert all(left > right for left, right in zip(forces, forces[1:])), (
        'Last five accepted residuals did not decrease strictly')
    return forces


def prepare_warm_followup(run, failed, warm):
    for name in ('plan.json', 'summary.json', 'last-newton.npz'):
        registered(P, warm, warm / name)
    previous = load(warm / 'plan.json')
    summary = load(warm / 'summary.json')
    assert previous['warm_restart_552'] and 'warm_followup_parent' not in previous
    assert previous['step'] == summary['step'] == STEP
    assert previous['failed_restart_run'] == failed.name and previous['full_solve']
    assert not previous['reuse_jacobian'] and not previous['reuse_gamg']
    assert summary['complete'] and not summary['passed'] and TIMEOUT in summary['stop_reason']
    assert not summary['physical_new_state_generated'] and not summary['reference_continuation_allowed']
    assert summary['reused_jacobian_updates'] == summary['jacobian_reuse_rejections'] == 0
    plan = prepare_warm_restart(run, failed, HPC / previous['warm_restart_screen'])
    for name in ('manifest.json', 'restart.npz'):
        key = f"data/processed/{plan['target_name']}/{name}"
        assert previous['frozen'][key] == plan['frozen'][key]
    assert previous['saved_iterate'] == plan['saved_iterate']
    assert previous['cumulative_seconds_before'] == plan['cumulative_seconds_before']
    assert previous['generation_seconds_before'] == plan['generation_seconds_before']
    saved = warm / 'last-newton.npz'
    options = load(P / plan['stream_protocol'])['admissible_newton']
    force, shift, iteration = saved_iterate_parameters(saved, summary, options)
    trace = warm / 'trace.json'
    trend = verified_final_trend(load(trace), summary, force, shift, iteration)
    for path in (warm / 'plan.json', warm / 'summary.json', saved, trace):
        plan['frozen'][str(path.relative_to(P))] = sha(path)
    elapsed = summary['cumulative_seconds']
    generation = previous['generation_seconds_before'] + summary['wall_seconds']
    assert elapsed > plan['cumulative_seconds_before'] and summary['wall_seconds'] > 0
    assert elapsed + 3600 <= plan['cumulative_pipeline_budget_hours'] * 3600
    assert generation + 3600 <= plan['native_generation_budget_hours'] * 3600
    return dict(plan, warm_followup_parent=warm.name, prior_final_force_trend_N=trend,
                cumulative_seconds_before=elapsed, generation_seconds_before=generation,
                saved_iterate=str(saved.relative_to(P)), saved_iterate_force_N=force,
                saved_iterate_iteration=iteration, warm_shift=shift)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--failed-restart', default='cmbbe-restart-551-01')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--startup-failure')
    modes.add_argument('--damped-screen-after')
    modes.add_argument('--warm-restart-after-screen')
    modes.add_argument('--warm-followup-after')
    args = parser.parse_args()
    for value in filter(None, (args.run_id, args.failed_restart, args.startup_failure,
                               args.damped_screen_after, args.warm_restart_after_screen,
                               args.warm_followup_after)):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('Invalid run ID')
    with (P / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(P)
        run = HPC / args.run_id
        startup_failure = HPC / args.startup_failure if args.startup_failure else None
        failed = HPC / args.failed_restart
        if args.warm_followup_after:
            plan = prepare_warm_followup(run, failed, HPC / args.warm_followup_after)
        elif args.warm_restart_after_screen:
            plan = prepare_warm_restart(run, failed, HPC / args.warm_restart_after_screen)
        elif args.damped_screen_after:
            plan = prepare_damped_screen(run, failed, HPC / args.damped_screen_after)
        else:
            plan = prepare(run, failed, startup_failure)
        queue_recovery(run, plan)
