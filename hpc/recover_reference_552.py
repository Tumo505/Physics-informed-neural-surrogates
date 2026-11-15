"""Register one bounded state-552 recovery from the unchanged admitted checkpoint."""
import argparse
import fcntl
from pathlib import Path
import re

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--failed-restart', default='cmbbe-restart-551-01')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--startup-failure')
    modes.add_argument('--damped-screen-after')
    args = parser.parse_args()
    for value in filter(None, (args.run_id, args.failed_restart, args.startup_failure, args.damped_screen_after)):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('Invalid run ID')
    with (P / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(P)
        run = HPC / args.run_id
        startup_failure = HPC / args.startup_failure if args.startup_failure else None
        failed = HPC / args.failed_restart
        if args.damped_screen_after:
            plan = prepare_damped_screen(run, failed, HPC / args.damped_screen_after)
        else:
            plan = prepare(run, failed, startup_failure)
        queue_recovery(run, plan)
