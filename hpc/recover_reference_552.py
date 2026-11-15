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


def prepare(run, failed):
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
    assert elapsed + 3600 <= pilot['cumulative_pipeline_budget_hours'] * 3600
    assert generation + 3600 <= pilot['native_generation_budget_hours'] * 3600
    return dict(pilot, run_id=run.name, step=STEP, frozen=frozen, queue='nvidia',
                failed_restart_run=failed.name, cumulative_seconds_before=elapsed,
                generation_seconds_before=generation, solver_seconds=1800,
                screen_and_solve=True, full_solve=True, log_phases=True,
                reuse_jacobian=True, reuse_gamg=True, regularization_policy='two_good_steps',
                late_forcing_floor=1e-5,
                scope='Screen seven predictors, then solve one state within the original Newton limit. '
                      'A converged candidate receives an independent audit. Reference admission is separate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--failed-restart', default='cmbbe-restart-551-01')
    args = parser.parse_args()
    for value in (args.run_id, args.failed_restart):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('Invalid run ID')
    with (P / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(P)
        run = HPC / args.run_id
        queue_recovery(run, prepare(run, HPC / args.failed_restart))
