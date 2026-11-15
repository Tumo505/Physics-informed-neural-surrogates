"""Submit a checkpointed state-553 recovery from accepted state 552."""
import argparse
import fcntl
from pathlib import Path
import re
import zipfile

from probe_regularization_recovery import P, load, queue_recovery
from qualify_reference_restart import check_checkpoint, TIMEOUT
from recover_reference_552 import recovery_plan
from registry import sha
from submit_probe_profile import assert_idle, registered

HPC = P / 'experiments/hpc'
STEP = 552


def prepare(run, qualification):
    evidence = [qualification / name for name in ('plan.json', 'state.json', 'command-553.log')]
    for path in (evidence[0], evidence[2]):
        registered(P, qualification, path)
    previous, state = map(load, evidence[:2])
    assert previous['fresh_jacobian'] and (previous['start'], previous['stop']) == (STEP, STEP + 2)
    assert state['status'] == 'blocked' and state['terminal_rejection'] and not state['measurements']
    assert state['phase'] == 'state_553' and state['completed_steps'] == STEP
    assert f'RuntimeError: {TIMEOUT}' in evidence[2].read_text()
    target = P / 'data/processed' / previous['target_name']
    manifest, digest = check_checkpoint(target, STEP)
    assert digest == previous['start_restart_sha256']
    assert sha(target / 'manifest.json') == previous['start_manifest_sha256']
    trace_path = target / 'custom_trace_000552.json'
    trace = load(trace_path)
    assert trace['step'] == STEP and trace['solver_policy'] == 'fresh_jacobian_two_good_steps'
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
    for path, expected in frozen.items():
        if P / path != recovery_driver:
            assert sha(P / path) == expected, 'Changed inherited input: ' + path
    for path in (*evidence, trace_path, target / 'manifest.json', target / 'restart.npz',
                 target / 'step_journal/000550.json', target / 'step_journal/000551.json',
                 recovery_driver, Path(__file__).resolve()):
        frozen[str(path.relative_to(P))] = sha(path)
    elapsed, generation = state['cumulative_seconds'], state['generation_seconds']
    assert elapsed + 3600 <= previous['cumulative_pipeline_budget_hours'] * 3600
    assert generation + 3600 <= previous['native_generation_budget_hours'] * 3600
    return recovery_plan(previous, run_id=run.name, step=STEP, frozen=frozen,
                         queue='nvidia', checkpointed_recovery_553=True,
                         failed_qualification=qualification.name, full_solve=True, log_phases=True,
                         solver_seconds=1800, regularization_policy='two_good_steps',
                         reuse_jacobian=False, reuse_gamg=False, late_forcing_floor=1e-5,
                         cumulative_seconds_before=elapsed, generation_seconds_before=generation,
                         scope='One state-553 recovery from accepted state 552. Atomically preserve '
                               'each accepted Newton iterate. Independently audit a converged candidate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', default='cmbbe-fresh-restart-552-01')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    for value in (args.qualification, args.run_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('Invalid run ID')
    with (P / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(P)
        run = HPC / args.run_id
        queue_recovery(run, prepare(run, HPC / args.qualification))
