import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
import qualify_reference_restart as restart
from registry import artifact, atomic, sha


@pytest.mark.parametrize('marker', [None, restart.TIMEOUT, 'different error'])
def test_registered_timeout_with_optional_marker(tmp_path, monkeypatch, marker):
    monkeypatch.setattr(restart, 'ROOT', tmp_path)
    phase = 'reference_000547'
    parent = tmp_path / 'parent'
    target = tmp_path / 'target'
    target.mkdir()
    log = parent / phase / 'attempt-001/command.log'
    log.parent.mkdir(parents=True)
    log.write_text(f'RuntimeError: {restart.TIMEOUT}\n')
    artifact(tmp_path, parent, log, 'Executed command log')
    state = dict(phase=phase, attempts={phase: 1})

    if marker is not None:
        atomic(target / 'failure.json', dict(error=marker))
    if marker == 'different error':
        with pytest.raises(AssertionError, match='different failure marker'):
            restart.failure_evidence(parent, target, state)
        return

    expected = {} if marker is None else {'failure.json': sha(target / 'failure.json')}
    assert restart.failure_evidence(parent, target, state) == (log, expected)
    log.write_text('Changed log')
    with pytest.raises(AssertionError, match='Unregistered or changed artifact'):
        restart.failure_evidence(parent, target, state)


def test_state_552_plan_does_not_inherit_state_551_solver_inputs():
    from recover_reference_552 import recovery_plan, SHARED_FIELDS

    pilot = dict.fromkeys(SHARED_FIELDS, 'shared input')
    prior_solver = dict(saved_iterate='previous-iterate.npz', saved_iterate_force_N=9.34e-6,
                        warm_shift=0.001, screened_force_N=0.06, screen_predictors=True,
                        compare_shifts=True, fixed_shift=0.01)
    pilot.update(prior_solver)
    plan = recovery_plan(pilot, step=551, screen_and_solve=True, full_solve=True)
    assert not (prior_solver.keys() & plan.keys())
    assert plan['stream_protocol'] == pilot['stream_protocol']
    assert plan['step'] == 551 and plan['screen_and_solve'] and plan['full_solve']
