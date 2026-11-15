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


def test_state_552_warm_iterate_must_match_verified_summary(tmp_path):
    import numpy as np
    from recover_reference_552 import saved_iterate_parameters

    path = tmp_path / 'last-newton.npz'
    np.savez(path, step=551, accepted=False, force_N=.087, next_shift=.0625, iteration=35)
    summary = dict(last_newton_iterate_sha256=sha(path), last_accepted_custom_force_N=.087)
    options = dict(minimum_regularization=1e-9, maximum_regularization=100)
    assert saved_iterate_parameters(path, summary, options) == (.087, .0625, 35)
    summary['last_accepted_custom_force_N'] = .09
    with pytest.raises(AssertionError):
        saved_iterate_parameters(path, summary, options)


def test_state_552_followup_checks_final_trend():
    from recover_reference_552 import verified_final_trend

    forces = [.08, .06, .03, .002, .00005]
    rows = [dict(iteration=i, next_regularization=.0625, accepted_custom_update=True,
                 trials=[dict(native_force_N=force, whole_element_admissible=True, armijo_passed=True)])
            for i, force in enumerate(forces)]
    trace = dict(step=551, custom_newton_steps=rows)
    summary = dict(accepted_custom_iterations=5)
    assert verified_final_trend(trace, summary, forces[-1], .0625, 4) == forces
    rows[2]['trials'][0]['native_force_N'] = .07
    with pytest.raises(AssertionError, match='did not decrease'):
        verified_final_trend(trace, summary, forces[-1], .0625, 4)
