import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
from probe_regularization_recovery import check_followup, check_trace


@pytest.fixture
def timeout_trace():
    def accepted(alpha, ratio):
        return dict(alpha=alpha, armijo_passed=True, whole_element_admissible=True,
                    actual_to_predicted_merit_ratio=ratio)

    steps = []
    for alpha, ratio in zip((.25, .25, .25, .5), (.86, .88, .89, .96)):
        steps.append(dict(regularization=.00390625, next_regularization=.015625,
                          accepted_custom_update=True,
                          trials=[dict(alpha=1., armijo_passed=False), accepted(alpha, .6)]))
        steps.append(dict(regularization=.015625, next_regularization=.00390625,
                          accepted_custom_update=True, trials=[accepted(1., ratio)]))
    return dict(step=550, custom_newton_steps=steps)


def test_observed_armijo_cycle_qualifies_for_recovery_probe(timeout_trace):
    check_trace(timeout_trace)


@pytest.mark.parametrize('row,trial,key,value', [
    (0, 0, 'armijo_passed', True),
    (1, 0, 'actual_to_predicted_merit_ratio', .75),
    (3, 0, 'whole_element_admissible', False),
])
def test_different_timeout_mechanism_is_rejected(timeout_trace, row, trial, key, value):
    timeout_trace['custom_newton_steps'][row]['trials'][trial][key] = value
    with pytest.raises(AssertionError):
        check_trace(timeout_trace)


def test_two_good_steps_hold_shift_after_backtracking():
    from lv_surrogate.physics.regularization_recovery_policy import update_shift

    def advance(mu, count, full_step, ratio):
        return update_shift(mu, count, accepted=True, full_step=full_step,
                            ratio=ratio, minimum=1e-9, maximum=1.)

    mu, count = advance(.0078125, 0, False, .78)
    assert (mu, count) == (.03125, 0)
    mu, count = advance(mu, count, True, .97)
    assert (mu, count) == (.03125, 1)
    assert advance(mu, count, True, .98) == (.015625, 0)
    assert advance(mu, count, True, .6) == (.03125, 0)
    assert advance(mu, count, True, .086) == (.125, 0)


def test_followup_accepts_observed_cycle_ending_in_a_poor_full_step(timeout_trace):
    steps = timeout_trace['custom_newton_steps']
    for low, high in zip(steps[::2], steps[1::2]):
        low.update(regularization=.0078125, next_regularization=.03125)
        high.update(regularization=.03125, next_regularization=.0078125)
    steps[-2]['trials'] = [dict(alpha=1., armijo_passed=True,
                               whole_element_admissible=True,
                               actual_to_predicted_merit_ratio=.086)]
    timeout_trace['custom_newton_steps'] = [
        row for low, high in zip(steps[::2], steps[1::2]) for row in (high, low)
    ]
    check_followup(timeout_trace)
    steps[-2]['trials'][0]['actual_to_predicted_merit_ratio'] = .5
    with pytest.raises(AssertionError):
        check_followup(timeout_trace)


def test_damped_screen_preserves_baseline_and_ranks_only_admissible_candidates():
    import numpy as np
    from probe_regularization_recovery import damped_predictors, summarize_damped_screen

    old = np.array([2.])
    candidates = dict(damped_predictors(old, np.array([1.]), np.array([4.]), np.array([1.3])))
    assert len(candidates) == 11
    np.testing.assert_array_equal(candidates['stress extrapolation'], [1.3])
    np.testing.assert_array_equal(candidates['damped velocity 0.125'], [2.5])
    np.testing.assert_array_equal(candidates['damped displacement -0.25'], [1.75])
    rows = [
        dict(predictor='stress extrapolation', admissible=True, force_N=.167),
        dict(predictor='damped velocity 0.125', admissible=False, force_N=.01),
        dict(predictor='damped displacement -0.25', admissible=True, force_N=.12),
    ]
    assert summarize_damped_screen(rows) == dict(
        baseline_force_N=.167, best_damped_predictor=rows[2], improves_on_baseline=True)
    assert not summarize_damped_screen(rows[:2])['improves_on_baseline']
