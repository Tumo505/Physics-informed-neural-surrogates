import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
from probe_regularization_recovery import check_trace


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
