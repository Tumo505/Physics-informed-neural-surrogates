import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
import compare_reference_timesteps_555 as refinement
from registry import atomic, sha
from lv_surrogate.physics.regularization_recovery_policy import retry_linear_accuracy


def test_reported_false_convergence_retries_tighter_without_relaxing_gate():
    limit = 1.1e-5 * 1.9049182240979737e-7
    attempts = [dict(reason=2, true_residual_N=3.1259302578872842e-12, qualified=False)]
    calls = []

    def solve(reuse, rtol, atol):
        calls.append((reuse, rtol, atol))
        true = (2.2e-12, 1.5e-12)[len(calls)-1]
        return dict(reason=2, true_residual_N=true, qualified=true <= limit)

    retry_linear_accuracy(solve, attempts, rtol=1e-5, limit=limit)
    assert len(calls) == 2 and attempts[-1]['qualified']
    assert calls[0] == pytest.approx((False, 1e-6, limit*.05))
    assert calls[1] == pytest.approx((False, 1e-7, limit*.05))


@pytest.mark.parametrize('reason,true,qualified,expected_calls', [
    (2, 1e-13, True, 0), (-3, 1., False, 0), (2, None, False, 0), (2, 1., False, 2),
])
def test_linear_retries_are_bounded_and_never_waive_qualification(reason, true, qualified, expected_calls):
    last = dict(reason=reason, true_residual_N=true, qualified=qualified)
    calls = []

    def solve(*args):
        calls.append(args)
        return dict(last)

    attempts = [last]
    retry_linear_accuracy(solve, attempts, rtol=1e-5, limit=1e-12)
    assert len(calls) == expected_calls
    assert attempts[-1]['qualified'] == qualified


@pytest.fixture
def accepted_run(tmp_path):
    refinement.save_npz(tmp_path / 'seed-550.npz', u=np.zeros(3))
    atomic(tmp_path / 'pressure.json', dict(factors={'4': dict(endpoint_times=[.344], pressures=[14100.])}))
    step = tmp_path / 'factor-4/substep-001'
    step.mkdir(parents=True)
    values = dict(u=np.ones(3), u_old=np.ones(3), v_old=np.full(3, 2.),
                  a_old=np.full(3, 3.), previous=np.zeros(3))
    refinement.save_npz(step / 'checkpoint.npz', **values, completed_substeps=1, factor=4, time_s=.344)
    atomic(step / 'trace.json', dict(custom_force_N=2e-10))
    row = dict(substep=1, time_s=.344, pressure_Pa=14100., custom_force_N=2e-10,
               native_force_N=2e-10, native_updates=0,
               input_checkpoint_sha256=sha(tmp_path / 'seed-550.npz'),
               checkpoint_sha256=sha(step / 'checkpoint.npz'), trace_sha256=sha(step / 'trace.json'))
    atomic(step / 'accepted.json', row)
    atomic(tmp_path / 'result.json', dict(complete=True, outcomes={'4': dict(completed=dict(
        factor=4, start_state=550, stop_state=555, complete=False, records=[row], completed_substeps=1))}))
    return tmp_path, step, row, values


def test_resume_restores_accepted_histories_and_previous_displacement(accepted_run):
    parent, step, row, expected = accepted_run
    assert refinement.accepted_prefix(parent, 4) == [row]
    actual = refinement.restored_checkpoint(step / 'checkpoint.npz', row, 4)
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values)


@pytest.mark.parametrize('filename', ['checkpoint.npz', 'trace.json', 'accepted.json'])
def test_resume_rejects_modified_accepted_files(accepted_run, filename):
    parent, step, _, _ = accepted_run
    (step / filename).write_text('{}')
    with pytest.raises(AssertionError):
        refinement.accepted_prefix(parent, 4)


def test_resume_rejects_checkpoint_from_another_factor(accepted_run):
    _, step, row, _ = accepted_run
    with pytest.raises(AssertionError):
        refinement.restored_checkpoint(step / 'checkpoint.npz', row, 2)


def test_resume_rejects_broken_history_chain(accepted_run):
    parent, step, row, _ = accepted_run
    row['input_checkpoint_sha256'] = 'wrong predecessor'
    atomic(step / 'accepted.json', row)
    result = refinement.load(parent / 'result.json')
    result['outcomes']['4']['completed']['records'] = [row]
    atomic(parent / 'result.json', result)
    with pytest.raises(AssertionError):
        refinement.accepted_prefix(parent, 4)
