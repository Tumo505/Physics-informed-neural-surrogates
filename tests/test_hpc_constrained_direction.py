from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
from refined_constrained_direction import point_constraint, project_joint_direction, correct_direction
from lv_surrogate.physics.bernstein import BernsteinAudit
import launch_refined_production as production
from registry import artifact, atomic, event, sha
from test_hpc_refined_production import production_run


@pytest.mark.parametrize('factor', [1., .6])
def test_determinant_constraint_gradient_matches_finite_difference(factor):
    vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    audit = BernsteinAudit(dict(coords=vertices, cells=np.array([[0,1,2,3]]),
                                canonical_cell_dofs=np.arange(30).reshape(1,10,3)))
    rng = np.random.default_rng(813578)
    nodal, step = rng.normal(size=(10,3))*.01, rng.normal(size=(10,3))
    point = np.array([.1,.2,.3,.4])
    _, gradient = point_constraint(nodal, factor, point, audit.gl[0], audit.edges)
    epsilon = 1e-6
    plus = point_constraint(nodal+epsilon*factor*step, factor, point, audit.gl[0], audit.edges)[0]
    minus = point_constraint(nodal-epsilon*factor*step, factor, point, audit.gl[0], audit.edges)[0]
    assert float(gradient @ step.ravel()) == pytest.approx((plus-minus)/(2*epsilon), rel=1e-7, abs=1e-8)


def test_joint_projection_uses_global_descent_when_local_merit_and_constraint_conflict():
    d = np.array([-1., .2, 0.])
    h = np.array([-1., 1., 100.])  # negative force-merit gradient; third component is fixed
    original = d.copy()
    corrected, detail = project_joint_direction(d, h, np.array([0]), np.array([[1.]]), np.array([.1+1e-9]))
    assert detail['applied']
    assert corrected[0] >= -.05-1e-10 and corrected[2] == 0
    assert h @ corrected > 0 and h @ corrected >= .5*(h @ original)-1e-10
    assert detail['corrected_merit_slope'] < 0
    assert detail['outside_correction_norm'] > 0
    np.testing.assert_array_equal(d, original)


def test_multiple_determinant_constraints_and_fixed_dofs_are_preserved():
    d = np.array([-1.,-1.,.2,0.])
    h = np.array([-1.,-1.,1.,100.])
    corrected, detail = project_joint_direction(d, h, np.array([0,1]), np.eye(2), np.array([.1+1e-9,.2+1e-9]))
    assert detail['applied'] and detail['violated_linear_constraints'] == 2
    assert corrected[0] >= -.05-1e-10 and corrected[1] >= -.1-1e-10
    assert corrected[3] == 0 and h @ corrected > 0
    assert np.linalg.norm(corrected) <= 4*np.linalg.norm(d)


def test_corrected_direction_has_actual_small_step_force_descent_with_expected_gradient_sign():
    residual = np.array([1., .2])
    d = -residual
    corrected, _ = project_joint_direction(d, -residual, np.array([0]), np.array([[1.]]), np.array([.1+1e-9]))
    assert residual @ corrected < 0
    assert np.linalg.norm(residual+1e-4*corrected) < np.linalg.norm(residual)


def test_unconstrained_newton_direction_is_retained():
    d = np.array([-.01,.1])
    result, detail = project_joint_direction(d, -d, np.array([0]), np.array([[1.]]), np.array([.1]))
    assert result is None and not detail['applied']


def test_infeasible_joint_descent_does_not_silently_return_an_invalid_direction():
    with pytest.raises(RuntimeError, match='No joint determinant-safe force-descent direction'):
        project_joint_direction(np.array([-1.]), np.array([-1.]), np.array([0]), np.array([[1.]]), np.array([1e-9]))


def test_full_p2_direction_correction_avoids_inversion_and_passes_endpoint_and_stress_proofs():
    from itertools import combinations
    vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    audit = BernsteinAudit(dict(coords=vertices, cells=np.array([[0,1,2,3]]),
                                canonical_cell_dofs=np.arange(30).reshape(1,10,3)))
    nodes = np.vstack((vertices, [(vertices[i]+vertices[j])/2 for i,j in combinations(range(4),2)]))
    current = np.zeros(33)
    direction = np.zeros(33)
    direction[:30].reshape(10,3)[:,0] = -2*nodes[:,0]
    direction[30:] = .2
    with pytest.raises(ValueError, match='nonpositive determinant witness'):
        audit.pair(current+direction, current)
    corrected, detail = correct_direction(audit, 0, current, current, direction, direction)
    assert detail['applied'] and detail['endpoint_constraints'] >= 20 and detail['stress_constraints'] >= 20
    assert direction @ corrected > 0
    proof = audit.pair(current+corrected, current)
    assert proof['endpoint']['passed'] and proof['stress']['passed']
    assert detail['corrected_merit_slope'] < 0


def test_production_retry_starts_from_physical_checkpoint_and_keeps_certification(production_run):
    failed, _, target = production_run
    source = production.load(failed / 'plan.json')
    source.update(adaptive_interval_proof=dict(margin=1e-9), certified_depth_retry=dict(depth=32))
    atomic(failed / 'plan.json', source)
    artifact(production.ROOT, failed, failed / 'plan.json', 'Adaptive source fixture')
    atomic(failed / 'state.json', dict(status='blocked', slice=1, job='12344.pbs'))
    event(failed, 'job_submission', return_code=0, stdout='12344.pbs')
    atomic(failed / 'factor-4/substep-021/trace-slice-001.json', dict(substep=21,
        positivity_fallback_algorithm='determinant_adaptive_interval_bernstein', custom_newton_steps=[
            dict(initial_force_N=.2348, trials=[dict(positivity_failure=
                'Direct determinant margin violation witness (stage=endpoint, element=813578)')])]))
    run = failed.parent / 'joint-direction-production'
    production.initialize_direction_retry(run, failed)
    plan = production.load(run / 'plan.json')
    assert plan['constrained_direction']['element'] == 813578 and plan['constrained_direction']['margin'] == 1e-9
    assert plan['adaptive_interval_proof']['margin'] == 1e-9 and plan['certified_depth_retry']['depth'] == 32
    assert plan['requested_substeps'] == 4200 and plan['walltime_seconds'] == 168*3600
    assert sha(run / 'factor-4/substep-020/checkpoint.npz') == sha(failed / 'factor-4/substep-020/checkpoint.npz')
    assert sha(run / 'pressure.json') == sha(failed / 'pressure.json')
    assert not (run / 'factor-4/substep-021').exists()
    assert production.load(target / 'manifest.json')['completed_steps'] == 554
