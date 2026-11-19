from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
from coupled_newton_direction import ActiveSet, solve_coupled, build_constraints
from lv_surrogate.physics.bernstein import BernsteinAudit
import launch_refined_production as production
from registry import artifact, atomic, event, sha
from test_hpc_refined_production import production_run


def dense_direction(A, b, normals, lower):
    raw = np.linalg.solve(A,b)
    controller = ActiveSet(normals, lower, raw)
    responses = {}
    while True:
        kind, payload = controller.action()
        assert kind != 'failed', payload
        if kind == 'done':
            direction = raw.copy()
            for index, multiplier in zip(payload['active'],payload['multipliers'],strict=True):
                direction += multiplier*responses[index]
            return direction, payload
        assert kind == 'response'
        responses[payload] = np.linalg.solve(A,normals[payload])
        controller.add_response(payload,responses[payload])


def test_jacobian_coupling_moves_a_free_dof_that_local_clipping_cannot_move():
    A = np.array([[4.,1.],[1.,2.]])
    b = np.array([-4.,-1.])
    normals, lower = np.array([[1.,0.]]), np.array([0.])
    direction, detail = dense_direction(A,b,normals,lower)
    np.testing.assert_allclose(direction,[0.,-.5],atol=1e-12)
    multiplier = detail['multipliers'][0]
    np.testing.assert_allclose(A @ direction-b-normals[0]*multiplier,0.,atol=1e-12)
    assert multiplier >= 0 and (-b) @ (A @ direction) < 0
    # Merely clipping the offending raw component produces zero, without this response.
    raw = np.linalg.solve(A,b)
    raw[0] = 0
    np.testing.assert_allclose(raw,0.,atol=1e-12)


def test_multiple_active_constraints_satisfy_independently_assembled_block_system():
    rng = np.random.default_rng(813578)
    matrix = rng.normal(size=(5,5))
    A = matrix.T @ matrix+np.eye(5)
    normals = np.eye(5)[:3]
    lower = np.array([-.1,-.2,-10.])
    expected = np.array([-.1,-.2,.1,.3,-.4])
    lam = np.array([2.,3.,0.])
    b = A @ expected-normals.T @ lam
    actual, detail = dense_direction(A,b,normals,lower)
    np.testing.assert_allclose(actual,expected,atol=1e-10)
    active = detail['active']
    n = normals[active]
    block = np.block([[A,-n.T],[n,np.zeros((len(active),len(active)))]])
    independent = np.linalg.solve(block,np.r_[b,lower[active]])
    np.testing.assert_allclose(actual,independent[:5],atol=1e-10)
    assert np.all(normals @ actual >= lower-1e-12)


def test_redundant_weaker_constraint_does_not_change_solution():
    direction, _ = dense_direction(np.eye(2),np.array([-2.,1.]),
                                   np.array([[1.,0.],[1.,0.]]),np.array([-1.,0.]))
    np.testing.assert_allclose(direction,[0.,1.],atol=1e-12)


def test_feasible_unconstrained_direction_needs_no_response_solves():
    direction, detail = dense_direction(np.eye(2),np.array([1.,2.]),np.array([[1.,0.]]),np.array([0.]))
    assert not detail['active']
    np.testing.assert_array_equal(direction,[1.,2.])


def test_p2_coupled_direction_has_a_positive_force_decreasing_line_search_trial():
    from itertools import combinations
    vertices = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    audit = BernsteinAudit(dict(coords=vertices,cells=np.array([[0,1,2,3]]),
                               canonical_cell_dofs=np.arange(30).reshape(1,10,3)))
    nodes = np.vstack((vertices,[(vertices[i]+vertices[j])/2 for i,j in combinations(range(4),2)]))
    raw = np.zeros(33)
    raw[:30].reshape(10,3)[:,0] = -2*nodes[:,0]
    raw[30:] = .2
    with pytest.raises(ValueError):
        audit.pair(raw,np.zeros(33))
    spec = build_constraints(audit,0,np.zeros(33),np.zeros(33),raw)
    normals = np.zeros((len(spec['lower']),33))
    normals[:,:30] = spec['normals']
    A = np.eye(33)+.01*np.ones((33,33))
    b = A @ raw
    direction, detail = dense_direction(A,b,normals,np.array(spec['lower']))
    assert detail['active'] and -b @ (A @ direction) < 0
    alpha = .25
    proof = audit.pair(alpha*direction,np.zeros(33))
    assert proof['endpoint']['passed'] and proof['stress']['passed']
    assert np.linalg.norm(alpha*A @ direction-b) < np.linalg.norm(b)


class Vec:
    def __init__(self, values):
        self.values = np.array(values,dtype=float)
        self.destroyed = False

    def duplicate(self):return Vec(np.zeros_like(self.values))
    def copy(self, target=None):
        if target is None:return Vec(self.values)
        target.values[:] = self.values
    def getArray(self, readonly=False):return self.values
    def set(self,value):self.values[:] = value
    def axpy(self,alpha,other):self.values += alpha*other.values
    def norm(self):return np.linalg.norm(self.values)
    def destroy(self):self.destroyed = True


class Matrix:
    def __init__(self,values):self.values = np.array(values)
    def mult(self,x,y):y.values[:] = self.values @ x.values
    def createVecs(self):return Vec(np.zeros(len(self.values))),Vec(np.zeros(len(self.values)))


class KSP:
    def __init__(self, matrix, wrong=False):
        self.matrix, self.wrong = matrix, wrong
        self.reuse = False
    def getPC(self):return self
    def setReusePreconditioner(self,value):self.reuse = value
    def setTolerances(self,**kwargs):pass
    def solve(self,b,x):x.values[:] = 0 if self.wrong else np.linalg.solve(self.matrix.values,b.values)
    def getConvergedReason(self):return 2
    def getIterationNumber(self):return 1
    def getOperators(self):return self.matrix,self.matrix


@pytest.mark.parametrize('wrong',[False,True])
@pytest.mark.parametrize('cuda',[False,True])
def test_distributed_adapter_mapping_and_true_equation_qualification(monkeypatch, wrong, cuda):
    monkeypatch.setitem(sys.modules,'mpi4py',SimpleNamespace(MPI=SimpleNamespace(SUM=object())))
    comm = SimpleNamespace(bcast=lambda x,root=0:x, allreduce=lambda x,op=None:x)
    mapping = np.array([1,0])
    A = Matrix(np.array([[4.,1.],[1.,2.]])[np.ix_(mapping,mapping)])
    b, delta = Vec(np.array([-4.,-1.])[mapping]), Vec(np.array([-1.,0.])[mapping])
    ksp = KSP(A,wrong)
    adapter = SimpleNamespace(comm=comm,rank=0,mapping=mapping)
    builder = lambda current, raw:dict(ids=[0],normals=[[1.]],lower=[0.],
        minimum_point_determinant=1e-9,endpoint_constraints=1,stress_constraints=0)
    kwargs = dict(cuda_linear=cuda,budget=lambda:None,eta=1e-4)
    if wrong:
        with pytest.raises(RuntimeError,match='response failed true residual qualification'):
            solve_coupled(A,ksp,b,delta,adapter,builder,np.zeros(2),np.array([-1.,0.]),**kwargs)
        np.testing.assert_array_equal(delta.values,np.array([-1.,0.])[mapping])
    else:
        detail = solve_coupled(A,ksp,b,delta,adapter,builder,np.zeros(2),np.array([-1.,0.]),**kwargs)
        np.testing.assert_allclose(delta.values,np.array([0.,-.5])[mapping],atol=1e-12)
        assert detail['stationarity_residual_N'] <= detail['stationarity_limit_N']
        assert detail['response_solves'] == 1 and all(a['qualified'] for a in detail['response_attempts'])
    assert not b.destroyed and not delta.destroyed and not ksp.reuse


def test_coupled_retry_preserves_checkpoint_and_disables_previous_projection(production_run):
    failed, _, target = production_run
    source = production.load(failed / 'plan.json')
    source.update(adaptive_interval_proof=dict(margin=1e-9),certified_depth_retry=dict(depth=32),
                  constrained_direction=dict(kind='joint_determinant_merit_qp',margin=1e-9))
    atomic(failed / 'plan.json',source)
    artifact(production.ROOT,failed,failed / 'plan.json','Projected-direction source')
    atomic(failed / 'state.json',dict(status='running',slice=1,job='12344.pbs'))
    event(failed,'job_submission',return_code=0,stdout='12344.pbs')
    atomic(failed / 'factor-4/substep-021/trace-slice-001.json',dict(substep=21,
        direction_policy='joint_determinant_merit_qp',custom_newton_steps=[
            dict(initial_force_N=.19834,determinant_direction=dict(applied=True)) for _ in range(8)]))
    run = failed.parent / 'coupled-newton'
    production.initialize_coupled_retry(run,failed)
    plan = production.load(run / 'plan.json')
    assert plan['coupled_newton']['kind'] == 'assembled_jacobian_active_set'
    assert not plan.get('constrained_direction')
    assert plan['coupled_newton']['margin'] == plan['adaptive_interval_proof']['margin'] == 1e-9
    assert sha(run / 'factor-4/substep-020/checkpoint.npz') == sha(failed / 'factor-4/substep-020/checkpoint.npz')
    assert sha(run / 'pressure.json') == sha(failed / 'pressure.json')
    assert plan['requested_substeps'] == 4200 and plan['walltime_seconds'] == 168*3600
    assert production.load(target / 'manifest.json')['completed_steps'] == 554
