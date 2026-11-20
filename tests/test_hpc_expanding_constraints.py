from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
import coupled_newton_direction as coupled
import launch_refined_production as production
from registry import artifact, atomic, event, sha
from test_hpc_refined_production import production_run
from test_hpc_coupled_newton import Vec, Matrix, KSP, dense_direction


def test_growing_pool_keeps_old_cells_and_refreshes_history(monkeypatch):
    audit = SimpleNamespace(maps=np.zeros((5,10,3), dtype=int))
    history = [np.array([1.])]
    pool = coupled.GrowingConstraints(audit,[1],lambda:history[0],max_elements=3)
    calls = []
    monkeypatch.setattr(coupled,'build_multi_constraints',
        lambda a, elements, current, old, raw:calls.append((set(elements),old.copy())))
    reason = 'Direct determinant margin violation witness (stage=endpoint, element=2)'
    assert pool.observe_failure(reason)['elements'] == [1,2]
    assert pool.observe_failure(reason) is None
    assert pool.observe_failure('Armijo rejected') is None
    pool(None,None)
    history[0] = np.array([2.])
    pool.observe_failure('Whole-element positivity remains unresolved at maximum depth (stage=stress, element=3)')
    pool(None,None)
    assert calls[0][0] == {1,2} and calls[1][0] == {1,2,3}
    np.testing.assert_array_equal(calls[1][1],[2.])
    with pytest.raises(RuntimeError,match='no constraints dropped'):
        pool.observe_failure('Direct nonpositive determinant witness (stage=endpoint, element=4)')
    assert pool.elements == {1,2,3}


def test_shared_cell_dofs_are_embedded_into_one_coupled_system(monkeypatch):
    import refined_constrained_direction
    maps = np.array([np.arange(30).reshape(10,3),np.arange(15,45).reshape(10,3)])
    audit = SimpleNamespace(maps=maps,margin=1e-9)
    def constraints(audit,element,current,old,raw):
        normals = np.zeros((2,30));normals[0,0]=2.;normals[1,-1]=3.
        return normals,np.array([1e-9,1e-9]),['endpoint','stress']
    monkeypatch.setattr(refined_constrained_direction,'determinant_constraints',constraints)
    spec = coupled.build_multi_constraints(audit,[1,0,1],None,None,None)
    assert spec['elements'] == [0,1] and spec['ids'] == list(range(45))
    n = np.array(spec['normals'])
    assert list(np.argmax(n,axis=1)) == [0,29,15,44]
    assert spec['endpoint_constraints'] == spec['stress_constraints'] == 2
    A = np.eye(45)+.01*np.ones((45,45))
    b = -np.ones(45)
    direction,detail = dense_direction(A,b,n,np.array(spec['lower']))
    assert len(detail['active']) == 4
    assert np.all(n @ direction >= -1e-12)


def test_retry_restores_only_accepted_history_and_retains_original_constraint(production_run):
    failed,_,target = production_run
    source = production.load(failed / 'plan.json')
    source.update(adaptive_interval_proof=dict(margin=1e-9),certified_depth_retry=dict(depth=32),
                  coupled_newton=dict(kind='assembled_jacobian_active_set',element=813578,margin=1e-9))
    atomic(failed / 'plan.json',source)
    artifact(production.ROOT,failed,failed / 'plan.json','Coupled source')
    atomic(failed / 'state.json',dict(status='blocked',slice=1,job='12345.pbs'))
    event(failed,'job_submission',return_code=0,stdout='12345.pbs')
    atomic(failed / 'factor-4/substep-021/trace-slice-001.json',dict(substep=21,
        direction_policy='assembled_jacobian_active_set',custom_newton_steps=[dict(
            initial_force_N=.138098, trials=[dict(positivity_failure=
                'Direct determinant margin violation witness (stage=endpoint, element=813580)')])]))
    run = failed.parent / 'expanding'
    production.initialize_expanding_retry(run,failed)
    plan = production.load(run / 'plan.json')
    spec = plan['coupled_newton']
    assert spec['dynamic_elements'] and spec['initial_elements'] == [813578]
    assert spec['witnessed_elements'] == [813580] and spec['maximum_elements'] == 32
    assert not plan.get('constrained_direction')
    assert plan['adaptive_interval_proof']['margin'] == spec['margin'] == 1e-9
    assert sha(run / 'factor-4/substep-020/checkpoint.npz') == sha(failed / 'factor-4/substep-020/checkpoint.npz')
    assert sha(run / 'pressure.json') == sha(failed / 'pressure.json')
    assert len(production.production_records(run)) == 20
    assert plan['requested_substeps'] == 4200 and plan['walltime_seconds'] == 168*3600
    assert production.load(target / 'manifest.json')['completed_steps'] == 554


def test_new_cell_rebuilds_direction_before_acceptance_or_shift_change(monkeypatch):
    """Run the actual Newton loop with dense algebra in place of PETSc/FEniCS."""
    import importlib.util
    import lv_surrogate.physics.regularization_recovery_policy as policy
    class Vector(Vec):
        def duplicate(self):return Vector(np.zeros_like(self.values))
        def copy(self,target=None):
            if target is None:return Vector(self.values)
            target.values[:] = self.values
        def norm(self,*args):return np.linalg.norm(self.values,ord=np.inf if args and args[0]=='infinity' else 2)
        def scale(self,a):self.values *= a
        def abs(self):self.values[:] = abs(self.values)
        def dot(self,other):return self.values @ other.values
    class Operator(Matrix):
        comm = None
        def getDiagonal(self):return Vector(np.diag(self.values))
        def setDiagonal(self,v):np.fill_diagonal(self.values,v.values)
        def assemble(self):pass
    class Solver(KSP):
        NormType=SimpleNamespace(UNPRECONDITIONED=0)
        def __init__(self):pass
        def create(self,comm):return self
        def setOperators(self,matrix):self.matrix=matrix
        def setType(self,*args):pass
        def setGMRESRestart(self,*args):pass
        def setNormType(self,*args):pass
        def setMonitor(self,*args):pass
        def setUp(self):pass
        def destroy(self):pass
        def solve(self,b,x):x.values[:] = np.linalg.solve(self.matrix.values,b.values)
    matrix = Operator(np.eye(2))
    field = np.zeros(2)
    comm = SimpleNamespace(bcast=lambda x,root=0:x,allreduce=lambda x,op=None:x,gather=lambda x,root=0:[x])
    adapter = SimpleNamespace(comm=comm,rank=0,ndof=2,parts=[np.arange(2)],mapping=np.arange(2),
        gather=lambda name:field.copy(),assign=lambda name,x:field.__setitem__(slice(None),x))
    problem = SimpleNamespace(u=SimpleNamespace(vector=lambda:None),
        _problem=SimpleNamespace(J=lambda *args:None,bcs=[]))
    backend = lambda x:SimpleNamespace(mat=lambda:matrix,vec=lambda:x)
    fake_df = SimpleNamespace(PETScMatrix=lambda:object(),as_backend_type=backend)
    monkeypatch.setitem(sys.modules,'lv_surrogate.physics.reference_fem',SimpleNamespace(
        df=fake_df,residual=lambda p:Vector(field-np.ones(2))))
    monkeypatch.setitem(sys.modules,'mpi4py',SimpleNamespace(MPI=SimpleNamespace(MAX=0)))
    monkeypatch.setitem(sys.modules,'petsc4py',SimpleNamespace(PETSc=SimpleNamespace(KSP=Solver,
        NormType=SimpleNamespace(NORM_INFINITY='infinity'))))
    path = Path(__file__).resolve().parents[1] / 'src/lv_surrogate/physics/admissible_newton_gentle_shrink.py'
    spec = importlib.util.spec_from_file_location('lv_surrogate.physics._test_expanding_solver',path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    audit_calls=[]
    def proof(candidate,old):
        audit_calls.append(candidate.copy())
        if candidate[1] > .2:
            raise ValueError('Direct determinant margin violation witness (stage=endpoint, element=2)')
        return dict(passed=True)
    pool = coupled.GrowingConstraints(SimpleNamespace(maps=np.zeros((3,10,3))),[1],lambda:np.zeros(2))
    calls=[]
    def solve_coupled(A,ksp,b,delta,adapter,builder,current,raw,**kwargs):
        calls.append((set(builder.elements),current.copy(),raw.copy(),delta.values.copy(),A.values.copy()))
        # Deliberately alter delta to detect accidental reuse of a previously corrected direction.
        delta.values[:] = [.3,.3] if len(builder.elements)==1 else [.3,.1]
        return dict(elements=sorted(builder.elements),active=[],response_solves=0,stationarity_residual_N=0)
    monkeypatch.setattr(coupled,'solve_coupled',solve_coupled)
    options=dict(initial_regularization=1.,minimum_regularization=1e-8,maximum_regularization=1e8,
        custom_absolute_tolerance_N=1e-9,maximum_custom_newton_iterations=10,
        maximum_backtracks=20,armijo_constant=1e-4,wall_budget_seconds=100.)
    record={}
    module.solve(problem,adapter,{'u_old':np.zeros(2)},SimpleNamespace(pair=proof),options,record,
        lambda:None,policy='two_good_steps',stop_after=1,directional_armijo=True,coupled_constraints=pool)
    assert len(calls)==2 and calls[0][0]=={1} and calls[1][0]=={1,2}
    for i in (1,2,3,4):np.testing.assert_array_equal(calls[0][i],calls[1][i])
    np.testing.assert_array_equal(field,[.3,.1])
    rows=record['custom_newton_steps']
    assert len(rows)==1 and rows[0]['iteration']==0 and rows[0]['regularization']==1.
    assert len(rows[0]['direction_attempts'])==2
    assert rows[0]['direction_attempts'][0]['expansion']['element']==2
    assert rows[0]['accepted_custom_update'] and record['coupled_elements']==[1,2]
