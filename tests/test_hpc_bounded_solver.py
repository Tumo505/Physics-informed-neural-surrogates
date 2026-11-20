from pathlib import Path
import sys
from types import SimpleNamespace
import time

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hpc'))
import bounded_refined_solver as study
import constrained_residual_direction as merit
import launch_refined_production as production
from registry import artifact,atomic,event,sha
from test_hpc_refined_production import production_run
from test_hpc_coupled_newton import Vec,Matrix,KSP


def prepare_study(production_run):
    failed,parent,target=production_run
    plan=production.load(failed/'plan.json')
    plan.update(adaptive_interval_proof=dict(margin=1e-9),certified_depth_retry=dict(depth=32),
                coupled_newton=dict(kind='assembled_jacobian_active_set',element=813578,
                    initial_elements=[813578],dynamic_elements=True,maximum_elements=32,margin=1e-9))
    atomic(failed/'plan.json',plan);artifact(production.ROOT,failed,failed/'plan.json','Source')
    atomic(failed/'state.json',dict(status='blocked',slice=1,job='12345.pbs'))
    event(failed,'job_submission',return_code=0,stdout='12345.pbs')
    atomic(failed/'factor-4/substep-021/trace-slice-001.json',dict(substep=21,
        direction_policy='assembled_jacobian_active_set',coupled_elements=[813578,813580],
        custom_newton_steps=[dict(initial_force_N=.01816,trials=[],direction_attempts=[dict(trials=[dict(
            positivity_failure='Direct determinant margin violation witness (stage=endpoint, element=813580)')])])]))
    run=failed.parent/'bounded'
    study.initialize(run,failed)
    return run,failed,target


def test_study_caps_four_steps_preserves_histories_and_forcing(production_run):
    run,failed,target=prepare_study(production_run)
    plan=production.load(run/'plan.json')
    assert plan['requested_substeps']==24 and plan['stop_state']==556
    assert plan['solver_study']['wall_budget_seconds']==12*3600
    assert not plan['solver_study']['automatic_successor']
    assert plan['coupled_newton']['initial_elements']==[813578,813580]
    assert plan['coupled_newton']['residual_minimizing']
    assert plan['adaptive_interval_proof']['margin']==1e-9
    assert production.load(run/'state.json')['requested_substeps']==24
    assert sha(run/'factor-4/substep-020/checkpoint.npz')==sha(failed/'factor-4/substep-020/checkpoint.npz')
    original=production.load(failed/'pressure.json')['factors']['4']
    bounded=production.load(run/'pressure.json')['factors']['4']
    for key in ('endpoint_times','pressure_times','pressures'):assert bounded[key]==original[key][:24]
    assert production.load(target/'manifest.json')['completed_steps']==554


@pytest.mark.parametrize('worker_exit,paused',[(1,False),(0,True)])
def test_failed_or_exhausted_study_writes_report_and_never_queues_successor(production_run,monkeypatch,worker_exit,paused):
    run,_,target=prepare_study(production_run)
    nodes=run/'nodes';nodes.write_text('vis01\n')
    monkeypatch.setenv('PBS_JOBID','12346.pbs');monkeypatch.setenv('PBS_NODEFILE',str(nodes))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-a,GPU-b,GPU-c,GPU-d')
    commands=[]
    def worker(command,**kwargs):
        commands.append(command)
        assert command[0]!='qsub'
        atomic(run/'factor-4/summary.json',dict(complete=False,completed_substeps=20,paused_for_walltime=paused))
        return SimpleNamespace(returncode=worker_exit)
    monkeypatch.setattr(production.subprocess,'run',worker)
    production.execute(run)
    assert len(commands)==1
    result=production.load(run/'result.json')
    assert not result['passed'] and not result['failing_substep_accepted']
    assert result['accepted_after']==20 and result['accepted_substep_solve_seconds']==[]
    assert result['stop_reason']==('worker_failure' if worker_exit else 'budget_exhausted')
    assert production.load(run/'state.json')['status']=='study_stopped'


def test_pass_requires_all_four_qualified_commits_and_reports_runtime(production_run,monkeypatch):
    run,_,_=prepare_study(production_run)
    rows=production.production_records(run)
    schedule=production.load(run/'pressure.json')['factors']['4']
    for index in range(21,25):
        step=run/f'factor-4/substep-{index:03d}';step.mkdir()
        (step/'checkpoint.npz').write_bytes(str(index).encode())
        atomic(step/'trace.json',dict(custom_force_N=1e-10))
        row=dict(substep=index,time_s=schedule['endpoint_times'][index-1],
            pressure_time_s=schedule['pressure_times'][index-1],pressure_Pa=schedule['pressures'][index-1],
            custom_force_N=1e-10,native_force_N=1e-10,native_updates=0,
            checkpoint_sha256=sha(step/'checkpoint.npz'),trace_sha256=sha(step/'trace.json'),
            input_checkpoint_sha256=rows[-1]['checkpoint_sha256'],solve_wall_seconds=10.*index)
        atomic(step/'accepted.json',row);artifact(production.ROOT,run,step/'accepted.json','Qualified fixture')
        rows.append(row)
    assert study.finish(run,time.time()-1000,'target_reached',0)=='study_passed'
    report=production.load(run/'result.json')
    assert report['passed'] and report['failing_substep_accepted']
    assert report['subsequent_substeps_accepted']==3 and report['mean_accepted_substep_solve_seconds']==225.
    assert not report['automatic_production_authorized']
    row['custom_force_N']=1e-5
    atomic(step/'accepted.json',row);artifact(production.ROOT,run,step/'accepted.json','Invalid fixture')
    with pytest.raises(AssertionError):study.finish(run,time.time(),'target_reached',0)


class Vector(Vec):
    def duplicate(self):return Vector(np.zeros_like(self.values))
    def copy(self,target=None):
        if target is None:return Vector(self.values)
        target.values[:]=self.values
    def getSizes(self):return len(self.values),len(self.values)
    def dot(self,other):return self.values @ other.values


class Dense(Matrix):
    comm=None
    def copy(self):return Dense(self.values.copy())
    def convert(self,kind):pass
    def createVecs(self):return Vector(np.zeros(len(self.values))),Vector(np.zeros(len(self.values)))
    def multTranspose(self,x,y):y.values[:]=self.values.T @ x.values
    def destroy(self):pass


class Shell:
    def createPython(self,size,context,comm):self.context=context;self.n=size[0][0];return self
    def setUp(self):pass
    def mult(self,x,y):self.context.mult(self,x,y)
    def destroy(self):pass
    @property
    def values(self):
        output=[]
        for x in np.eye(self.n):
            y=Vector(np.zeros(self.n));self.mult(Vector(x),y);output.append(y.values)
        return np.array(output).T


class Solver(KSP):
    NormType=SimpleNamespace(UNPRECONDITIONED=0)
    def __init__(self):self.wrong=False;self.reuse=False
    def create(self,comm):return self
    def setType(self,kind):pass
    def setOperators(self,H):self.matrix=H
    def setNormType(self,kind):pass
    def setMonitor(self,callback):pass
    def destroy(self):pass


@pytest.mark.parametrize('cuda',[False,True])
def test_distributed_merit_direction_matches_independent_constrained_least_squares(monkeypatch,cuda):
    from scipy.optimize import minimize
    monkeypatch.setitem(sys.modules,'petsc4py',SimpleNamespace(PETSc=SimpleNamespace(Mat=Shell,KSP=Solver)))
    monkeypatch.setitem(sys.modules,'mpi4py',SimpleNamespace(MPI=SimpleNamespace(SUM=0)))
    comm=SimpleNamespace(gather=lambda x,root=0:[x],bcast=lambda x,root=0:x,allreduce=lambda x,op=None:x)
    mapping=np.array([2,0,1])
    adapter=SimpleNamespace(comm=comm,rank=0,mapping=mapping,parts=[mapping],ndof=3)
    J=np.array([[2.,3.,0.],[-1.,2.,1.],[0.,-.5,1.]])
    b=np.array([-2.,1.,.5]);s=np.array([2.,2.,1.]);mu=.1
    normals=np.array([[1.,0.,0.],[0.,1.,0.]])
    lower=np.array([-.05,-.02])
    builder=lambda current,raw:dict(ids=[0,1,2],normals=normals.tolist(),lower=lower.tolist(),
        endpoint_constraints=1,stress_constraints=1,minimum_point_determinant=1e-8,elements=[1,2])
    delta=Vector(np.zeros(3))
    detail=merit.residual_direction(Dense(J[np.ix_(mapping,mapping)]),Vector(b[mapping]),delta,
        adapter,builder,np.zeros(3),Vector(s[mapping]),mu,cuda=cuda,budget=lambda:None,eta=1e-5)
    actual=np.empty(3);actual[mapping]=delta.values
    H=J.T@J+mu*np.diag(s*s);g=J.T@b
    expected=minimize(lambda d:.5*d@H@d-g@d,np.zeros(3),jac=lambda d:H@d-g,
        constraints=[dict(type='ineq',fun=lambda d:normals@d-lower,jac=lambda d:normals)],
        method='SLSQP',options=dict(ftol=1e-14,maxiter=100))
    assert expected.success
    np.testing.assert_allclose(actual,expected.x,atol=1e-7)
    assert detail['merit_slope']<0 and -b@(J@actual)<0
    assert np.all(normals@actual>=lower-1e-10)
    assert detail['stationarity_residual_N']<=detail['stationarity_limit_N']


def test_merit_direction_rejects_unqualified_normal_solve(monkeypatch):
    class WrongSolver(Solver):
        def solve(self,b,x):x.values[:]=0.
    monkeypatch.setitem(sys.modules,'petsc4py',SimpleNamespace(PETSc=SimpleNamespace(Mat=Shell,KSP=WrongSolver)))
    monkeypatch.setitem(sys.modules,'mpi4py',SimpleNamespace(MPI=SimpleNamespace(SUM=0)))
    comm=SimpleNamespace(gather=lambda x,root=0:[x],bcast=lambda x,root=0:x,allreduce=lambda x,op=None:x)
    adapter=SimpleNamespace(comm=comm,rank=0,mapping=np.arange(2),parts=[np.arange(2)],ndof=2)
    delta=Vector(np.zeros(2))
    with pytest.raises(RuntimeError,match='true normal-equation qualification'):
        merit.residual_direction(Dense(np.eye(2)),Vector([1.,1.]),delta,adapter,None,
            np.zeros(2),Vector([1.,1.]),1.,cuda=False,budget=lambda:None,eta=1e-5)
    np.testing.assert_array_equal(delta.values,[0.,0.])
