from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hpc'))
import adaptive_reference_control as control
import launch_adaptive_reference as launch
from registry import atomic,artifact,sha


def policy(**changes):
    result=dict(end_time=.2,initial_dt=.1,maximum_dt=.1,minimum_dt=.0125,maximum_attempts=30,
        quadrature=[8,10],quadrature_ladder=[8,10,12],field_rtol=0.,field_atol=[.002,.002,.002],
        gradient_rtol=0.,gradient_atol=.01,maximum_log_j_stage_change=.25,maximum_isotropic_stage_change=.5)
    result.update(changes);return result


def zero(n=2):return control.Snapshot(0.,np.zeros(n),np.zeros(n),np.zeros(n))

def local(state):return np.broadcast_to(np.eye(3),(2,3,3)).copy()


def test_rejected_solve_does_not_modify_start_and_halves_time_step():
    start=zero();calls=[];commits=[];rejects=[]
    def advance(old,h,q,label):
        calls.append((old.time,h,q,label,old.u.copy()))
        if h>.05:raise control.TrialRejected('Newton failure')
        return control.Snapshot(old.time+h,old.u+h,old.v+h,old.a+h)
    final,summary=control.integrate(start,.1,advance,local,lambda a,b:0.,policy(),lambda:None,
        lambda a,b,h,r:commits.append((a.time,h,b[1].time)),lambda a,h,r:rejects.append((a.time,h,r)))
    assert final.time==pytest.approx(.2) and summary['accepted_physical_substeps']==8
    assert rejects[0][:2]==(0.,.1) and commits[0][:2]==(0.,.05)
    np.testing.assert_array_equal(start.u,0.)
    np.testing.assert_array_equal(calls[1][-1],0.)


def test_localized_dof_error_cannot_be_hidden_by_a_large_global_field():
    a=zero(100000);b=zero(100000);b.u[783]=.1
    score,_=control.field_error(a,b,rtol=0,atol=[1e-3,1e-3,1e-3])
    assert score==100


def test_quadrature_error_increases_degree_without_changing_histories_or_dt():
    start=zero();commits=[];rejects=[]
    def advance(old,h,q,label):
        shift=.02*h if q==8 else 0.
        return control.Snapshot(old.time+h,old.u+h+shift,old.v.copy(),old.a.copy())
    p=policy(end_time=.1,field_atol=[.0001,.002,.002])
    final,_=control.integrate(start,.1,advance,local,lambda a,b:0.,p,lambda:None,
        lambda a,b,h,r:commits.append((a.u.copy(),h,r)),lambda a,h,r:rejects.append((h,r)))
    assert commits[0][1]==.1 and commits[0][2]['quadrature_degrees']==[10,12]
    assert rejects[0][1]['quadrature_error']>1
    np.testing.assert_array_equal(commits[0][0],start.u)
    np.testing.assert_allclose(final.u,.1)


def test_unresolved_quadrature_at_maximum_degree_never_commits():
    def advance(old,h,q,label):
        return control.Snapshot(old.time+h,old.u+h*q,old.v.copy(),old.a.copy())
    with pytest.raises(control.TrialRejected,match='highest configured degree'):
        control.integrate(zero(),.1,advance,local,lambda a,b:0.,policy(),lambda:None,
            lambda *args:pytest.fail('Unresolved candidate was committed'),lambda *args:None)


def test_large_endpoint_stage_distortion_rejects_even_if_field_comparisons_pass():
    endpoint=np.diag([1.,1.,1e-5])[None]
    stress=np.diag([1.,1.,.4])[None]
    assert control.quality_error(endpoint,stress,8.023,policy())>1
    assert np.isinf(control.quality_error(np.full((1,3,3),np.nan),stress,8.023,policy()))


def test_expired_budget_preserves_latest_commit_and_is_not_a_numerical_failure():
    deadlines=[0];commits=[]
    def check():
        deadlines[0]+=1
        if len(commits):raise control.BudgetExpired('pause')
    def advance(old,h,q,label):return control.Snapshot(old.time+h,old.u+h,old.v.copy(),old.a.copy())
    with pytest.raises(control.BudgetExpired):
        control.integrate(zero(),.1,advance,local,lambda a,b:0.,policy(),check,
            lambda *args:commits.append(args),lambda *args:None)
    assert len(commits)==1 and commits[0][1][1].time==.1


def test_commit_failure_does_not_advance_the_accepted_state():
    start=zero()
    def advance(old,h,q,label):return control.Snapshot(old.time+h,old.u+h,old.v.copy(),old.a.copy())
    def commit(*args):raise OSError('disk full')
    with pytest.raises(OSError):control.integrate(start,.1,advance,local,lambda a,b:0.,policy(),lambda:None,commit,lambda *args:None)
    assert start.time==0 and np.all(start.u==0)


def test_snapshot_roundtrip_is_finite_and_does_not_mutate_initial_file(tmp_path):
    path=tmp_path/'checkpoint.npz';state=zero();state.a[:]=[1.,2.]
    launch.snapshot_file(path,state);saved=launch.restore(path)
    for name in ('u','v','a'):np.testing.assert_array_equal(getattr(saved,name),getattr(state,name))
    assert saved.time==0.


def test_acceptance_reader_requires_registered_physics_and_both_checkpoint_hashes(tmp_path,monkeypatch):
    root=tmp_path;run=root/'experiments/hpc/adaptive';run.mkdir(parents=True)
    monkeypatch.setattr(launch,'ROOT',root)
    launch.snapshot_file(run/'initial.npz',zero())
    step=run/'accepted/000001';step.mkdir(parents=True)
    launch.snapshot_file(step/'midpoint.npz',control.Snapshot(.05,np.ones(2),np.ones(2),np.ones(2)))
    launch.snapshot_file(step/'checkpoint.npz',control.Snapshot(.1,np.ones(2)*2,np.ones(2),np.ones(2)))
    row=dict(input_time_s=0.,time_s=.1,macro_dt_s=.1,input_checkpoint_sha256=sha(run/'initial.npz'),checkpoint_sha256=sha(step/'checkpoint.npz'),
        midpoint_sha256=sha(step/'midpoint.npz'),errors={'maximum_scaled_error':.1},
        physics=[dict(custom_force_N=1e-10,force_N=1e-10,proof=dict(endpoint={'passed':True},stress={'passed':True}))]*2)
    atomic(step/'accepted.json',row);artifact(root,run,step/'accepted.json','test')
    assert len(launch.accepted(run))==1
    (step/'midpoint.npz').write_bytes(b'wrong checkpoint')
    with pytest.raises(AssertionError):launch.accepted(run)


@pytest.mark.parametrize('outcome',['success','linear_failure','callback_budget'])
def test_native_tentative_advance_restores_all_fields_and_classifies_budget(tmp_path,monkeypatch,outcome):
    from adaptive_reference_native import Native
    fields={name:np.zeros(2) for name in ('u','u_old','v_old','a_old')}
    class Constant:
        def __init__(self,value):self.value=value
        def assign(self,value):self.value=value
        def __float__(self):return float(self.value)
    class Deadline:
        expired=False
        def __call__(self):pass
    deadline=Deadline()
    def update():
        dt=float(problem.parameters['dt']);a=(fields['u']-fields['u_old']-dt*fields['v_old']-(.5-.36)*dt*dt*fields['a_old'])/(.36*dt*dt)
        v=fields['v_old']+(1-.7)*dt*fields['a_old']+.7*dt*a
        fields.update(a_old=a,v_old=v,u_old=fields['u'].copy())
    problem=SimpleNamespace(parameters={'dt':Constant(.1),'p':Constant(0)},_beta=.36,_gamma=.7,
        _update_fields=update,solver=SimpleNamespace(solve=lambda:None,
            _snes=SimpleNamespace(getConvergedReason=lambda:2,getIterationNumber=lambda:0)))
    model=SimpleNamespace(problem=problem,assign=lambda name,value:fields.__setitem__(name,value.copy()),
        gather=lambda name:fields[name].copy())
    def solve(problem,adapter,hist,audit,options,trace,persist,**kwargs):
        assert options['initial_regularization']==0. and kwargs['policy']=='fixed_shift'
        assert 'residual_minimizing' not in kwargs
        fields['u'][:]=.01
        if outcome=='callback_budget':
            deadline.expired=True
            raise RuntimeError('PETSc error -1')
        if outcome=='linear_failure':raise RuntimeError('Unqualified GMRES direction')
        trace['custom_force_N']=1e-10
    monkeypatch.setitem(sys.modules,'lv_surrogate.physics.admissible_newton_gentle_shrink',SimpleNamespace(solve=solve))
    monkeypatch.setitem(sys.modules,'lv_surrogate.physics.reference_fem',SimpleNamespace(
        residual=lambda p:SimpleNamespace(norm=lambda kind:1e-10)))
    native=Native.__new__(Native)
    native.models={10:model};native.rank=0;native.comm=SimpleNamespace(bcast=lambda x,root=0:x)
    native.run=tmp_path;native.trials=0;native.deadline=deadline;native.derivatives_checked={10}
    native.plan={'newton_options':{}};native.audit=SimpleNamespace(pair=lambda *args:dict(endpoint={'passed':True},stress={'passed':True}))
    native.root_call=lambda f:f();native.pressure=lambda t:10.
    start=zero()
    if outcome=='success':
        result=native.advance(start,.1,10,'test')
        np.testing.assert_allclose(result.u,.01)
        np.testing.assert_allclose(result.a,.01/(.36*.1**2))
        assert result.time==.1 and result.details['quadrature_degree']==10
    else:
        error=control.BudgetExpired if outcome=='callback_budget' else control.TrialRejected
        with pytest.raises(error):native.advance(start,.1,10,'test')
    for value in fields.values():np.testing.assert_array_equal(value,0.)
    for name in ('u','v','a'):np.testing.assert_array_equal(getattr(start,name),0.)


def test_unregistered_tail_after_hard_kill_is_not_an_accepted_checkpoint(tmp_path,monkeypatch):
    run=tmp_path/'experiments/hpc/new';run.mkdir(parents=True)
    monkeypatch.setattr(launch,'ROOT',tmp_path)
    launch.snapshot_file(run/'initial.npz',zero())
    pending=run/'accepted/000001';pending.mkdir(parents=True)
    atomic(pending/'accepted.json',{'partial':'not registered'})
    assert launch.accepted(run)==[]


from test_hpc_refined_production import production_run
from test_hpc_bounded_solver import prepare_study


def test_new_pilot_starts_from_rest_and_preserves_original_source(production_run,monkeypatch):
    _,failed,target=prepare_study(production_run)
    root=failed.parents[2]
    source=failed.parent/'qualification-source';source.mkdir()
    source_plan=launch.load(failed/'plan.json')
    actual=Path(__file__).resolve().parents[1]
    for name in ('external','src','hpc'):
        (root/name).symlink_to(actual/name,target_is_directory=True)
    manifest=launch.load(target/'manifest.json');manifest['config']['displacement_degree']=2
    atomic(target/'manifest.json',manifest)
    source_plan['frozen'][str(target/'manifest.json')]=sha(target/'manifest.json')
    protocol=root/'protocol.json'
    atomic(protocol,dict(source_reference='cached-ref',admissible_newton=dict(
        custom_absolute_tolerance_N=1e-9,maximum_regularization=1e8,maximum_backtracks=20,armijo_constant=1e-4)))
    atomic(source/'plan.json',source_plan);artifact(root,source,source/'plan.json','source')
    atomic(source/'qualification-result.json',dict(complete=True,passed=False,branches=[dict(derivative_gate_passed=True,accepted=20)]*2))
    artifact(root,source,source/'qualification-result.json','source result')
    monkeypatch.setattr(launch,'ROOT',root);monkeypatch.setattr(launch,'TARGET',target)
    run=failed.parent/'adaptive'
    original=sha(target/'restart.npz')
    launch.initialize(run,source)
    plan=launch.load(run/'plan.json')
    assert not plan['original_reference_admission'] and not plan['automatic_successor']
    assert plan['policy']['end_time']==pytest.approx(.35) and plan['policy']['quadrature_ladder']==[8,10,12]
    assert plan['policy']['minimum_dt']==manifest['config']['dt_s']/64
    assert launch.load(run/'state.json')['accepted_physical_substeps']==0
    assert not (run/'initial.npz').exists(),'Initial histories must be derived on the compute node, not copied from a suspect late checkpoint'
    assert sha(target/'restart.npz')==original


def test_submission_keeps_existing_pbs_resources_without_an_automatic_successor(tmp_path,monkeypatch):
    run=tmp_path/'adaptive';run.mkdir();(run/'job.pbs').write_text('# test')
    atomic(run/'state.json',dict(slice=0,status='prepared'))
    calls=[]
    def qsub(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout='12345.pbs',stderr='',check_returncode=lambda:None)
    monkeypatch.setattr(launch.subprocess,'run',qsub)
    launch.submit(run)
    assert len(calls)==1
    assert 'select=1:ncpus=28:mem=240gb:ngpus=4' in calls[0] and 'walltime=168:00:00' in calls[0]
    assert not any('afterok' in item for item in calls[0])
