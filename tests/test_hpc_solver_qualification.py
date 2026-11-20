from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'hpc'))
import solver_qualification as qualification
import launch_refined_production as production
from qualified_normal_pc import LocalNormalPC
from lv_surrogate.physics.bernstein import BernsteinAudit
from test_hpc_refined_production import production_run
from test_hpc_bounded_solver import prepare_study,Vector


def test_derivative_gate_requires_two_adjacent_accurate_steps_and_detects_wrong_jacobian():
    matrix=np.array([[2.,3.],[-1.,2.]])
    x=np.array([.2,-.3]);direction=np.array([.3,.7])
    value=lambda x:matrix@x+np.array([x[0]**3,x[1]**3])
    J=matrix+np.diag(3*x*x)
    steps=[1e-3,1e-4,1e-5,1e-6]
    assert qualification.central_check(value,lambda d:J@d,x,direction,steps,1e-5)['passed']
    assert not qualification.central_check(value,lambda d:matrix@d,x,direction,steps,1e-5)['passed']


def test_determinant_gate_checks_both_history_scaled_stages():
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    audit=BernsteinAudit(dict(coords=vertices,cells=np.array([[0,1,2,3]]),
                             canonical_cell_dofs=np.arange(30).reshape(1,10,3)))
    current=np.sin(np.arange(30))*.001;old=current*.5
    result=qualification.determinant_gate(audit,[0],current,old)
    assert result['passed'] and len(result['checks'])==6
    assert {row['stage'] for row in result['checks']}=={'endpoint','stress'}


def test_ilu_product_is_spd_and_improves_a_badly_scaled_normal_system():
    from scipy.sparse import diags
    B=diags(np.logspace(-3,3,20),format='csr')
    block=SimpleNamespace(getValuesCSR=lambda:(B.indptr,B.indices,B.data))
    J=SimpleNamespace(getDiagonalBlock=lambda:block)
    mu=1e-4
    pc=LocalNormalPC(J,np.ones(20),mu)
    inverse=[]
    for row in np.eye(20):
        output=Vector(np.zeros(20));pc.apply(None,Vector(row),output);inverse.append(output.values)
    inverse=np.array(inverse).T
    np.testing.assert_allclose(inverse,inverse.T,atol=1e-12)
    assert np.linalg.eigvalsh(inverse).min()>0
    H=(B.T@B).toarray()+mu*np.eye(20)
    eigen=np.linalg.eigvals(inverse@H)
    assert eigen.max()/eigen.min() < np.linalg.cond(H)/1000
    assert pc.detail['kind']=='local_ilu_normal'
    pc.close()


def test_saved_newton_iterate_is_bound_to_histories_pressure_and_dt(tmp_path):
    path=tmp_path/'newton.npz'
    binding=qualification.restart_binding('accepted-history-sha',14169.,1/6400)
    qualification.save_newton(path,binding,np.arange(12.),.2,.01,22,[813578,813580])
    saved=qualification.load_newton(path,binding,(12,))
    np.testing.assert_array_equal(saved['u'],np.arange(12.))
    assert saved['iteration']==22 and saved['elements']==[813578,813580]
    with pytest.raises(AssertionError,match='different histories'):
        qualification.load_newton(path,dict(binding,dt_s=1/12800),(12,))


def test_half_recovery_changes_only_future_times_and_preserves_checkpoint(production_run):
    _,failed,target=prepare_study(production_run)
    run=failed.parent/'qualification'
    qualification.initialize(run,failed)
    plan=production.load(run/'plan.json')
    assert ' qualify ' in (run/'job.pbs').read_text()
    assert plan['solver_qualification']['derivative_gate']
    assert plan['solver_qualification']['normal_preconditioner']=='local_ilu_normal'
    half=failed.parent/'qualification-half'
    qualification.initialize_half(half,run)
    halfplan=production.load(half/'plan.json')
    assert halfplan['mixed_step_schedule'] and halfplan['requested_substeps']==28
    assert halfplan['dt_s']==1/12800 and not halfplan['solver_qualification']['allow_half_step']
    a=production.load(run/'pressure.json')['factors']['4']
    b=production.load(half/'pressure.json')['factors']['4']
    assert b['endpoint_times'][:20]==a['endpoint_times'][:20]
    assert b['pressures'][:20]==a['pressures'][:20]
    assert b['endpoint_times'][21::2]==a['endpoint_times'][20:]
    assert b['endpoint_times'][-1]==a['endpoint_times'][-1]
    assert sha_checkpoint(run)==sha_checkpoint(half)
    assert production.load(target/'manifest.json')['completed_steps']==554


def sha_checkpoint(run):return production.sha(run/'factor-4/substep-020/checkpoint.npz')


@pytest.mark.parametrize('gate,passed,verified,expected_branches',[
    (False,False,False,1),(True,False,False,2),(True,True,True,1),(True,True,False,1)])
def test_controller_never_passes_without_derivatives_solve_and_serial_verification(
        tmp_path,monkeypatch,gate,passed,verified,expected_branches):
    from registry import atomic
    root=tmp_path;run=root/'experiments/hpc/qualification';run.mkdir(parents=True)
    plan=dict(solver_qualification=dict(allow_half_step=True))
    atomic(run/'plan.json',plan);atomic(run/'state.json',{})
    monkeypatch.setattr(production,'ROOT',root)
    monkeypatch.setattr(production,'native_environment',lambda:({},''))
    calls=[]
    def execute(branch):
        calls.append(branch)
        atomic(branch/'factor-4/substep-021/derivative-check.json',dict(passed=gate))
        atomic(branch/'result.json',dict(passed=passed,accepted_after=24 if passed else 20,elapsed_wall_seconds=100))
    def initialize_half(branch,parent):
        branch.mkdir();atomic(branch/'plan.json',plan);atomic(branch/'state.json',{})
    def verify_reply(command,**kwargs):
        branch=run.with_name(command[-1])
        atomic(branch/'independent-verification.json',dict(passed=verified))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(production,'execute',execute)
    monkeypatch.setattr(qualification,'initialize_half',initialize_half)
    monkeypatch.setattr(qualification.subprocess,'run',verify_reply)
    monkeypatch.setattr(qualification,'compare_branches',lambda a,b:[])
    monkeypatch.setenv('PBS_JOBID','12345.pbs')
    qualification.execute(run)
    result=production.load(run/'qualification-result.json')
    assert len(calls)==expected_branches
    assert result['passed']==(gate and passed and verified)
    assert result['automatic_production_authorized'] is False


@pytest.mark.parametrize('wrong',[False,True])
def test_native_derivative_gate_restores_fields_and_blocks_wrong_assembled_jacobian(tmp_path,monkeypatch,wrong):
    from test_hpc_bounded_solver import Dense
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    audit=BernsteinAudit(dict(coords=vertices,cells=np.array([[0,1,2,3]]),
                             canonical_cell_dofs=np.arange(30).reshape(1,10,3)))
    current=np.sin(np.arange(30))*.001
    histories=dict(u_old=current*.5,v_old=np.zeros(30),a_old=np.ones(30))
    fields=dict(histories,u=current.copy())
    mapping=np.arange(30)[::-1]
    J=Dense(np.diag(2+3*current[mapping]**2+(1. if wrong else 0.)))
    comm=SimpleNamespace(gather=lambda x,root=0:[x],bcast=lambda x,root=0:x)
    problem=SimpleNamespace(parameters={'p':14169.,'dt':1/6400},u=SimpleNamespace(vector=lambda:None),
        _problem=SimpleNamespace(J=lambda *args:None))
    adapter=SimpleNamespace(comm=comm,rank=0,ndof=30,mapping=mapping,parts=[mapping],problem=problem,
        gather=lambda name:fields[name].copy(),assign=lambda name,value:fields.__setitem__(name,value.copy()))
    fake_df=SimpleNamespace(PETScMatrix=lambda:None,
        as_backend_type=lambda obj:SimpleNamespace(mat=lambda:J,vec=lambda:obj))
    monkeypatch.setitem(sys.modules,'lv_surrogate.physics.reference_fem',SimpleNamespace(df=fake_df,
        residual=lambda p:Vector((2*fields['u']+fields['u']**3)[mapping])))
    run=tmp_path/'experiments/hpc/qualification';step=run/'factor-4/substep-021';step.mkdir(parents=True)
    if wrong:
        with pytest.raises(RuntimeError,match='derivative gate failed'):
            qualification.derivative_gate(run,step,adapter,histories,audit,[0],current,'fixed-checkpoint',lambda:None)
    else:
        result=qualification.derivative_gate(run,step,adapter,histories,audit,[0],current,'fixed-checkpoint',lambda:None)
        assert result['passed'] and result['adjoint_relative_error']<1e-11
    np.testing.assert_array_equal(fields['u'],current)
    for name,expected in histories.items():np.testing.assert_array_equal(fields[name],expected)
    assert production.load(step/'derivative-check.json')['passed']==(not wrong)
