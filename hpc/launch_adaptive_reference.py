"""Separate error-controlled reference pilot from rest; never overwrites the original reference."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'hpc'),str(ROOT/'src')]
from registry import atomic,artifact,sha,read,event
from submit_probe_profile import registered,assert_idle
import launch_refined_production as legacy
from adaptive_reference_control import Snapshot,BudgetExpired,TrialRejected,integrate
from compare_reference_timesteps_555 import save_npz

TARGET=legacy.TARGET


def load(path):return json.loads(Path(path).read_text())


def initialize(run,source):
    for name in ('plan.json','qualification-result.json'):registered(ROOT,source,source/name)
    report=load(source/'qualification-result.json');source_plan=load(source/'plan.json')
    assert report['complete'] and not report['passed'] and len(report['branches'])==2
    assert all(item['derivative_gate_passed'] and item['accepted']==20 for item in report['branches'])
    for name,value in source_plan['frozen'].items():assert sha(Path(name))==value,name
    manifest=load(TARGET/'manifest.json');assert manifest['completed_steps']==554
    assert sha(TARGET/'restart.npz')==source_plan['source_checkpoint_sha256']
    protocol=load(Path(source_plan['protocol']))
    cfg=manifest['config'];base=cfg['dt_s']
    assert abs(base-.000625)<1e-15 and cfg['displacement_degree']==2
    refinement=ROOT/'experiments/hpc'/source_plan['parent_refinement']
    registered(ROOT,refinement,refinement/'patch.npz')
    frozen={str(path):sha(path) for path in (source/'plan.json',source/'qualification-result.json',
        TARGET/'manifest.json',TARGET/'restart.npz',TARGET/'inputs.npz',refinement/'patch.npz',Path(source_plan['protocol']))}
    for name in ('launch_adaptive_reference.py','adaptive_reference_native.py','adaptive_reference_control.py'):
        path=Path(__file__).with_name(name);frozen[str(path)]=sha(path)
    for name in ('external/cardiac_benchmark/src/cardiac_benchmark/problem.py',
        'external/cardiac_benchmark/src/cardiac_benchmark/material.py',
        'external/cardiac_benchmark/src/cardiac_benchmark/pressure_model.py',
        'src/lv_surrogate/physics/parallel_fem.py','src/lv_surrogate/physics/quadratic_solver.py',
        'src/lv_surrogate/physics/reference_fem.py','src/lv_surrogate/physics/admissible_newton_gentle_shrink.py',
        'src/lv_surrogate/physics/regularization_recovery_policy.py','src/lv_surrogate/physics/bernstein.py',
        'src/lv_surrogate/physics/adaptive_bernstein.py','hpc/cofactor_proof.py','hpc/refined_positivity_policy.py',
        'hpc/production_cuda_native.py','hpc/solver_qualification.py','hpc/compare_reference_timesteps_555.py'):
        path=ROOT/name;frozen[str(path)]=sha(path)
    plan=dict(config=cfg,source_reference=protocol['source_reference'],inputs=str((TARGET/'inputs.npz').relative_to(ROOT)),
        patch=str((refinement/'patch.npz').relative_to(ROOT)),source_run=source.name,
        original_checkpoint_sha256=sha(TARGET/'restart.npz'),gpu_library_sha256=sha(legacy.GPU/'lib/libpetsc.so'),
        newton_options=protocol['admissible_newton'],frozen=frozen,policy=dict(end_time=560*base,
            initial_dt=base,maximum_dt=base,minimum_dt=base/64,maximum_attempts=2000,
            quadrature=[8,10],quadrature_ladder=[8,10,12],field_rtol=1e-3,field_atol=[1e-7,1e-5,1e-2],gradient_rtol=1e-3,gradient_atol=1e-3,
            maximum_log_j_stage_change=.25,maximum_isotropic_stage_change=.5),
        slice_budget_seconds=8*3600,automatic_successor=False,original_reference_admission=False,
        scope='Separate adaptive pilot from rest through 0.35 s with consistent initial acceleration, unchanged material/loading/boundaries and alpha coefficients, full/half temporal error, Q8/Q10 quadrature error and endpoint-stage resolution gates. Numerical defaults require scientific validation; no claim that the physical problem is fixed.')
    run.mkdir();atomic(run/'plan.json',plan);artifact(ROOT,run,run/'plan.json','Adaptive reference pilot policy and physical source freeze')
    command=[str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'execute','--run-id',run.name]
    (run/'job.pbs').write_text('#!/bin/bash\nset -eu\ncd '+shlex.quote(str(ROOT))+'\nexec '+shlex.join(command)+'\n')
    artifact(ROOT,run,run/'job.pbs','Adaptive reference PBS controller')
    atomic(run/'state.json',dict(status='prepared',slice=0,accepted_physical_substeps=0,time_s=0.))


def submit(run):
    reply=subprocess.run(['qsub','-N','lv_adaptive','-q','iworkq','-l','select=1:ncpus=28:mem=240gb:ngpus=4',
        '-l','place=excl','-l','software=PBSJobApp','-l','walltime=168:00:00','-j','oe',
        '-o',str(run/f'pbs-{load(run/"state.json")["slice"]+1:03d}.log'),str(run/'job.pbs')],text=True,capture_output=True,cwd=ROOT)
    event(run,'job_submission',return_code=reply.returncode,stdout=reply.stdout.strip(),stderr=reply.stderr.strip())
    reply.check_returncode();job=reply.stdout.strip();assert re.fullmatch(r'\d+(?:\.[\w.-]+)?',job)
    state=load(run/'state.json');state.update(status='submitted',job=job);atomic(run/'state.json',state)
    print('Adaptive reference pilot job:',job,flush=True)


def prepare(run):
    from lv_surrogate.physics.reference_fem import pressure_function,default_parameters_benchmark1
    from compare_reference_timesteps_555 import pressure_reproduction
    plan=load(run/'plan.json');cfg=plan['config'];parameters=default_parameters_benchmark1()
    for key in ('t_sys_pre','t_dias_pre','gamma'):parameters[key]*=cfg['timing_scale']
    # Every allowed step is dyadic; alpha_f=.4 gives stress times on this exact
    # fifth-subdivision lattice. Evaluate the native pressure ODE once, without
    # introducing an interpolated approximation to the load between samples.
    quantum=plan['policy']['minimum_dt']/10
    count=int(round(cfg['duration_s']/quantum));times=np.arange(count+1)*quantum
    times[-1]=cfg['duration_s']
    pressure=cfg['amplitude']*pressure_function((0,cfg['duration_s']),parameters=parameters,t_eval=times)
    assert np.isfinite(pressure).all()
    with np.load(ROOT/plan['inputs'],allow_pickle=False) as inputs:
        indices=np.rint((inputs['time']-.4*cfg['dt_s'])/quantum).astype(int)
        audit=pressure_reproduction(pressure[indices],inputs['pressure'])
    save_npz(run/'forcing.npz',time=times,pressure=pressure,quantum=quantum)
    atomic(run/'forcing-audit.json',audit)
    artifact(ROOT,run,run/'forcing.npz','Frozen native-ODE dyadic loading lattice')
    artifact(ROOT,run,run/'forcing-audit.json','Original loading reproduction check')


def recorded(run,path):
    return any(item['kind']=='artifact' and item['path']==str(path.relative_to(ROOT)) for item in read(run))


def require_no_commits(run):
    assert not any(item['kind']=='artifact' and '/accepted/' in item.get('path','') for item in read(run)), 'Cannot replace initialization beneath accepted checkpoints'


def snapshot_file(path,state):save_npz(path,time_s=state.time,u=state.u,v=state.v,a=state.a)


def restore(path):
    with np.load(path,allow_pickle=False) as saved:
        state=Snapshot(float(saved['time_s']),saved['u'].copy(),saved['v'].copy(),saved['a'].copy())
    assert all(np.isfinite(value).all() and value.ndim==1 for value in (state.u,state.v,state.a))
    assert state.u.shape==state.v.shape==state.a.shape
    return state


def accepted(run):
    """Only a registered acceptance record makes a candidate pair durable."""
    rows=[];digest=sha(run/'initial.npz')
    ledger={item['path']:item['sha256'] for item in read(run) if item['kind']=='artifact'}
    paths=sorted((run/'accepted').glob('*/accepted.json'))
    previous_time=0.
    for index,path in enumerate(paths,1):
        relative=str(path.relative_to(ROOT))
        if relative not in ledger:
            assert not any(str(other.relative_to(ROOT)) in ledger for other in paths[index:]), 'Gap in durable acceptance chain'
            break  # A hard kill between JSON write and registry commit leaves a tentative tail.
        assert path.parent.name==f'{index:06d}'
        assert ledger[relative]==sha(path)
        row=load(path);assert row['input_checkpoint_sha256']==digest
        for name in ('midpoint','checkpoint'):
            assert sha(path.parent/f'{name}.npz')==row[name+'_sha256']
        assert row['input_time_s']==previous_time and row['macro_dt_s']>0
        assert abs(row['time_s']-previous_time-row['macro_dt_s'])<1e-12
        mid=restore(path.parent/'midpoint.npz');end=restore(path.parent/'checkpoint.npz')
        assert abs(mid.time-previous_time-row['macro_dt_s']/2)<1e-12 and end.time==row['time_s']
        assert row['errors']['maximum_scaled_error']<=1
        assert all(0<=item['custom_force_N']<1e-9 and 0<=item['force_N']<1e-8
                   and item['proof']['endpoint']['passed'] and item['proof']['stress']['passed'] for item in row['physics'])
        assert len(row['physics'])==2
        rows.append(row);digest=row['checkpoint_sha256'];previous_time=row['time_s']
    return rows


def worker(run):
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');rank=int(os.environ['PMI_RANK'])
    assert len(set(devices))==4
    os.environ['CUDA_VISIBLE_DEVICES']=devices[rank%4]
    from adaptive_reference_native import Native
    from mpi4py import MPI
    plan=load(run/'plan.json');started=time.time()
    class Deadline:
        expired=False
        def __call__(self):
            # MPI callbacks use this same object. Once expired, PETSc's callback
            # wrapping cannot erase the reason for stopping the current trial.
            self.expired=self.expired or MPI.COMM_WORLD.allreduce(time.time(),op=MPI.MAX)>=float(os.environ['ADAPTIVE_DEADLINE'])
            if self.expired:raise BudgetExpired('Adaptive slice budget exhausted')
    deadline=Deadline();native=None
    status='failed';failure=None;rows=[]
    try:
        native=Native(ROOT,run,plan,deadline)
        if rank==0:print('ADAPTIVE POLICY from_rest=True',plan['policy'],flush=True)
        if native.root_call(lambda:recorded(run,run/'initial.npz')):
            native.root_call(lambda:registered(ROOT,run,run/'initial.npz'))
            state=restore(run/'initial.npz')
        else:
            native.root_call(lambda:require_no_commits(run))
            state=native.initial()
            native.root_call(lambda:snapshot_file(run/'initial.npz',state))
            native.root_call(lambda:artifact(ROOT,run,run/'initial.npz','Rest initial displacement/velocity and mass-consistent acceleration'))
        rows=native.root_call(lambda:accepted(run))
        if rows:state=restore(run/'accepted'/f'{len(rows):06d}'/'checkpoint.npz')
        native.base_digest=rows[-1]['checkpoint_sha256'] if rows else sha(run/'initial.npz')
        def commit(old,pair,dt,errors):
            def persist():
                step=run/'accepted'/f'{len(rows)+1:06d}';step.mkdir(parents=True,exist_ok=True)
                snapshot_file(step/'midpoint.npz',pair[0]);snapshot_file(step/'checkpoint.npz',pair[1])
                physics=[]
                for point in pair:
                    detail=point.details
                    assert detail['quadrature_degree']==errors['quadrature_degrees'][1]
                    physics.append({key:detail[key] for key in ('custom_force_N','force_N','pressure_Pa','dt_s','quadrature_degree','proof','trace_path')})
                row=dict(time_s=pair[1].time,input_time_s=old.time,macro_dt_s=dt,
                    input_checkpoint_sha256=rows[-1]['checkpoint_sha256'] if rows else sha(run/'initial.npz'),
                    checkpoint_sha256=sha(step/'checkpoint.npz'),midpoint_sha256=sha(step/'midpoint.npz'),physics=physics,errors=errors)
                atomic(step/'accepted.json',row);artifact(ROOT,run,step/'accepted.json','Error-controlled pair of physical substeps')
                print('ADAPTIVE ACCEPTED substeps',2*(len(rows)+1),'time',pair[1].time,'error',errors['maximum_scaled_error'],flush=True)
                return row
            row=native.root_call(persist);rows.append(row);native.base_digest=row['checkpoint_sha256']
        def reject(old,dt,error):
            if native.rank==0:
                event(run,'trial_rejected',time_s=old.time,macro_dt_s=dt,errors=error)
                print('ADAPTIVE REJECTED time',old.time,'dt',dt,error,flush=True)
        dt=rows[-1]['macro_dt_s'] if rows else plan['policy']['initial_dt']
        if rows:plan['policy']['quadrature']=rows[-1]['errors']['quadrature_degrees']
        _,counts=integrate(state,dt,native.advance,native.local_fields,native.stage_quality,plan['policy'],deadline,commit,reject)
        status='pilot_complete'
    except BudgetExpired:status='paused'
    except Exception as exc:
        failure=f'{type(exc).__name__}: {exc}'
        atomic(run/f'failure-rank-{rank:02d}.json',dict(failure=failure,slice=int(os.environ['ADAPTIVE_SLICE'])))
        MPI.COMM_WORLD.Abort(1)
    finally:
        if rank==0:
            atomic(run/f'worker-result-{int(os.environ["ADAPTIVE_SLICE"]):03d}.json',dict(status=status,failure=failure,accepted_physical_substeps=2*len(rows),
                time_s=rows[-1]['time_s'] if rows else 0.,elapsed_seconds=time.time()-started,
                original_reference_admission=False,automatic_successor=False))
        if native is not None:native.close()


def execute(run):
    assert os.environ.get('PBS_JOBID')
    with (run/'execution.lock').open('a') as lock,(TARGET/'generation.lock').open('a') as generation:
        for handle in (lock,generation):fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=load(run/'plan.json');registered(ROOT,run,run/'plan.json');registered(ROOT,run,run/'job.pbs')
        for name,value in plan['frozen'].items():assert sha(Path(name))==value,name
        assert sha(legacy.GPU/'lib/libpetsc.so')==plan['gpu_library_sha256']
        assert len(set(Path(os.environ['PBS_NODEFILE']).read_text().splitlines()))==1
        state=load(run/'state.json');state.update(status='running',slice=state['slice']+1);atomic(run/'state.json',state)
        env,pkg=legacy.native_environment();base=[str(legacy.NATIVE/'bin/python'),str(Path(__file__).resolve())]
        if not recorded(run,run/'forcing.npz'):
            require_no_commits(run)
            with (run/f'prepare-{state["slice"]:03d}.log').open('x') as log:
                subprocess.run([*base,'prepare','--run-id',run.name],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
        registered(ROOT,run,run/'forcing.npz')
        visible=env.get('CUDA_VISIBLE_DEVICES') or subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip().replace('\n',',')
        assert len(set(visible.split(',')))==4
        env.update(CUDA_VISIBLE_DEVICES=visible,PETSC_OPTIONS='-use_gpu_aware_mpi 0',ADAPTIVE_SLICE=str(state['slice']),
            PKG_CONFIG_PATH=f'{legacy.GPU}/lib/pkgconfig:{pkg}',PKG_CONFIG_LIBDIR=f'{legacy.GPU}/lib/pkgconfig:{pkg}',
            ADAPTIVE_DEADLINE=str(time.time()+plan['slice_budget_seconds']-120))
        with (run/f'worker-{state["slice"]:03d}.log').open('x') as log:
            reply=subprocess.run([str(legacy.NATIVE/'bin/mpiexec'),'-n','12','/usr/bin/env',f'LD_PRELOAD={legacy.GPU}/lib/libpetsc.so',
                *base,'worker','--run-id',run.name],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if reply.returncode:
            rows=accepted(run) if (run/'initial.npz').exists() else []
            result=dict(status='failed',failure=f'Native worker exited {reply.returncode}; inspect worker-{state["slice"]:03d}.log',accepted_physical_substeps=2*len(rows),time_s=rows[-1]['time_s'] if rows else 0.)
        else:
            result=load(run/f'worker-result-{state["slice"]:03d}.json')
        if (run/'initial.npz').exists():rows=accepted(run);assert len(rows)*2==result['accepted_physical_substeps']
        assert sha(TARGET/'restart.npz')==plan['original_checkpoint_sha256']
        assert sha(TARGET/'manifest.json')==plan['frozen'][str(TARGET/'manifest.json')]
        if result['status']=='pilot_complete':
            with (run/'verification.log').open('w') as log:
                verified=subprocess.run([*base,'verify','--run-id',run.name],cwd=ROOT,env=dict(legacy.native_environment()[0],CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
            result['independent_serial_verified']=verified.returncode==0 and (run/'independent-verification.json').exists() and load(run/'independent-verification.json')['passed']
            result['status']='pilot_qualified' if result['independent_serial_verified'] else 'verification_failed'
        result.update(original_reference_admission=False,automatic_successor=False,
            numerical_policy_requires_scientific_validation=True)
        state.update(result);atomic(run/'state.json',state);atomic(run/'result.json',result)
        artifact(ROOT,run,run/'result.json','Adaptive pilot outcome; no full-cycle qualification')


def verify(run):
    """Independently reassemble the last three accepted pairs on serial CPU."""
    from lv_surrogate.physics.parallel_fem import ParallelP2
    from lv_surrogate.physics.reference_fem import df,residual
    from cofactor_proof import CofactorBernsteinAudit
    from production_cuda_native import use_positivity_depth
    from refined_positivity_policy import extend_unresolved_depth,install_adaptive_proof
    plan=load(run/'plan.json');rows=accepted(run);assert len(rows)>=3
    loaded={Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines() if '/libpetsc.so' in line}
    assert loaded=={(legacy.NATIVE/'lib/libpetsc.so').resolve()},loaded
    with np.load(ROOT/plan['inputs'],allow_pickle=False) as data:audit=CofactorBernsteinAudit(data,workers=8)
    use_positivity_depth(audit,20,24);extend_unresolved_depth(audit,32);install_adaptive_proof(audit)
    checks=[];model=None;degree=None
    try:
        for index in range(max(0,len(rows)-3),len(rows)):
            row=rows[index]
            previous=restore(run/'accepted'/f'{index:06d}'/'checkpoint.npz') if index else restore(run/'initial.npz')
            step=run/'accepted'/f'{index+1:06d}'
            for filename,detail in zip(('midpoint.npz','checkpoint.npz'),row['physics'],strict=True):
                if degree!=detail['quadrature_degree']:
                    degree=detail['quadrature_degree']
                    model=ParallelP2(ROOT,plan['source_reference'],dict(plan['config'],dt_s=detail['dt_s'],quadrature_degree=degree))
                    assert model.comm.size==1
                    model.problem.parameters['dt']=df.Constant(detail['dt_s']);model.problem._init_forms()
                    model.problem._problem._F=df.Form(model.problem._problem._F,form_compiler_parameters={'quadrature_degree':degree})
                problem=model.problem;problem.parameters['dt'].assign(detail['dt_s']);problem.parameters['p'].assign(detail['pressure_Pa'])
                candidate=restore(step/filename)
                for name,value in (('u',candidate.u),('u_old',previous.u),('v_old',previous.v),('a_old',previous.a)):model.assign(name,value)
                force=float(residual(problem).norm('l2'));proof=audit.pair(candidate.u,previous.u)
                problem._update_fields()
                for name,value in (('u_old',candidate.u),('v_old',candidate.v),('a_old',candidate.a)):
                    np.testing.assert_allclose(model.gather(name),value,rtol=1e-7,atol=1e-7 if name=='a_old' else 1e-10)
                checks.append(dict(time_s=candidate.time,serial_force_N=force,passed=bool(np.isfinite(force) and force<1e-8 and proof['endpoint']['passed'] and proof['stress']['passed'])))
                previous=candidate
        report=dict(complete=True,passed=all(row['passed'] for row in checks),checks=checks,
                    scope='Fresh serial CPU force, whole-cell positivity and dynamic-history reassembly of the last three accepted pairs; all prior pairs retain hash-checked temporal/quadrature/error records.')
        atomic(run/'independent-verification.json',report)
        artifact(ROOT,run,run/'independent-verification.json','Serial CPU verification of six consecutive adaptive substeps')
    finally:audit.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('submit','resume','execute','prepare','verify','worker'))
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--from-qualification',default='cmbbe-solver-qualification-01')
    args=parser.parse_args()
    for value in (args.run_id,args.from_qualification):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',value):parser.error('Invalid run ID')
    run=ROOT/'experiments/hpc'/args.run_id
    if args.action in ('submit','resume'):
        with (ROOT/'.hpc/profile-submission.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert_idle(ROOT)
            if args.action=='submit':initialize(run,ROOT/'experiments/hpc'/args.from_qualification)
            else:assert load(run/'state.json')['status'] in ('paused','running','submitted'),'Only a budget pause or interrupted allocation can resume after the idle-job check'
            submit(run)
    elif args.action=='execute':execute(run)
    elif args.action=='prepare':prepare(run)
    elif args.action=='verify':verify(run)
    else:worker(run)
