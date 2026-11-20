"""Qualification gates for a frozen dynamic step and one controlled time refinement."""
from pathlib import Path
import hashlib
import fcntl
import json
import os
import subprocess
import time
import numpy as np
from registry import atomic, artifact, sha


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values,dtype=np.float64).tobytes()).hexdigest()


def central_check(value, action, current, direction, steps, tolerance):
    expected=np.asarray(action(direction))
    denominator=max(float(np.linalg.norm(expected)),1e-30)
    rows=[]
    for h in steps:
        try:
            approximate=(np.asarray(value(current+h*direction))-np.asarray(value(current-h*direction)))/(2*h)
            error=float(np.linalg.norm(approximate-expected)/denominator)
        except (ValueError,FloatingPointError):
            error=float('inf')
        rows.append(dict(step=h,relative_error=error if np.isfinite(error) else None))
    good=[row['relative_error'] is not None and row['relative_error']<=tolerance for row in rows]
    return dict(passed=any(a and b for a,b in zip(good,good[1:])),tolerance=tolerance,samples=rows)


def determinant_gate(audit,elements,current,old):
    from refined_constrained_direction import point_constraint
    from cuda_determinant_direction import shape_gradients
    rows=[]
    for element in elements:
        dofs,gl=audit.maps[element],audit.gl[element]
        direction=np.sin(np.arange(30)+.7).reshape(10,3)
        for stage,values,factor in (('endpoint',current[dofs],1.),('stress',.4*old[dofs]+.6*current[dofs],.6)):
            for point in (np.full(4,.25),np.array([.7,.1,.1,.1]),np.array([.1,.1,.1,.7])):
                _,normal=point_constraint(values,factor,point,gl,audit.edges)
                shape=shape_gradients(point,gl,audit.edges)
                slope=float(normal@direction.ravel())
                row=central_check(lambda u:np.linalg.det(np.eye(3)+(values+factor*(u-values)).T@shape),
                    lambda d:slope,values,direction,[1e-6,3e-7,1e-7,3e-8,1e-8,3e-9],1e-5)
                rows.append(dict(element=int(element),stage=stage,barycentric=point.tolist(),**row))
    return dict(passed=all(row['passed'] for row in rows),checks=rows)


def derivative_gate(run,step,adapter,histories,audit,elements,current,checkpoint_sha,deadline):
    from lv_surrogate.physics.reference_fem import df,residual
    comm,rank,problem=adapter.comm,adapter.rank,adapter.problem
    saved={name:adapter.gather(name) for name in ('u_old','v_old','a_old')}
    matrix=df.PETScMatrix();problem._problem.J(matrix,problem.u.vector())
    J=df.as_backend_type(matrix).mat()
    base_residual=residual(problem)
    template=df.as_backend_type(base_residual).vec()
    x=template.duplicate();y=template.duplicate();w=template.duplicate();jt=template.duplicate()
    rows=[]
    try:
        def canonical(vector):
            arrays=comm.gather(np.array(vector.getArray(readonly=True)),root=0)
            result=None
            if rank==0:
                result=np.empty(adapter.ndof)
                for ids,values in zip(adapter.parts,arrays,strict=True):result[ids]=values
            return comm.bcast(result,root=0)
        def value(candidate):
            deadline();adapter.assign('u',candidate)
            result=residual(problem)
            return canonical(df.as_backend_type(result).vec())
        for phase in (.1,1.1,2.1):
            direction=np.sin(np.arange(adapter.ndof)*.173+phase)
            x.getArray()[:]=direction[adapter.mapping]
            J.mult(x,y);expected=canonical(y)
            rows.append(central_check(value,lambda d:expected,current,direction,
                        [1e-6,3e-7,1e-7,3e-8,1e-8,3e-9],1e-5))
        w.getArray()[:]=np.cos(adapter.mapping*.217+.3)
        J.multTranspose(w,jt)
        lhs,rhs=float(w.dot(y)),float(x.dot(jt))
        adjoint_error=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-30)
    finally:
        adapter.assign('u',current)
        for vector in (x,y,w,jt):vector.destroy()
    for name in saved:
        actual=adapter.gather(name)
        if rank==0:np.testing.assert_array_equal(saved[name],actual)
    report=None;error=None
    if rank==0:
        try:
            det=determinant_gate(audit,elements,current,histories['u_old'])
            report=dict(complete=True,passed=all(row['passed'] for row in rows) and det['passed'] and adjoint_error<1e-11,
                checkpoint_sha256=checkpoint_sha,current_sha256=digest(current),
                history_sha256={name:digest(values) for name,values in histories.items()},
                pressure_Pa=float(problem.parameters['p']),dt_s=float(problem.parameters['dt']),
                jacobian=rows,adjoint_relative_error=adjoint_error,determinant=det)
            atomic(step/'derivative-check.json',report)
            artifact(Path(run).parents[2],run,step/'derivative-check.json','Residual and determinant derivative gate')
            print('QUALIFICATION DERIVATIVES passed',report['passed'],flush=True)
        except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    error=comm.bcast(error,root=0)
    if error:raise RuntimeError(error)
    report=comm.bcast(report,root=0)
    if not report['passed']:raise RuntimeError('Solver qualification derivative gate failed; nonlinear solve disabled')
    return report


def restart_binding(checkpoint,pressure,dt):
    return dict(input_checkpoint_sha256=checkpoint,pressure_Pa=float(pressure),dt_s=float(dt))


def save_newton(path,binding,candidate,force,shift,iteration,elements):
    from compare_reference_timesteps_555 import save_npz
    save_npz(path,u=candidate,force_N=force,shift=shift,iteration=iteration,
             elements=np.array(sorted(elements),dtype=np.int64),binding=json.dumps(binding,sort_keys=True))


def load_newton(path,binding,shape):
    with np.load(path,allow_pickle=False) as data:
        assert json.loads(str(data['binding']))==binding,'Newton restart belongs to different histories or forcing'
        candidate=np.array(data['u'],copy=True)
        assert candidate.shape==shape and np.isfinite(candidate).all()
        shift,force=float(data['shift']),float(data['force_N'])
        assert np.isfinite(shift) and shift>0 and np.isfinite(force) and force>=0
        return dict(u=candidate,shift=shift,force_N=force,iteration=int(data['iteration']),
                    elements=data['elements'].astype(int).tolist())


def initialize(run,failed):
    import bounded_refined_solver as study
    import launch_refined_production as production
    study.initialize(run,failed)
    plan=production.load(run/'plan.json')
    plan['solver_qualification']=dict(branch='quarter',source_run=failed.name,allow_half_step=True,
        derivative_gate=True,independent_serial_verification=True,normal_preconditioner='local_ilu_normal',
        maximum_linear_recoveries=2)
    for path in (Path(__file__).resolve(),Path(__file__).with_name('qualified_normal_pc.py')):
        plan['frozen'][str(path)]=sha(path)
    plan['scope']='Bounded solver qualification: derivative gates, preconditioned constrained residual model, serial checkpoint verification and at most one half-step recovery. Original reference preserved.'
    atomic(run/'plan.json',plan);artifact(production.ROOT,run,run/'plan.json','Frozen solver qualification plan')
    script=run/'job.pbs'
    script.write_text(script.read_text().replace(" execute "," qualify "))
    artifact(production.ROOT,run,script,'Single bounded qualification controller')


def half_schedule(prefix,base_dt):
    start=prefix['endpoint_times'][19];end=prefix['endpoint_times'][23];dt=base_dt/8
    times=start+np.arange(1,9)*dt
    assert abs(times[-1]-end)<1e-12
    return dt,times,times-.4*dt


def initialize_half(run,source):
    import launch_refined_production as production
    source_plan=production.load(source/'plan.json')
    failed=production.ROOT/'experiments/hpc'/source_plan['solver_qualification']['source_run']
    initialize(run,failed)
    plan=production.load(run/'plan.json')
    plan['solver_qualification'].update(branch='half',allow_half_step=False)
    plan['solver_study']['stop_substep']=28
    plan['requested_substeps']=28
    plan['mixed_step_schedule']=True
    plan['solver_qualification']['quarter_branch']=source.name
    # The first twenty records remain the original quarter-step prefix. Only
    # future intervals are halved; histories are neither interpolated nor reset.
    pressure=production.load(run/'pressure.json')
    schedule=pressure['factors']['4']
    base_dt=production.load(production.TARGET/'manifest.json')['config']['dt_s']
    dt,times,stress=half_schedule(schedule,base_dt)
    plan['dt_s']=dt
    schedule.update(dt_s=dt,effective_factor=8,endpoint_times=schedule['endpoint_times'][:20]+times.tolist(),
        pressure_times=schedule['pressure_times'][:20]+stress.tolist(),pressures=schedule['pressures'][:20]+[None]*8)
    atomic(run/'pressure.json',pressure)
    plan['frozen'][str(run/'pressure.json')]=sha(run/'pressure.json')
    plan['solver_qualification']['pressure_preparation_required']=True
    atomic(run/'plan.json',plan);artifact(production.ROOT,run,run/'plan.json','Controlled half-step recovery plan')
    state=production.load(run/'state.json');state['requested_substeps']=28;atomic(run/'state.json',state)


def prepare_half_pressures(run):
    import launch_refined_production as production
    from lv_surrogate.physics.reference_fem import pressure_function,default_parameters_benchmark1
    plan=production.load(run/'plan.json');pressure=production.load(run/'pressure.json')
    schedule=pressure['factors']['4'];cfg=production.load(production.TARGET/'manifest.json')['config']
    parameters=default_parameters_benchmark1()
    for key in ('t_sys_pre','t_dias_pre','gamma'):parameters[key]*=cfg['timing_scale']
    values=cfg['amplitude']*pressure_function((0,cfg['duration_s']),parameters=parameters,
                t_eval=np.array(schedule['pressure_times'][20:]))
    assert np.isfinite(values).all()
    schedule['pressures'][20:]=values.tolist()
    atomic(run/'pressure.json',pressure)
    plan['frozen'][str(run/'pressure.json')]=sha(run/'pressure.json')
    plan['solver_qualification']['pressure_preparation_required']=False
    atomic(run/'plan.json',plan)
    artifact(production.ROOT,run,run/'pressure.json','Direct pressure evaluation at half-step stress times')
    artifact(production.ROOT,run,run/'plan.json','Pinned half-step forcing')


def verify(run):
    """Fresh serial CPU assembly of every new accepted checkpoint."""
    import launch_refined_production as production
    from lv_surrogate.physics.parallel_fem import ParallelP2
    from lv_surrogate.physics.reference_fem import residual
    from cofactor_proof import CofactorBernsteinAudit
    from production_cuda_native import use_positivity_depth
    from refined_positivity_policy import extend_unresolved_depth,install_adaptive_proof
    plan=production.load(run/'plan.json');schedule=production.load(run/'pressure.json')['factors']['4']
    cfg=dict(production.load(production.TARGET/'manifest.json')['config'],dt_s=schedule['dt_s'])
    protocol=production.load(Path(plan['protocol']))
    adapter=ParallelP2(production.ROOT,protocol['source_reference'],cfg)
    assert adapter.comm.size==1
    loaded={Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines() if '/libpetsc.so' in line}
    assert loaded=={(production.NATIVE/'lib/libpetsc.so').resolve()},loaded
    with np.load(production.TARGET/'inputs.npz',allow_pickle=False) as inputs:audit=CofactorBernsteinAudit(inputs,workers=8)
    use_positivity_depth(audit,20,24);extend_unresolved_depth(audit,32);install_adaptive_proof(audit)
    checks=[]
    try:
        rows=production.production_records(run,verify_all_checkpoints=True)
        for row in rows[20:]:
            index=row['substep']
            with (np.load(run/f'factor-4/substep-{index-1:03d}/checkpoint.npz',allow_pickle=False) as old,
                  np.load(run/f'factor-4/substep-{index:03d}/checkpoint.npz',allow_pickle=False) as new):
                for name in ('u_old','v_old','a_old'):adapter.assign(name,old[name])
                adapter.assign('u',new['u'])
                adapter.problem.parameters['p'].assign(row['pressure_Pa'])
                force=float(residual(adapter.problem).norm('l2'))
                proof=audit.pair(new['u'],old['u_old'])
                adapter.problem._update_fields()
                for name in ('u_old','v_old','a_old'):
                    np.testing.assert_allclose(adapter.gather(name),new[name],rtol=1e-7,atol=1e-10 if name!='a_old' else 1e-7)
                checks.append(dict(substep=index,serial_force_N=force,passed=np.isfinite(force) and force<1e-8,
                    endpoint_proof=proof['endpoint']['passed'],stress_proof=proof['stress']['passed']))
        report=dict(complete=True,passed=bool(checks) and all(row['passed'] and row['endpoint_proof'] and row['stress_proof'] for row in checks),checks=checks)
        atomic(run/'independent-verification.json',report)
        artifact(production.ROOT,run,run/'independent-verification.json','Independent serial CPU checkpoint and history verification')
    finally:audit.close()


def compare_branches(quarter,half):
    rows=[]
    for index in range(21,25):
        q=quarter/f'factor-4/substep-{index:03d}/accepted.json'
        h=half/f'factor-4/substep-{20+2*(index-20):03d}/accepted.json'
        if not q.exists() or not h.exists():continue
        with np.load(q.parent/'checkpoint.npz',allow_pickle=False) as a,np.load(h.parent/'checkpoint.npz',allow_pickle=False) as b:
            assert abs(float(a['time_s'])-float(b['time_s']))<1e-12
            row=dict(time_s=float(a['time_s']))
            for name in ('u','v_old','a_old'):
                row[name+'_relative_l2']=float(np.linalg.norm(a[name]-b[name])/max(np.linalg.norm(b[name]),1e-30))
            rows.append(row)
    return rows


def execute(run):
    assert os.environ.get('PBS_JOBID'), 'Use a PBS compute allocation'
    with (run/'qualification.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return execute_locked(run)


def execute_locked(run):
    import launch_refined_production as production
    started=time.perf_counter()
    plan=production.load(run/'plan.json')
    outcomes=[]
    candidates=[run]
    for branch in candidates:
        if branch!=run:
            try:
                if not branch.exists():initialize_half(branch,run)
                if production.load(branch/'plan.json')['solver_qualification'].get('pressure_preparation_required'):
                    env,_=production.native_environment()
                    subprocess.run([str(production.NATIVE/'bin/python'),str(Path(production.__file__)),
                        'qualification-prepare','--run-id',branch.name],env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True,timeout=3600)
            except Exception as exc:
                outcomes.append(dict(run=branch.name,derivative_gate_passed=False,solve_passed=False,
                    independently_verified=False,error=f'Half-step preparation failed: {type(exc).__name__}: {exc}'))
                break
        execution_error=None
        try:production.execute(branch)
        except Exception as exc:execution_error=f'{type(exc).__name__}: {exc}'
        if execution_error:
            outcomes.append(dict(run=branch.name,derivative_gate_passed=False,solve_passed=False,
                independently_verified=False,error=execution_error))
            break
        report=production.load(branch/'result.json')
        derivative=branch/'factor-4/substep-021/derivative-check.json'
        gate=production.load(derivative)['passed'] if derivative.exists() else False
        verified=False
        verification_started=time.perf_counter()
        if report['passed']:
            env,_=production.native_environment()
            verification_timeout=False
            try:
                with (branch/'verification.log').open('w') as log:
                    reply=subprocess.run([str(production.NATIVE/'bin/python'),str(Path(production.__file__)),
                    'qualification-verify','--run-id',branch.name],env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,timeout=2*3600)
            except (subprocess.TimeoutExpired,OSError):
                verification_timeout=True
            path=branch/'independent-verification.json'
            verified=not verification_timeout and reply.returncode==0 and path.exists() and production.load(path)['passed']
        branch_plan=production.load(branch/'plan.json')
        outcomes.append(dict(run=branch.name,dt_s=branch_plan.get('dt_s'),derivative_gate_passed=gate,accepted=report['accepted_after'],
            solve_passed=report['passed'],independently_verified=verified,elapsed_wall_seconds=report['elapsed_wall_seconds'],
            verification_seconds=time.perf_counter()-verification_started if report['passed'] else None))
        if report['passed'] and verified:break
        if branch==run and gate and not report['passed'] and plan['solver_qualification']['allow_half_step']:
            candidates.append(run.with_name(run.name+'-half'))
    passed=any(row['solve_passed'] and row['independently_verified'] and row['derivative_gate_passed'] for row in outcomes)
    result=dict(complete=True,passed=passed,branches=outcomes,elapsed_wall_seconds=time.perf_counter()-started,automatic_production_authorized=False,
        decision='review_bounded_success' if passed else 'investigate_model_mesh_or_solver; no production retry',
        common_time_comparison=compare_branches(run,candidates[-1]) if len(candidates)>1 else [],
        original_reference_steps=554,qualification_is_not_full_cycle_time_convergence=True)
    atomic(run/'qualification-result.json',result)
    artifact(production.ROOT,run,run/'qualification-result.json','End-to-end bounded solver qualification decision')
    state=production.load(run/'state.json');state['status']='qualification_passed' if passed else 'qualification_failed'
    atomic(run/'state.json',state)
