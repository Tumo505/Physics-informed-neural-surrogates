"""Bounded continuation from rejected-reference checkpoint; no data replacement."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'src'))
from lv_surrogate.physics.reference_fem import setup, residual, df, pressure_function, default_parameters_benchmark1
from lv_surrogate.physics.quadratic_solver import make_quadratic_problem
from lv_surrogate.physics.quadratic import QuadraticKinematics

ap = argparse.ArgumentParser()
ap.add_argument('--protocol', default='configs/physics/dynamic_quadratic_strict_probe.json')
args = ap.parse_args()
protocol_path = P/args.protocol
protocol = json.loads(protocol_path.read_text())
if 'required_audit' in protocol:
    audit=json.loads((P/protocol['required_audit']).read_text())
    assert audit['complete'] and audit['passed'], 'Recovery histories must pass independent audit before use'
root = P/'data/processed'/protocol['reference']
assert (root/'failure.json').exists()
m = json.loads((root/'manifest.json').read_text()); cfg = m['config']
saved = dict(np.load(root/'restart.npz')); inputs = np.load(root/'inputs.npz')
start_step = int(saved['completed_steps']); assert start_step == m['completed_steps']
resume_description = {}
if 'resume_probe_protocol' in protocol:
    previous_protocol = json.loads((P/protocol['resume_probe_protocol']).read_text())
    previous_result = json.loads((P/previous_protocol['output']).read_text())
    previous_checkpoint = P/previous_protocol['checkpoint']
    assert previous_protocol['reference'] == protocol['reference']
    assert previous_protocol.get('timestep_divisor',1) == protocol.get('timestep_divisor',1) == 1
    assert hashlib.sha256(previous_checkpoint.read_bytes()).hexdigest() == previous_result['probe_checkpoint_sha256']
    assert previous_result['checkpoint_sha256'] == hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest()
    saved = dict(np.load(previous_checkpoint))
    accepted_count = int(saved['completed_probe_steps'])
    assert all(r['accepted'] for r in previous_result['records'][:accepted_count])
    start_step = previous_result['start_step']+accepted_count
    resume_description = dict(resumed_checkpoint_sha256=previous_result['probe_checkpoint_sha256'],
                              resumed_accepted_probe_steps=accepted_count)
result = dict(complete=False, engineering_only=True, protocol=protocol, records=[],
    protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    reference_config_sha256=m['config_sha256'], start_step=start_step,
    checkpoint_sha256=hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
result.update(resume_description)
output = P/protocol.get('output', 'experiments/metrics/dynamic_quadratic_strict_stopping_probe.json')
lock = (P/'experiments/logs'/f'{output.stem}.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
if output.exists():
    raise RuntimeError('Probe output already exists; preserve it and use a separate protocol/output for another attempt')
divisor = int(protocol.get('timestep_divisor', 1)); assert divisor >= 1
dt = cfg['dt_s']/divisor
times = float(inputs['time'][start_step-1])+dt*np.arange(1, protocol['steps_after_checkpoint']+1)
if divisor == 1:
    pressures = inputs['pressure'][start_step:start_step+len(times)]
else:
    par = default_parameters_benchmark1()
    for key in ['t_sys_pre', 't_dias_pre', 'gamma']:
        par[key] *= cfg['timing_scale']
    pressures = cfg['amplitude']*pressure_function((0, cfg['duration_s']), parameters=par, t_eval=times-.4*dt)
result.update(dt_s=dt, full_cycle=False, changed_time_discretization=divisor!=1)
def persist():
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
persist()
df.PETScOptions.set('mat_mumps_icntl_23', protocol['mumps_memory_limit_MB'])
geo, material, base = setup(dt=dt, geometry_scales=cfg['geometry_scales'],
                            refinement_level=cfg['refinement_level'])
parameters = dict(base.parameters); parameters['function_space'] = 'P_2'
df.parameters['form_compiler']['quadrature_degree'] = cfg['quadrature_degree']
method = protocol.get('native_solver','mumps')
problem = make_quadratic_problem(geo, material, parameters, method,
    linear_maximum_iterations=protocol.get('linear_maximum_iterations',1000),
    gmres_restart=protocol.get('gmres_restart',100))
settings = problem.solver._solver.parameters
settings['absolute_tolerance'] = protocol['absolute_tolerance_N']
settings['relative_tolerance'] = protocol['relative_tolerance']
settings['maximum_iterations'] = protocol['maximum_newton_iterations']
settings['line_search'] = 'bt'
if 'nonlinear_method' in protocol:
    settings['method']=protocol['nonlinear_method']
np.testing.assert_allclose(problem.u_space.tabulate_dof_coordinates(), inputs['dof_coordinates'], rtol=0, atol=1e-14)
for name in ['u', 'u_old', 'v_old', 'a_old']:
    field = getattr(problem, name); field.vector().set_local(saved[name]); field.vector().apply('insert')
guess_source=None
if 'initial_guess_reference' in protocol:
    source_root=P/'data/processed'/protocol['initial_guess_reference']
    source_manifest=json.loads((source_root/'manifest.json').read_text());sc=source_manifest['config']
    assert source_manifest['complete'] and not (source_root/'failure.json').exists()
    for name,digest in source_manifest['output_sha256'].items():
        assert hashlib.sha256((source_root/name).read_bytes()).hexdigest()==digest
    for key in ['dt_s','duration_s','displacement_degree','geometry_scales','amplitude','timing_scale','quadrature_degree']:
        assert sc[key]==cfg[key], key
    source_inputs=np.load(source_root/'inputs.npz')
    source_geo,_,source_base=setup(dt=dt,geometry_scales=sc['geometry_scales'],refinement_level=sc['refinement_level'])
    source_space=df.VectorFunctionSpace(source_geo.mesh,'CG',2)
    np.testing.assert_allclose(source_space.tabulate_dof_coordinates(),source_inputs['dof_coordinates'],atol=1e-14,rtol=0)
    np.testing.assert_array_equal(source_geo.mesh.cells(),source_inputs['cells'])
    source_fields=np.load(source_root/'displacement_coefficients.npy',mmap_mode='r')
    guess_source=(source_space,source_inputs,source_fields)
    result['initial_guess_reference_output_sha256']=source_manifest['output_sha256'];persist()
kin = QuadraticKinematics(problem.u_space, inputs['qpoints'], inputs['qweights'])
checkpoint_path = P/protocol['checkpoint'] if 'checkpoint' in protocol else None
fields = None
if 'fields' in protocol:
    fields = np.lib.format.open_memmap(P/protocol['fields'], mode='w+', dtype=np.float64,
        shape=(protocol['steps_after_checkpoint'], problem.u_space.dim()))
started = time.perf_counter()
history = []
if protocol.get('inexact_newton',False):
    assert method == 'gamg'
    problem.solver._snes.setUseEW(True,version=2,rtol_0=.01,rtol_max=.1)
def monitor(snes, iteration, norm):
    history.append(dict(iteration=int(iteration), residual_N=float(norm) if np.isfinite(norm) else None,
        KSP_rtol=float(snes.getKSP().getTolerances()[0])))
    print('STRICT PROBE', iteration, norm, flush=True)
    if 'iterate_checkpoint' in protocol:
        iterate_path = P/protocol['iterate_checkpoint']
        temporary = iterate_path.with_suffix('.pending.npz')
        np.savez_compressed(temporary, accepted=False,
            protocol_sha256=result['protocol_sha256'],
            original_checkpoint_sha256=result['checkpoint_sha256'],
            time_s=result['last_attempt_time_s'], step=i,
            u=snes.getSolution().getArray(readonly=True).copy(),
            **{name:getattr(problem,name).vector().get_local() for name in ['u_old','v_old','a_old']})
        temporary.replace(iterate_path)
    if time.perf_counter()-started > protocol['wall_budget_seconds']:
        result['budget_exhausted'] = True
        persist()
        raise RuntimeError('Declared engineering wall budget exceeded')
problem.solver._snes.setMonitor(monitor)
try:
    for offset, (time_s, pressure) in enumerate(zip(times, pressures)):
        i = start_step+offset
        result['last_attempt_time_s'] = float(time_s)
        old = problem.u_old.vector().get_local().copy()
        velocity = problem.v_old.vector().get_local()
        acceleration = problem.a_old.vector().get_local()
        best = old; best_R = float('inf')
        problem.parameters['p'].assign(float(pressure))
        for guess in [old, old+dt*velocity, old+dt*velocity+.5*dt**2*acceleration]:
            problem.u.vector().set_local(guess); problem.u.vector().apply('insert')
            R = float(residual(problem).norm('l2'))
            if np.isfinite(R) and R < best_R:
                best = guess; best_R = R
        if guess_source is not None:
            source_space,source_inputs,source_fields=guess_source
            np.testing.assert_allclose(source_inputs['time'][i],time_s,atol=1e-12,rtol=0)
            np.testing.assert_allclose(source_inputs['pressure'][i],pressure,atol=1e-10,rtol=0)
            source_field=df.Function(source_space);source_field.vector().set_local(source_fields[i]);source_field.vector().apply('insert')
            prolonged=df.interpolate(source_field,problem.u_space)
            problem.u.assign(prolonged);R=float(residual(problem).norm('l2'))
            improvement=best_R/max(R,1e-300) if np.isfinite(R) else 0.
            result['initial_guess_check']=dict(original_predictor_residual_N=best_R if np.isfinite(best_R) else None,
                prolonged_residual_N=R if np.isfinite(R) else None,improvement_factor=improvement,
                source_reference=protocol['initial_guess_reference'],source_step=i,initialization_only=True)
            persist()
            assert np.isfinite(R) and improvement>=protocol.get('minimum_initial_residual_improvement_factor',1), 'Prolongation did not meet predeclared initialization improvement gate'
            best=prolonged.vector().get_local();best_R=R
        problem.u.vector().set_local(best); problem.u.vector().apply('insert')
        history.clear(); tick = time.perf_counter()
        problem.solver.solve()
        snes = problem.solver._snes
        if 'nonlinear_method' in protocol:
            assert snes.getType()==protocol['nonlinear_method'], 'Requested nonlinear method was not effective'
        if protocol.get('inexact_newton',False):
            assert snes.getUseEW(), 'Requested inexact Newton was not effective'
        R = float(residual(problem).norm('l2')); u = problem.u.vector().get_local()
        endpoint, _ = kin.audit(u); stress, _ = kin.audit(.4*old+.6*u)
        accepted = bool(snes.getConvergedReason()>0 and np.isfinite(R)
                        and R<min(1e-6,protocol['absolute_tolerance_N'])
                        and np.isfinite(u).all() and min(endpoint, stress)>0)
        result['records'].append(dict(step=i, time_s=float(time_s), accepted=accepted,
            SNES_reason=int(snes.getConvergedReason()), residual_N=R if np.isfinite(R) else None,
            KSP_reason=int(snes.getKSP().getConvergedReason()),
            effective_KSP=snes.getKSP().getType(), effective_PC=snes.getKSP().getPC().getType(),
            effective_SNES=snes.getType(),
            EW_effective=bool(snes.getUseEW()),
            last_linear_iterations=int(snes.getKSP().getIterationNumber()),
            minimum_endpoint_sampled_detF=endpoint, minimum_stress_sampled_detF=stress,
            nonlinear_seconds=time.perf_counter()-tick, iteration_history=list(history)))
        persist()
        if not accepted:
            raise RuntimeError('Physical acceptance criteria failed in strict probe')
        problem._update_fields()
        if fields is not None:
            fields[offset] = u
            fields.flush()
        if checkpoint_path is not None:
            temporary = checkpoint_path.with_suffix('.pending.npz')
            np.savez_compressed(temporary, completed_probe_steps=offset+1,
                protocol_sha256=result['protocol_sha256'],
                original_checkpoint_sha256=result['checkpoint_sha256'],
                **{name:getattr(problem,name).vector().get_local()
                   for name in ['u','u_old','v_old','a_old']})
            temporary.replace(checkpoint_path)
            result['probe_checkpoint_sha256'] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            persist()
    result['complete'] = True
    result['all_requested_steps_accepted'] = True
except Exception as error:
    result['failure'] = str(error)
    result['all_requested_steps_accepted'] = False
finally:
    result['finished'] = True
    result['last_iteration_history'] = list(history)
    assert hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest() == result['checkpoint_sha256']
    result['wall_seconds'] = time.perf_counter()-started
    persist()
print('STRICT PROBE RESULT', result['all_requested_steps_accepted'], flush=True)
