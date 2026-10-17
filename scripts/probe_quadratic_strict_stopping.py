"""Bounded continuation from rejected-reference checkpoint; no data replacement."""
import argparse
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
root = P/'data/processed'/protocol['reference']
assert (root/'failure.json').exists()
m = json.loads((root/'manifest.json').read_text()); cfg = m['config']
saved = dict(np.load(root/'restart.npz')); inputs = np.load(root/'inputs.npz')
start_step = int(saved['completed_steps']); assert start_step == m['completed_steps']
result = dict(complete=False, engineering_only=True, protocol=protocol, records=[],
    protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    reference_config_sha256=m['config_sha256'], start_step=start_step,
    checkpoint_sha256=hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
output = P/protocol.get('output', 'experiments/metrics/dynamic_quadratic_strict_stopping_probe.json')
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
problem = make_quadratic_problem(geo, material, parameters, 'mumps')
settings = problem.solver._solver.parameters
settings['absolute_tolerance'] = protocol['absolute_tolerance_N']
settings['relative_tolerance'] = protocol['relative_tolerance']
settings['maximum_iterations'] = protocol['maximum_newton_iterations']
settings['line_search'] = 'bt'
np.testing.assert_allclose(problem.u_space.tabulate_dof_coordinates(), inputs['dof_coordinates'], rtol=0, atol=1e-14)
for name in ['u', 'u_old', 'v_old', 'a_old']:
    field = getattr(problem, name); field.vector().set_local(saved[name]); field.vector().apply('insert')
kin = QuadraticKinematics(problem.u_space, inputs['qpoints'], inputs['qweights'])
started = time.perf_counter()
history = []
def monitor(snes, iteration, norm):
    history.append(dict(iteration=int(iteration), residual_N=float(norm) if np.isfinite(norm) else None))
    print('STRICT PROBE', iteration, norm, flush=True)
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
        problem.u.vector().set_local(best); problem.u.vector().apply('insert')
        history.clear(); tick = time.perf_counter()
        problem.solver.solve()
        snes = problem.solver._snes
        R = float(residual(problem).norm('l2')); u = problem.u.vector().get_local()
        endpoint, _ = kin.audit(u); stress, _ = kin.audit(.4*old+.6*u)
        accepted = bool(snes.getConvergedReason()>0 and np.isfinite(R) and R<1e-6
                        and np.isfinite(u).all() and min(endpoint, stress)>0)
        result['records'].append(dict(step=i, time_s=float(time_s), accepted=accepted,
            SNES_reason=int(snes.getConvergedReason()), residual_N=R if np.isfinite(R) else None,
            KSP_reason=int(snes.getKSP().getConvergedReason()),
            minimum_endpoint_sampled_detF=endpoint, minimum_stress_sampled_detF=stress,
            nonlinear_seconds=time.perf_counter()-tick, iteration_history=list(history)))
        persist()
        if not accepted:
            raise RuntimeError('Physical acceptance criteria failed in strict probe')
        problem._update_fields()
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
