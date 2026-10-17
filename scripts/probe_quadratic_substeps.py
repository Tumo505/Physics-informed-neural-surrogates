"""Two finer timesteps across the failed interval; preserve original checkpoint."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'src'))
from lv_surrogate.physics.reference_fem import (setup, LVProblem, residual,
    pressure_function, default_parameters_benchmark1)
from lv_surrogate.physics.quadratic import QuadraticKinematics

root = P/'data/processed/cardiac_quadratic_reference_v1'
checkpoint_hash = hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest()
saved = dict(np.load(root/'restart.npz'))
inputs = dict(np.load(root/'inputs.npz'))
manifest = json.loads((root/'manifest.json').read_text())
cfg = manifest['config']; step = int(saved['completed_steps'])
dt = cfg['dt_s']/2
start_time = float(inputs['time'][step-1])
times = start_time + np.arange(1, 3)*dt
par = default_parameters_benchmark1()
for key in ['t_sys_pre', 't_dias_pre', 'gamma']:
    par[key] *= cfg['timing_scale']
pressures = cfg['amplitude']*pressure_function((0, cfg['duration_s']), parameters=par,
                                              t_eval=times-.4*dt)
geo, material, base = setup(dt=dt, geometry_scales=cfg['geometry_scales'])
parameters = dict(base.parameters); parameters['function_space'] = 'P_2'
problem = LVProblem(geo, material, parameters=parameters,
                    solver_parameters={'linear_solver': 'mumps',
                                       'petsc': {'ksp_type': 'preonly', 'pc_type': 'none'}})
problem.solver._solver.parameters['line_search'] = 'bt'
problem.solver._solver.parameters['maximum_iterations'] = 30
np.testing.assert_allclose(problem.u_space.tabulate_dof_coordinates(),
                           inputs['dof_coordinates'], rtol=0, atol=1e-14)
for name in ['u', 'u_old', 'v_old', 'a_old']:
    field = getattr(problem, name); field.vector().set_local(saved[name]); field.vector().apply('insert')
kin = QuadraticKinematics(problem.u_space, inputs['qpoints'], inputs['qweights'])
records = []
for j, (t, pressure) in enumerate(zip(times, pressures)):
    old = problem.u_old.vector().get_local().copy()
    guess = old+dt*problem.v_old.vector().get_local()
    problem.u.vector().set_local(guess); problem.u.vector().apply('insert')
    problem.parameters['p'].assign(float(pressure))
    problem.solver._snes.setMonitor(lambda snes, i, r: print('SUBSTEP', j, i, r, flush=True))
    tick = time.perf_counter(); iterations, _ = problem.solver.solve(); seconds = time.perf_counter()-tick
    reason = int(problem.solver._snes.getConvergedReason())
    force = float(residual(problem).norm('l2'))
    current = problem.u.vector().get_local()
    finite = bool(np.isfinite(current).all() and np.isfinite(force))
    endpoint = stress = None
    if finite:
        endpoint, _ = kin.audit(current)
        stress, _ = kin.audit(.4*old+.6*current)
    accepted = bool(reason>0 and finite and force<1e-6 and min(endpoint, stress)>0)
    records.append(dict(time_s=float(t), dt_s=dt, pressure_Pa=float(pressure),
        SNES_reason=reason, iterations=int(iterations), residual_l2_N=force if np.isfinite(force) else None,
        minimum_endpoint_sampled_detF=endpoint, minimum_stress_sampled_detF=stress,
        nonlinear_seconds=seconds, accepted=accepted))
    if not accepted:
        break
    problem._update_fields()
assert hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest() == checkpoint_hash
result = dict(complete=True, full_cycle=False, all_two_substeps_accepted=len(records)==2
              and all(r['accepted'] for r in records), original_zero_based_failed_step=step,
              original_dt_s=cfg['dt_s'], original_checkpoint_sha256=checkpoint_hash, records=records,
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              scope='A local finer-timegrid diagnostic from the existing accepted state. '
                    'Changes the time discretization; does not solve the rejected original '
                    '2.5ms nonlinear problem or establish full-cycle convergence.')
(P/'experiments/metrics/dynamic_quadratic_substep_probe.json').write_text(json.dumps(result,
                                                                                   indent=2, allow_nan=False))
print('SUBSTEP DIAGNOSTIC COMPLETE', result['all_two_substeps_accepted'], flush=True)
