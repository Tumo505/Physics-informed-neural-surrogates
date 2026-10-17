"""History/load homotopy to the EXACT rejected step; intermediate states are not data."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'src'))
from lv_surrogate.physics.reference_fem import setup, LVProblem, residual
from lv_surrogate.physics.quadratic import QuadraticKinematics

root = P/'data/processed/cardiac_quadratic_reference_v1'
saved = dict(np.load(root/'restart.npz'))
inputs = dict(np.load(root/'inputs.npz'))
config = json.loads((root/'manifest.json').read_text())['config']
checkpoint_hash = hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest()
step = int(saved['completed_steps']); assert step >= 2
coefficients = np.load(root/'displacement_coefficients.npy', mmap_mode='r')
current = {name: saved[name].copy() for name in ['u_old', 'v_old', 'a_old']}
geo, material, base = setup(dt=config['dt_s'], geometry_scales=config['geometry_scales'])
parameters = dict(base.parameters); parameters['function_space'] = 'P_2'
problem = LVProblem(geo, material, parameters=parameters,
                    solver_parameters={'linear_solver': 'mumps',
                                       'petsc': {'ksp_type': 'preonly', 'pc_type': 'none'}})
beta = float(problem._beta); gamma = float(problem._gamma); dt = config['dt_s']
u_previous = coefficients[step-2].copy()
a_previous = ((beta-gamma)*current['a_old']
              -(current['u_old']-u_previous-dt*current['v_old'])/dt**2)/(.5-gamma+beta)
v_previous = current['v_old']-dt*((1-gamma)*a_previous+gamma*current['a_old'])
previous = dict(u_old=u_previous, v_old=v_previous, a_old=a_previous)
np.testing.assert_allclose(problem.u_space.tabulate_dof_coordinates(),
                           inputs['dof_coordinates'], rtol=0, atol=1e-14)
problem.u.vector().set_local(current['u_old']); problem.u.vector().apply('insert')
for name, values in previous.items():
    f = getattr(problem, name); f.vector().set_local(values); f.vector().apply('insert')
problem.parameters['p'].assign(float(inputs['pressure'][step-1]))
prior_force = float(residual(problem).norm('l2'))
assert np.isfinite(prior_force) and prior_force < 1e-6, 'Prior equilibrium not reproduced'
settings = problem.solver._solver.parameters
settings['method'] = 'newtonls'; settings['line_search'] = 'bt'; settings['maximum_iterations'] = 10
kin = QuadraticKinematics(problem.u_space, inputs['qpoints'], inputs['qweights'])
records = []
for fraction in np.linspace(.1, 1., 10):
    for name in current:
        field = getattr(problem, name)
        field.vector().set_local((1-fraction)*previous[name]+fraction*current[name])
        field.vector().apply('insert')
    pressure = (1-fraction)*inputs['pressure'][step-1]+fraction*inputs['pressure'][step]
    problem.parameters['p'].assign(float(pressure))
    iterations, _ = problem.solver.solve()
    reason = int(problem.solver._snes.getConvergedReason())
    force = float(residual(problem).norm('l2'))
    u = problem.u.vector().get_local()
    finite = bool(np.isfinite(u).all() and np.isfinite(force))
    endpoint = stress = None
    if finite:
        try:
            endpoint, _ = kin.audit(u)
            stress, _ = kin.audit(.4*problem.u_old.vector().get_local()+.6*u)
        except FloatingPointError:
            finite = False
    accepted = bool(reason>0 and finite and force<1e-6 and min(endpoint, stress)>0)
    records.append(dict(continuation_fraction=float(fraction), SNES_reason=reason,
        iterations=int(iterations), residual_l2_N=force,
        minimum_endpoint_sampled_detF=endpoint, minimum_stress_sampled_detF=stress,
        accepted=accepted))
    print('HOMOTOPY', fraction, reason, force, accepted, flush=True)
    if not accepted:
        break
recovered = len(records)==10 and all(r['accepted'] for r in records)
assert hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest()==checkpoint_hash
result = dict(complete=True, full_cycle=False, original_failed_step_recovered=recovered,
    zero_based_target_step=step, unchanged_dt_s=dt, previous_equilibrium_residual_l2_N=prior_force,
    checkpoint_sha256=checkpoint_hash, records=records,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Solver homotopy from the prior converged algebraic equation to the exact '
          'original failed equation. Material, timestep, final histories/loading/tolerances '
          'unchanged. Intermediate states are not physical timesteps or surrogate labels. '
          'Original checkpoint remains untouched; this is not a full-cycle result.')
def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value
(P/'experiments/metrics/dynamic_quadratic_continuation_probe.json').write_text(
    json.dumps(clean(result), indent=2, allow_nan=False))
if recovered:
    np.savez_compressed(P/'experiments/logs/quadratic_continuation_recovered.npz', u=problem.u.vector().get_local())
print('EXACT-STEP CONTINUATION COMPLETE', recovered, flush=True)
