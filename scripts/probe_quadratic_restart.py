"""Bounded same-state solver retry; no accepted trajectory state is overwritten."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'src'))
from lv_surrogate.physics.reference_fem import setup, LVProblem, residual
from lv_surrogate.physics.quadratic import QuadraticKinematics

ap = argparse.ArgumentParser()
ap.add_argument('--method', choices=['newtontr', 'newtonls'], default='newtontr')
ap.add_argument('--guess', choices=['old', 'velocity', 'acceleration'], default='old')
ap.add_argument('--iterations', type=int, default=30)
args = ap.parse_args()
root = P/'data/processed/cardiac_quadratic_reference_v1'
inputs = dict(np.load(root/'inputs.npz'))
saved = dict(np.load(root/'restart.npz'))
cfg = json.loads((root/'manifest.json').read_text())['config']
step = int(saved['completed_steps'])
geo, material, base = setup(dt=cfg['dt_s'], geometry_scales=cfg['geometry_scales'])
parameters = dict(base.parameters); parameters['function_space'] = 'P_2'
problem = LVProblem(geo, material, parameters=parameters,
                    solver_parameters={'linear_solver': 'mumps',
                                       'petsc': {'ksp_type': 'preonly', 'pc_type': 'none'}})
for name in ['u', 'u_old', 'v_old', 'a_old']:
    f = getattr(problem, name); f.vector().set_local(saved[name]); f.vector().apply('insert')
np.testing.assert_allclose(problem.u_space.tabulate_dof_coordinates(),
                           inputs['dof_coordinates'], rtol=0, atol=1e-14)
problem.parameters['p'].assign(float(inputs['pressure'][step]))
guess = saved['u_old'].copy()
if args.guess in ['velocity', 'acceleration']:
    guess += cfg['dt_s']*saved['v_old']
if args.guess == 'acceleration':
    guess += .5*cfg['dt_s']**2*saved['a_old']
problem.u.vector().set_local(guess); problem.u.vector().apply('insert')
settings = problem.solver._solver.parameters
settings['method'] = args.method
settings['line_search'] = 'bt'
settings['maximum_iterations'] = args.iterations
history = []
def monitor(snes, iteration, norm):
    history.append(dict(iteration=iteration, residual_norm=float(norm)))
    print('RETRY', args.method, args.guess, iteration, norm, flush=True)
problem.solver._snes.setMonitor(monitor)
start = time.perf_counter()
iterations, _ = problem.solver.solve()
seconds = time.perf_counter()-start
reason = int(problem.solver._snes.getConvergedReason())
R = float(residual(problem).norm('l2'))
u = problem.u.vector().get_local()
finite = bool(np.isfinite(u).all() and np.isfinite(R))
endpoint = stress = None
if finite:
    kin = QuadraticKinematics(problem.u_space, inputs['qpoints'], inputs['qweights'])
    endpoint, _ = kin.audit(u)
    stress, _ = kin.audit(.4*saved['u_old']+.6*u)
accepted = bool(reason>0 and finite and R<1e-6 and min(endpoint, stress)>0)
record = dict(engineering_retry=True, complete=True, accepted=accepted, zero_based_step=step,
    time_s=float(inputs['time'][step]), method_requested=args.method,
    method_effective=problem.solver._snes.getType(), guess=args.guess,
    maximum_iterations=args.iterations, SNES_reason=reason, iterations=int(iterations),
    residual_l2_N=R, minimum_endpoint_sampled_detF=endpoint,
    minimum_stress_sampled_detF=stress, nonlinear_seconds=seconds, iteration_history=history,
    checkpoint_sha256=hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Same frozen FE history/pressure, unchanged material, timestep and tolerances. '
          'A successful one-step retry is not full-cycle convergence.')
(P/f'experiments/metrics/dynamic_quadratic_retry_{args.method}_{args.guess}.json').write_text(
    json.dumps(record, indent=2, allow_nan=False))
if accepted:
    np.savez_compressed(P/f'experiments/logs/quadratic_retry_{args.method}_{args.guess}.npz', u=u)
print('QUADRATIC RETRY COMPLETE', accepted, reason, R, flush=True)
