"""Check native nonlinear GMRES/GAMG against accepted same-state MUMPS roots.

Engineering probes only: no reference trajectory, checkpoint or model is replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from petsc4py import PETSc

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'src'))
from lv_surrogate.physics.reference_fem import setup, LVProblem, residual, df
from lv_surrogate.physics.quadratic import QuadraticKinematics

ap = argparse.ArgumentParser()
ap.add_argument('--reference', default='cardiac_quadratic_reference_dt00125_v1')
ap.add_argument('--steps', nargs='+', type=int, default=[0, 223, 399])
ap.add_argument('--allow-incomplete', action='store_true')
ap.add_argument('--local-probe', action='store_true', help='Use saved accepted half-step diagnostic fields and its nonzero initial history')
ap.add_argument('--output', default='experiments/metrics/dynamic_quadratic_nonlinear_krylov.json')
args = ap.parse_args()
source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
root = P/'data/processed'/args.reference
m = json.loads((root/'manifest.json').read_text())
local_saved = None
if args.local_probe:
    protocol = json.loads((P/'configs/physics/dynamic_quadratic_refined_substep_extended_probe.json').read_text())
    local = json.loads((P/protocol['output']).read_text())
    assert protocol['reference'] == args.reference and local['all_requested_steps_accepted']
    assert hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest() == local['checkpoint_sha256']
    local_saved = dict(np.load(root/'restart.npz'))
    m = dict(m, complete=False, completed_steps=len(local['records']),
             config=dict(m['config'], dt_s=local['dt_s']),
             steps=[dict(r, nonlinear_solve_seconds=r['nonlinear_seconds']) for r in local['records']])
assert m['complete'] or args.allow_incomplete
assert min(args.steps) >= 0 and max(args.steps) < m['completed_steps']
cfg = m['config']; data = np.load(root/'inputs.npz')
u = np.load(root/'displacement_coefficients.npy', mmap_mode='r')
if args.local_probe:
    from lv_surrogate.physics.reference_fem import pressure_function, default_parameters_benchmark1
    u = np.load(P/protocol['fields'], mmap_mode='r')
    data = dict(data)
    data['time'] = np.array([r['time_s'] for r in local['records']])
    par = default_parameters_benchmark1()
    for key in ['t_sys_pre', 't_dias_pre', 'gamma']:
        par[key] *= m['config']['timing_scale']
    data['pressure'] = m['config']['amplitude']*pressure_function((0,m['config']['duration_s']),parameters=par,t_eval=data['time']-.4*m['config']['dt_s'])
geo, material, base = setup(dt=cfg['dt_s'], geometry_scales=cfg['geometry_scales'],
                            refinement_level=cfg['refinement_level'])
parameters = dict(base.parameters); parameters['function_space'] = 'P_2'
problem = LVProblem(geo, material, parameters=parameters, solver_parameters={
    'linear_solver': 'gmres', 'preconditioner': 'petsc_amg',
    'petsc': {'ksp_type': 'gmres', 'pc_type': 'gamg', 'ksp_gmres_restart': 100,
              'ksp_norm_type': 'unpreconditioned'},
    'krylov_solver': {'absolute_tolerance': 1e-12, 'relative_tolerance': 1e-10,
        'maximum_iterations': 1000, 'monitor_convergence': False}})
np.testing.assert_allclose(problem.u_space.tabulate_dof_coordinates(),
                           data['dof_coordinates'], rtol=0, atol=1e-14)
expressions = [('1','0','0'), ('0','1','0'), ('0','0','1'),
               ('0','-x[2]','x[1]'), ('x[2]','0','-x[0]'), ('-x[1]','x[0]','0')]
modes = [df.interpolate(df.Expression(e, degree=1), problem.u_space).vector().copy()
         for e in expressions]
basis = df.VectorSpaceBasis(modes); basis.orthonormalize()
nullspace = PETSc.NullSpace().create(vectors=[df.as_backend_type(v).vec() for v in modes])
original_J = problem._problem.J
jacobian_calls = []


def with_nullspace(A, x):
    original_J(A, x)
    df.as_backend_type(A).mat().setNearNullSpace(nullspace)
    jacobian_calls.append(True)


problem._problem.J = with_nullspace
problem.solver._solver.parameters['line_search'] = 'bt'
problem.solver._solver.parameters['maximum_iterations'] = 20
if args.local_probe:
    problem.solver._solver.parameters['relative_tolerance'] = 0.
    problem.solver._solver.parameters['absolute_tolerance'] = 1e-8
kin = QuadraticKinematics(problem.u_space, data['qpoints'], data['qweights'])
dt = cfg['dt_s']; beta = float(problem._beta); gamma = float(problem._gamma)
previous = np.zeros(u.shape[1]); velocity = previous.copy(); acceleration = previous.copy()
if local_saved is not None:
    previous, velocity, acceleration = (local_saved[k].copy() for k in ['u_old','v_old','a_old'])
records = []
target = P/args.output


def assign(field, values):
    field.vector().set_local(values); field.vector().apply('insert')


def persist(complete):
    result = dict(complete=complete, engineering_only=True, reference_name=args.reference,
        local_changed_grid_probe=args.local_probe,
        reference_complete=m['complete'], config_sha256=m['config_sha256'],
        reference_output_sha256=m.get('output_sha256'), requested_steps=args.steps,
        executed_source_sha256=source_hash, records=records,
        scope='Native nonlinear solves at the same frozen histories, pressure, passive law and P2 discretization as accepted MUMPS states. Reconstructed histories must first reproduce the recorded solution residual. The three original production predictor candidates are screened by native residual. Only linear solver/preconditioner differ. A 20-Newton-iteration/300s monitor budget bounds engineering effort; the time guard acts at nonlinear iteration boundaries. Acceptance additionally requires same-root relative coefficient agreement below1e-6 and positive sampled endpoint/stress Jacobians. These isolated probes do not establish full-cycle equivalence or isolated runtime speedup, and never overwrite reference data.')
    target.write_text(json.dumps(result, indent=2, allow_nan=False))


persist(False)
for i in range(max(args.steps)+1):
    current = u[i]
    if i in args.steps:
        for name, value in [('u_old', previous), ('v_old', velocity),
                            ('a_old', acceleration), ('u', current)]:
            assign(getattr(problem, name), value)
        problem.parameters['p'].assign(float(data['pressure'][i]))
        known_R = float(residual(problem).norm('l2'))
        assert np.isfinite(known_R) and known_R < 1e-6, 'History reconstruction failed'
        best, best_R = previous, float('inf')
        for guess in [previous, previous+dt*velocity,
                      previous+dt*velocity+.5*dt**2*acceleration]:
            assign(problem.u, guess)
            R = float(residual(problem).norm('l2'))
            if np.isfinite(R) and R < best_R:
                best, best_R = guess, R
        assign(problem.u, best)
        history = []; jacobian_calls.clear()
        started = time.perf_counter()
        def monitor(snes, iteration, norm):
            history.append(dict(iteration=int(iteration), residual_l2_N=float(norm)))
            print('NONLINEAR KRYLOV', i, iteration, norm, flush=True)
            if time.perf_counter()-started > 300:
                raise RuntimeError('Declared engineering nonlinear wall limit exceeded')
        problem.solver._snes.cancelMonitor()
        problem.solver._snes.setMonitor(monitor)
        error = None
        try:
            problem.solver.solve()
        except (RuntimeError, PETSc.Error) as exc:
            error = str(exc)
        elapsed = time.perf_counter()-started
        snes = problem.solver._snes; ksp = snes.getKSP()
        R = float(residual(problem).norm('l2'))
        prediction = problem.u.vector().get_local()
        finite = bool(np.isfinite(R) and np.isfinite(prediction).all())
        endpoint = stress = relative = None
        if finite:
            endpoint, _ = kin.audit(prediction)
            stress, _ = kin.audit(.4*previous+.6*prediction)
            relative = float(np.linalg.norm(prediction-current)/max(np.linalg.norm(current), 1e-30))
        accepted = bool(error is None and finite and snes.getConvergedReason() > 0
            and R < 1e-6 and min(endpoint, stress) > 0 and relative < 1e-6
            and ksp.getType() == 'gmres' and ksp.getPC().getType() == 'gamg' and jacobian_calls)
        record = dict(zero_based_step=i, time_s=float(data['time'][i]), accepted=accepted,
            seconds=elapsed, SNES_reason=int(snes.getConvergedReason()),
            nonlinear_iterations=int(snes.getIterationNumber()),
            total_linear_iterations=int(snes.getLinearSolveIterations()),
            effective_KSP=ksp.getType(), effective_PC=ksp.getPC().getType(),
            Jacobian_near_nullspace_attachments=len(jacobian_calls),
            known_reference_residual_l2_N=known_R, residual_l2_N=R if finite else None,
            relative_coefficient_difference_vs_MUMPS=relative,
            minimum_endpoint_sampled_detF=endpoint, minimum_stress_sampled_detF=stress,
            error=error, iteration_history=history,
            accepted_reference_nonlinear_seconds=m['steps'][i]['nonlinear_solve_seconds'],
            reference_field_sha256=hashlib.sha256(np.ascontiguousarray(current).tobytes()).hexdigest())
        records.append(record); persist(False)
        print('NONLINEAR KRYLOV RESULT', json.dumps(record, allow_nan=False), flush=True)
    new_a = (current-(previous+dt*velocity+(.5-beta)*dt**2*acceleration))/(beta*dt**2)
    velocity = velocity+(1-gamma)*dt*acceleration+gamma*dt*new_a
    acceleration = new_a; previous = current
persist(True)
