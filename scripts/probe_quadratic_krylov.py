"""Bounded engineering comparison of linear solvers on identical native tangents.

No trajectory or model result is replaced. Run inside the native environment.
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

ap = argparse.ArgumentParser()
ap.add_argument('--reference', default='cardiac_quadratic_reference_dt00125_v1')
ap.add_argument('--steps', type=int, nargs='+', default=[0, 223, 399])
ap.add_argument('--allow-incomplete', action='store_true')
ap.add_argument('--output', default='experiments/metrics/dynamic_quadratic_krylov_probe.json')
args = ap.parse_args()
source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
root = P/'data/processed'/args.reference
m = json.loads((root/'manifest.json').read_text())
assert m['complete'] or args.allow_incomplete
assert min(args.steps) >= 0 and max(args.steps) < m['completed_steps']
data = np.load(root/'inputs.npz')
u = np.load(root/'displacement_coefficients.npy', mmap_mode='r')
geo, material, base = setup(dt=m['config']['dt_s'], geometry_scales=m['config']['geometry_scales'],
                            refinement_level=m['config']['refinement_level'])
parameters = dict(base.parameters); parameters['function_space'] = 'P_2'
problem = LVProblem(geo, material, parameters=parameters,
    solver_parameters={'linear_solver': 'mumps', 'petsc': {'ksp_type': 'preonly', 'pc_type': 'none'}})
np.testing.assert_allclose(problem.u_space.tabulate_dof_coordinates(), data['dof_coordinates'], rtol=0, atol=1e-14)
expressions = [('1','0','0'), ('0','1','0'), ('0','0','1'),
               ('0','-x[2]','x[1]'), ('x[2]','0','-x[0]'), ('-x[1]','x[0]','0')]
modes = [df.interpolate(df.Expression(e, degree=1), problem.u_space).vector().copy() for e in expressions]
basis = df.VectorSpaceBasis(modes); basis.orthonormalize()
nullspace = PETSc.NullSpace().create(vectors=[df.as_backend_type(v).vec() for v in modes])
dt = m['config']['dt_s']; beta = float(problem._beta); gamma = float(problem._gamma)
previous = np.zeros(u.shape[1]); velocity = previous.copy(); acceleration = previous.copy()
records = []
steps = args.steps


def assign(field, values):
    field.vector().set_local(values); field.vector().apply('insert')


for i in range(max(steps)+1):
    current = u[i]
    if i in steps:
        for name, values in [('u_old', previous), ('v_old', velocity), ('a_old', acceleration), ('u', current)]:
            assign(getattr(problem, name), values)
        problem.parameters['p'].assign(float(data['pressure'][i]))
        known_residual = float(residual(problem).norm('l2'))
        assert np.isfinite(known_residual) and known_residual < 1e-6
        # Same previous-state Newton initial guess used for both linear methods.
        assign(problem.u, previous)
        native_A = df.assemble(problem._problem._J)
        A = df.as_backend_type(native_A).mat()
        native_rhs = residual(problem)
        rhs = df.as_backend_type(native_rhs).vec().copy(); rhs.scale(-1)
        A.setNearNullSpace(nullspace)
        solutions = {}
        for method in ['mumps', 'gamg']:
            ksp = PETSc.KSP().create(); ksp.setOperators(A)
            ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
            if method == 'mumps':
                ksp.setType('preonly'); ksp.getPC().setType('lu')
                ksp.getPC().setFactorSolverType('mumps')
            else:
                ksp.setType('gmres'); ksp.setGMRESRestart(100)
                ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
                ksp.getPC().setType('gamg')
            interrupted = []
            started = time.perf_counter()
            def monitor(solver, iteration, norm):
                if time.perf_counter()-started > 120:
                    interrupted.append(True)
                    raise RuntimeError('Declared 120s Krylov solve limit exceeded')
            ksp.setMonitor(monitor)
            delta = rhs.duplicate(); delta.set(0)
            error = None
            try:
                ksp.solve(rhs, delta)
            except (RuntimeError, PETSc.Error) as exc:
                error = str(exc)
            elapsed = time.perf_counter()-started
            r = rhs.duplicate(); A.mult(delta, r); r.axpy(-1, rhs)
            reason = int(ksp.getConvergedReason())
            value = delta.getArray(readonly=True).copy()
            record = dict(zero_based_step=i, time_s=float(data['time'][i]),
                method=method, seconds=elapsed, iterations=int(ksp.getIterationNumber()),
                KSP_reason=reason, true_linear_residual_l2_N=float(r.norm()),
                known_nonlinear_reference_residual_l2_N=known_residual, error=error,
                accepted=bool(reason > 0 and error is None and np.isfinite(value).all() and r.norm() < 1e-8))
            if method == 'mumps':
                assert record['accepted'], 'Direct tangent oracle failed'
            else:
                record['relative_correction_difference_vs_MUMPS'] = float(np.linalg.norm(value-solutions['mumps'])/max(np.linalg.norm(solutions['mumps']), 1e-30))
            solutions[method] = value
            records.append(record)
            print(json.dumps(record, allow_nan=False), flush=True)
            ksp.destroy()
        del solutions
    new_acceleration = (current-(previous+dt*velocity+(.5-beta)*dt**2*acceleration))/(beta*dt**2)
    velocity = velocity+(1-gamma)*dt*acceleration+gamma*dt*new_acceleration
    acceleration = new_acceleration; previous = current
result = dict(complete=True, engineering_only=True, records=records,
    reference_name=args.reference, reference_complete=m['complete'], steps=steps,
    executed_source_sha256=source_hash, config_sha256=m['config_sha256'],
    reference_output_sha256=m.get('output_sha256'),
    compared_field_sha256={str(i): hashlib.sha256(np.ascontiguousarray(u[i]).tobytes()).hexdigest() for i in steps},
    scope='Identical native P2 tangent matrices and right-hand sides at the specified known-case Newton initial guesses. GAMG receives six rigid-body near-nullspace vectors. No equations, states, trajectories or model selection change. Linear correction agreement does not establish nonlinear full-cycle equivalence or end-to-end speedup. Startup results on an incomplete reference are engineering evidence only. 120s per-solve monitor guard is checked at Krylov iteration boundaries, not inside preconditioner setup.')
(P/args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
