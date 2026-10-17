"""Full-cycle source P2 displacement solve, with resumable coefficient storage."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import sys
import time
import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'src'))
from lv_surrogate.physics.reference_fem import (setup, LVProblem, df, residual,
    pressure_function, default_parameters_benchmark1)
from lv_surrogate.physics.quadratic import QuadraticKinematics
from FIAT.reference_element import UFCTetrahedron
from FIAT.quadrature_schemes import create_quadrature

ap = argparse.ArgumentParser()
ap.add_argument('--max-steps', type=int)
ap.add_argument('--config', default='configs/physics/dynamic_quadratic_reference.json')
args = ap.parse_args()
config_path = P / args.config
cfg = json.loads(config_path.read_text())
assert cfg['displacement_degree'] == 2
config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out = P / 'data/processed' / cfg['name']
out.mkdir(parents=True, exist_ok=True)
lock = (out / '.run.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX)
manifest = out / 'manifest.json'
old = json.loads(manifest.read_text()) if manifest.exists() else {}
if old and old['config_sha256'] != config_hash:
    raise RuntimeError('Reference protocol changed; a separate run is required')
if old.get('complete'):
    for name, digest in old['output_sha256'].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    print('REUSED COMPLETE QUADRATIC REFERENCE', flush=True)
    sys.exit(0)
failure_flag = out/'failure.json'
if failure_flag.exists():
    failure_flag.unlink()  # The public failure record remains preserved in metrics/Git.
if 'mumps_memory_limit_MB' in cfg:
    df.PETScOptions.set('mat_mumps_icntl_23', int(cfg['mumps_memory_limit_MB']))
geo, material, base_problem = setup(dt=cfg['dt_s'], geometry_scales=cfg['geometry_scales'],
                                    refinement_level=cfg['refinement_level'])
parameters = dict(base_problem.parameters)
parameters['function_space'] = 'P_2'
df.parameters['form_compiler']['quadrature_degree'] = cfg['quadrature_degree']
problem = LVProblem(geo, material, parameters=parameters,
                    solver_parameters={'linear_solver': 'mumps',
                                       'petsc': {'ksp_type': 'preonly', 'pc_type': 'none'}})
del base_problem
problem.solver._solver.parameters['line_search'] = 'bt'
dt = cfg['dt_s']
n = int(round(cfg['duration_s'] / dt))
times = np.arange(1, n + 1) * dt
par = default_parameters_benchmark1()
for key in ['t_sys_pre', 't_dias_pre', 'gamma']:
    par[key] *= cfg['timing_scale']
pressure = cfg['amplitude'] * pressure_function((0, cfg['duration_s']), parameters=par,
                                              t_eval=times - .4 * dt)
quadrature = create_quadrature(UFCTetrahedron(), cfg['quadrature_degree'])
kin = QuadraticKinematics(problem.u_space, quadrature.get_points(), quadrature.get_weights())
ndof = problem.u_space.dim()
coeff_path = out / 'displacement_coefficients.npy'
node_path = out / 'displacement_vertices.npy'
mode = 'r+' if old else 'w+'
coefficients = np.lib.format.open_memmap(coeff_path, mode=mode, dtype=np.float64,
                                        shape=(n, ndof))
nodes = np.lib.format.open_memmap(node_path, mode=mode, dtype=np.float64,
                                shape=(n, geo.mesh.num_vertices(), 3))
if old:
    prior_inputs = np.load(out/'inputs.npz')
    np.testing.assert_array_equal(prior_inputs['cells'], geo.mesh.cells())
    np.testing.assert_allclose(prior_inputs['dof_coordinates'],
                               problem.u_space.tabulate_dof_coordinates(),
                               rtol=0, atol=1e-14)
    np.testing.assert_array_equal(prior_inputs['time'], times)
    np.testing.assert_allclose(prior_inputs['pressure'], pressure, rtol=0, atol=1e-10)
np.savez_compressed(out / 'inputs.npz', time=times, pressure=pressure,
                    coords=geo.mesh.coordinates(), cells=geo.mesh.cells(),
                    dof_coordinates=problem.u_space.tabulate_dof_coordinates(),
                    canonical_cell_dofs=kin.maps, qpoints=quadrature.get_points(),
                    qweights=quadrature.get_weights())
records = old.get('steps', [])
result = old
start_step = len(records)
segments = old.get('source_segments', [])
if old and not segments:
    segments.append(dict(start_step=0, generator_sha256=old['source_sha256']))
segments.append(dict(start_step=start_step, generator_sha256=source_hash,
                     kinematics_sha256=hashlib.sha256(
                         (P / 'src/lv_surrogate/physics/quadratic.py').read_bytes()).hexdigest()))
native_checks = old.get('native_integral_checked_steps', [0] if start_step else [])
if start_step:
    saved = np.load(out / 'restart.npz')
    assert int(saved['completed_steps']) == start_step
    for name in ['u', 'u_old', 'v_old', 'a_old']:
        field = getattr(problem, name)
        field.vector().set_local(saved[name]); field.vector().apply('insert')

# Independently verify local P2 ordering and gradients with an affine manufactured
# field and native UFL integration, before any trajectory step is accepted.
saved_u = problem.u.vector().get_local().copy()
affine = df.interpolate(df.Expression(('0.01*x[0]+0.02*x[1]',
                                      '-0.03*x[0]+0.01*x[2]',
                                      '0.02*x[1]-0.01*x[2]'), degree=1), problem.u_space)
G = np.array([[.01, .02, 0], [-.03, 0, .01], [0, .02, -.01]])
expected = np.linalg.det(np.eye(3) + G)
jmin, integral = kin.audit(affine.vector().get_local())
np.testing.assert_allclose([jmin, integral / kin.volume], expected, rtol=1e-11, atol=1e-12)
problem.u.assign(affine)
native_integral = float(df.assemble(df.det(df.Identity(3) + df.grad(problem.u)) * df.dx))
np.testing.assert_allclose(integral, native_integral, rtol=1e-11, atol=1e-14)
problem.u.vector().set_local(saved_u); problem.u.vector().apply('insert')
start = time.perf_counter()
stop = min(n, args.max_steps) if args.max_steps else n
def reject(step, reason, force, kind):
    durable = json.loads(manifest.read_text()).get('completed_steps', 0) if manifest.exists() else 0
    failure = dict(complete_trajectory=False, reference_name=cfg['name'],
        accepted_steps_before_failure=step, persisted_checkpoint_steps=durable,
        failed_zero_based_step=step, failed_time_s=float(times[step]), SNES_reason=reason,
        residual_l2_N=float(force) if force is not None and np.isfinite(force) else None,
        rejection_kind=kind, config=cfg, config_sha256=config_hash,
        source_sha256=source_hash, scope='Rejected reference attempt; no full-cycle result.')
    payload = json.dumps(failure, indent=2, allow_nan=False)
    failure_flag.write_text(payload)
    (P/f'experiments/metrics/{cfg["name"]}_failure.json').write_text(payload)
    raise RuntimeError(f'Invalid quadratic reference at step {step}: SNES={reason}, R={force}, {kind}')
for i in range(start_step, stop):
    uold = problem.u_old.vector().get_local().copy()
    vold = problem.v_old.vector().get_local()
    aold = problem.a_old.vector().get_local()
    problem.parameters['p'].assign(float(pressure[i]))
    best = uold
    best_residual = float('inf')
    for guess in [uold, uold + dt * vold, uold + dt * vold + .5 * dt**2 * aold]:
        problem.u.vector().set_local(guess); problem.u.vector().apply('insert')
        r = float(residual(problem).norm('l2'))
        if np.isfinite(r) and r < best_residual:
            best = guess; best_residual = r
    problem.u.vector().set_local(best); problem.u.vector().apply('insert')
    tick = time.perf_counter()
    try:
        iterations, _ = problem.solver.solve()
    except RuntimeError:
        reject(i, int(problem.solver._snes.getConvergedReason()), None, 'native_solver_exception')
    solve_seconds = time.perf_counter() - tick
    reason = int(problem.solver._snes.getConvergedReason())
    R = float(residual(problem).norm('l2'))
    current = problem.u.vector().get_local().copy()
    if reason <= 0 or not np.isfinite(current).all() or not np.isfinite(R) or R > 1e-6:
        reject(i, reason, R, 'nonconvergence_or_invalid_force_or_field')
    try:
        endpoint_min, integral = kin.audit(current)
        stress_min, _ = kin.audit(.4 * uold + .6 * current)
    except FloatingPointError:
        reject(i, reason, R, 'nonfinite_sampled_Jacobian')
    if min(endpoint_min, stress_min) <= 0:
        reject(i, reason, R, 'nonpositive_sampled_Jacobian')
    if i in [0, n//4-1, n//2-1, n-1]:
        native = float(df.assemble(df.det(df.Identity(3) + df.grad(problem.u)) * df.dx))
        np.testing.assert_allclose(integral, native, rtol=1e-10, atol=1e-13)
        native_checks.append(i)
    coefficients[i] = current
    nodes[i] = problem.u.compute_vertex_values(geo.mesh).reshape(3, -1).T
    problem._update_fields()
    records.append(dict(step=i, time_s=float(times[i]), iterations=int(iterations),
                        SNES_reason=reason, residual_l2_N=R,
                        minimum_endpoint_sampled_detF=endpoint_min,
                        minimum_stress_sampled_detF=stress_min,
                        nonlinear_solve_seconds=solve_seconds))
    if (i + 1) % cfg['checkpoint_interval'] == 0 or i == stop - 1:
        coefficients.flush(); nodes.flush()
        np.savez_compressed(out / 'restart.npz', completed_steps=i + 1,
                            **{name: getattr(problem, name).vector().get_local()
                               for name in ['u', 'u_old', 'v_old', 'a_old']})
        result = dict(complete=False, completed_steps=i + 1, n_steps=n,
                      config=cfg, config_sha256=config_hash, source_sha256=source_hash,
                      source_segments=segments, native_integral_checked_steps=native_checks,
                      dofs=ndof, tetrahedra=geo.mesh.num_cells(), steps=records,
                      measured_wall_seconds=old.get('measured_wall_seconds', 0)
                                            + time.perf_counter() - start,
                      kinematics_verification='Affine-field determinant and volume integral '
                        'verified independently against native UFL; actual-field integral '
                        'checks are recorded in native_integral_checked_steps.',
                      positivity_scope='Source degree-6 quadrature points and all tetrahedral '
                        'vertices, endpoint and generalized-alpha stress states; not a '
                        'mathematical positivity guarantee between sample locations.')
        manifest.write_text(json.dumps(result, indent=2, allow_nan=False))
        print('QUADRATIC REFERENCE', i + 1, '/', n, flush=True)
if stop == n:
    result['output_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [coeff_path, node_path, out / 'inputs.npz']}
    result['complete'] = True
    manifest.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('QUADRATIC FULL CYCLE COMPLETE', flush=True)
