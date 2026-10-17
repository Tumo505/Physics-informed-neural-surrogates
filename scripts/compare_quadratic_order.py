"""Full-cycle P1/P2 comparison at identical 1ms timestep and geometry."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'src'))
from lv_surrogate.physics.reference_fem import setup, df, assign_nodes, assign_history, residual

root = P/'data/processed/cardiac_quadratic_reference_dt001_v1'
m = json.loads((root/'manifest.json').read_text())
assert m['complete'] and m['n_steps'] == 1000 and m['config']['dt_s'] == .001
for name, digest in m['output_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
inputs = np.load(root/'inputs.npz')
coefficients = np.load(root/'displacement_coefficients.npy', mmap_mode='r')
geo, material, p1 = setup(dt=.001, geometry_scales=[1, 1])
V2 = df.VectorFunctionSpace(geo.mesh, 'CG', 2)
np.testing.assert_array_equal(geo.mesh.cells(), inputs['cells'])
np.testing.assert_allclose(geo.mesh.coordinates(), inputs['coords'], rtol=0, atol=1e-14)
np.testing.assert_allclose(V2.tabulate_dof_coordinates(), inputs['dof_coordinates'], rtol=0, atol=1e-14)
dx = df.Measure('dx', domain=geo.mesh, metadata={'quadrature_degree': 4})
M = df.assemble(df.inner(df.TrialFunction(V2), df.TestFunction(V2))*dx)
volume = float(df.assemble(df.Constant(1)*dx))
T = df.PETScDMCollection.create_transfer_matrix(p1.u_space, V2)
mapped, quadratic, diff = df.Function(V2), df.Function(V2), df.Function(V2)
work = diff.vector().copy()
endpoint_path = P/'data/processed/cardiac_dynamic_endpoint_1s/endpoint.npz'
endpoint_metric = json.loads((P/'experiments/metrics/dynamic_time_endpoint_1s.json').read_text())
assert hashlib.sha256(endpoint_path.read_bytes()).hexdigest() == endpoint_metric['endpoint_sha256']
records = []; native_checks = []
for i in range(1000):
    path = P/f'data/processed/cardiac_passive_trajectory_v1/step_{i:04d}.npz' if i < 999 else endpoint_path
    snapshot = np.load(path)
    np.testing.assert_allclose(float(snapshot['time']), inputs['time'][i], rtol=0, atol=1e-14)
    np.testing.assert_allclose(float(snapshot['dt']), .001, rtol=0, atol=1e-15)
    np.testing.assert_allclose(float(snapshot['pressure']), inputs['pressure'][i], rtol=0, atol=1e-8)
    assign_nodes(p1, snapshot['u'])
    T.mult(p1.u.vector(), mapped.vector())
    quadratic.vector().set_local(coefficients[i]); quadratic.vector().apply('insert')
    diff.vector().zero(); diff.vector().axpy(1, quadratic.vector()); diff.vector().axpy(-1, mapped.vector())
    M.mult(diff.vector(), work); error2 = float(diff.vector().inner(work))
    M.mult(quadratic.vector(), work); reference2 = float(quadratic.vector().inner(work))
    assert np.isfinite([error2, reference2]).all() and min(error2, reference2) >= 0
    if i in [0, 249, 499, 999]:
        # Confirm the archived P1 histories solve the current same-law weak form.
        assign_history(p1, snapshot['u_old'], snapshot['v_old'], snapshot['a_old'])
        p1.parameters['p'].assign(float(snapshot['pressure']))
        force = float(residual(p1).norm('l2'))
        assert np.isfinite(force) and force < 1e-6
        independent = df.interpolate(p1.u, V2)
        native = float(df.assemble(df.inner(quadratic-independent, quadratic-independent)*dx))
        np.testing.assert_allclose(error2, native, rtol=1e-7, atol=1e-24)
        native_checks.append(dict(step=i, time_s=float(inputs['time'][i]),
            P1_native_residual_l2_N=force, algebraic_norm_squared=error2,
            independent_native_norm_squared=native,
            relative_norm_discrepancy=abs(error2-native)/max(native, 1e-30)))
    records.append(dict(step=i, time_s=float(inputs['time'][i]),
        difference_squared_integral_m5=error2, P2_squared_integral_m5=reference2,
        P1_snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if (i+1) % 100 == 0:
        print('SAME-TIMESTEP ORDER COMPARISON', i+1, '/ 1000', flush=True)
times = np.r_[0., inputs['time']]
error = float(np.trapezoid(np.r_[0., [r['difference_squared_integral_m5'] for r in records]], times))
reference = float(np.trapezoid(np.r_[0., [r['P2_squared_integral_m5'] for r in records]], times))
relative = float(np.sqrt(error/reference))
result = dict(complete=True, n_steps=1000, dt_s=.001, changed_factor='displacement_degree_P1_to_P2',
    same_mesh_tetrahedra=geo.mesh.num_cells(), physical_volume_m3=volume,
    relative_space_time_volume_L2_difference=relative,
    component_volume_time_RMSE_m=float(np.sqrt(error/(3*volume))),
    within_one_percent_sensitivity=bool(relative <= .01),
    native_verification=native_checks, P2_output_sha256=m['output_sha256'],
    P1_endpoint_metric=endpoint_metric, records=records,
    executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Same polyhedral domain, P1 fibers, passive law, pressure history, timestep and zero initial histories. Only displacement polynomial degree differs; each native nonlinear solver selects its accepted discrete root. Exact consistent-mass spatial integration and common 1ms trapezoid including zero initial fields. Archived P1 native histories and independent UFL norms checked at four frames; all pressure/time values checked. This is a post-hoc numerical sensitivity diagnosis on one known case, not a new held-out model test, locking attribution, asymptotic p-convergence rate or continuum truth.')
(P/'experiments/metrics/dynamic_quadratic_order_control.json').write_text(json.dumps(result, indent=2, allow_nan=False))
(P/'experiments/manifests'/f'{m["config"]["name"]}.json').write_text(json.dumps(m, indent=2, allow_nan=False))
print(json.dumps({k:result[k] for k in ['complete','relative_space_time_volume_L2_difference','component_volume_time_RMSE_m','within_one_percent_sensitivity']}, indent=2), flush=True)
