"""Controlled P2 space/time sensitivity with native consistent-mass norms.

Run in the native FEniCS environment. Partial runs are explicitly pilot evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'src'))
from lv_surrogate.physics.reference_fem import setup, df

ap = argparse.ArgumentParser()
ap.add_argument('--config-a', required=True)
ap.add_argument('--config-b', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--pilot', action='store_true')
args = ap.parse_args()


def read(config):
    path = P / config
    cfg = json.loads(path.read_text())
    root = P / 'data/processed' / cfg['name']
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['config_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    if not args.pilot:
        assert manifest['complete'], 'Full-cycle comparison requires both completed references'
        assert not (root / 'failure.json').exists()
        for filename, digest in manifest['output_sha256'].items():
            assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest
        (P/'experiments/manifests'/f'{cfg["name"]}.json').write_text(
            json.dumps(manifest, indent=2, allow_nan=False))
    inputs = np.load(root / 'inputs.npz')
    coefficients = np.load(root / 'displacement_coefficients.npy', mmap_mode='r')
    geo, _, _ = setup(dt=cfg['dt_s'], refinement_level=cfg['refinement_level'],
                      geometry_scales=cfg['geometry_scales'])
    V = df.VectorFunctionSpace(geo.mesh, 'CG', 2)
    np.testing.assert_allclose(V.tabulate_dof_coordinates(), inputs['dof_coordinates'],
                               rtol=0, atol=1e-14)
    np.testing.assert_array_equal(geo.mesh.cells(), inputs['cells'])
    return cfg, manifest, inputs, coefficients, geo, V


ca, ma, ia, ua, ga, Va = read(args.config_a)
cb, mb, ib, ub, gb, Vb = read(args.config_b)
for key in ['duration_s', 'geometry_scales', 'amplitude', 'timing_scale',
            'displacement_degree', 'quadrature_degree']:
    assert ca[key] == cb[key], f'Uncontrolled change: {key}'
mesh_changed = ca['refinement_level'] != cb['refinement_level']
time_changed = ca['dt_s'] != cb['dt_s']
assert mesh_changed != time_changed, 'Change exactly one discretization factor'
assert ca.get('native_solver', 'mumps') == cb.get('native_solver', 'mumps'), 'Match solver backend for a one-factor discretization control'
assert cb['refinement_level'] >= ca['refinement_level']
assert cb['dt_s'] <= ca['dt_s']
ta = ia['time'][:ma['completed_steps']]
tb = ib['time'][:mb['completed_steps']]
indices = []
for i, t in enumerate(ta):
    j = int(np.argmin(abs(tb - t)))
    if abs(tb[j] - t) < 1e-12:
        indices.append((i, j))
assert len(indices) >= 2, 'Need at least two common positive-time endpoints'
full_cycle = bool(ma.get('complete') and mb.get('complete')
                  and abs(ta[indices[-1][0]] - ca['duration_s']) < 1e-12)
assert args.pilot or full_cycle
dx = df.Measure('dx', domain=gb.mesh, metadata={'quadrature_degree': 4})
trial, test = df.TrialFunction(Vb), df.TestFunction(Vb)
M = df.assemble(df.inner(trial, test) * dx)
volume = float(df.assemble(df.Constant(1) * dx))
volume_a = float(df.assemble(df.Constant(1) * df.dx(domain=ga.mesh)))
np.testing.assert_allclose(volume, volume_a, rtol=1e-11, atol=1e-14)
T = df.PETScDMCollection.create_transfer_matrix(Va, Vb) if mesh_changed else None
a, b, mapped, diff = (df.Function(Va), df.Function(Vb), df.Function(Vb), df.Function(Vb))
work = b.vector().copy()


def set_field(field, values):
    field.vector().set_local(values)
    field.vector().apply('insert')


# Verify transfer independently on a manufactured quadratic vector field.
expression = df.Expression(('x[0]*x[0]+x[1]', 'x[1]*x[2]', 'x[2]*x[2]-x[0]'), degree=2)
a.assign(df.interpolate(expression, Va))
if T is not None:
    T.mult(a.vector(), mapped.vector())
else:
    set_field(mapped, a.vector().get_local())
manufactured = df.interpolate(expression, Vb)
transfer_error = float(np.max(abs(mapped.vector().get_local() - manufactured.vector().get_local())))
assert transfer_error < 1e-10
checks, records = [], []
check_positions = set(np.linspace(0, len(indices)-1, min(4, len(indices)), dtype=int))
for k, (i, j) in enumerate(indices):
    set_field(a, ua[i]); set_field(b, ub[j])
    if T is not None:
        T.mult(a.vector(), mapped.vector())
    else:
        set_field(mapped, a.vector().get_local())
    diff.vector().zero(); diff.vector().axpy(1, b.vector()); diff.vector().axpy(-1, mapped.vector())
    M.mult(diff.vector(), work)
    error2 = float(diff.vector().inner(work))
    M.mult(b.vector(), work)
    reference2 = float(b.vector().inner(work))
    assert np.isfinite([error2, reference2]).all() and min(error2, reference2) >= 0
    if k in check_positions:
        # Independent interpolation plus UFL integration, distinct from T and M.
        independent = df.interpolate(a, Vb)
        native = float(df.assemble(df.inner(b-independent, b-independent) * dx))
        np.testing.assert_allclose(error2, native, rtol=1e-7, atol=1e-24)
        checks.append(dict(time_s=float(ta[i]), mass_norm_squared=error2,
                           independent_UFL_norm_squared=native,
                           relative_discrepancy=abs(error2-native)/max(native, 1e-30)))
    records.append(dict(time_s=float(ta[i]), a_step=int(i), b_step=int(j),
                        error_squared_integral_m5=error2,
                        reference_squared_integral_m5=reference2))
    if k % 25 == 0:
        print('P2 COMPARISON', k+1, '/', len(indices), flush=True)
times = np.r_[0., [r['time_s'] for r in records]]
errors = np.r_[0., [r['error_squared_integral_m5'] for r in records]]
references = np.r_[0., [r['reference_squared_integral_m5'] for r in records]]
error_integral = float(np.trapezoid(errors, times))
reference_integral = float(np.trapezoid(references, times))
relative = float(np.sqrt(error_integral/reference_integral))
result = dict(complete=full_cycle and not args.pilot, pilot=args.pilot,
    comparison='mesh' if mesh_changed else 'timestep', config_a=ca, config_b=cb,
    config_paths=[args.config_a, args.config_b],
    config_sha256=[ma['config_sha256'], mb['config_sha256']],
    output_sha256=[ma.get('output_sha256'), mb.get('output_sha256')],
    executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    common_endpoints=len(indices), end_time_s=float(times[-1]), physical_volume_m3=volume,
    relative_space_time_L2=relative,
    component_volume_time_RMSE_m=float(np.sqrt(error_integral/(3*volume*times[-1]))),
    tolerance_relative_L2=.01, passes_declared_one_percent_sensitivity=bool(full_cycle and not args.pilot and relative <= .01),
    integration='Trapezoid in time including exactly zero initial fields; exact degree-4 consistent-mass spatial integration at common endpoints.',
    manufactured_transfer_max_abs_error=transfer_error,
    independent_native_checks=checks, records=records,
    scope='Controlled two-level sensitivity in one factor with frozen mechanics and initial/loading conditions. Finer discretization is not assumed continuum truth. Two levels do not establish an asymptotic convergence rate. Partial pilot checks never pass the full-cycle criterion.')
target = P / args.output
target.write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps({key: result[key] for key in ['complete', 'comparison', 'common_endpoints',
    'relative_space_time_L2', 'component_volume_time_RMSE_m', 'passes_declared_one_percent_sensitivity']}, indent=2), flush=True)
