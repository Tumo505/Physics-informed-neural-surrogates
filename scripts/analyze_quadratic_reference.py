"""Exact cross-discretization L2 norms, independently checked by native UFL."""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import numpy as np

P = Path(__file__).resolve().parents[1]
ENTRY_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ap = argparse.ArgumentParser()
ap.add_argument('--wait', action='store_true')
ap.add_argument('--native-verify', action='store_true')
ap.add_argument('--pilot', action='store_true')
ap.add_argument('--config', default='configs/physics/dynamic_quadratic_reference.json')
args = ap.parse_args()
reference_config = json.loads((P/args.config).read_text())
root = P / 'data/processed' / reference_config['name']
metric = P / ('experiments/metrics/dynamic_quadratic_comparison_pilot.json'
              if args.pilot else 'experiments/metrics/dynamic_quadratic_comparison.json')

if args.native_verify:
    sys.path.insert(0, str(P / 'src'))
    from lv_surrogate.physics.reference_fem import setup, LVProblem, assign_nodes, df
    from lv_surrogate.physics.quadratic import QuadraticKinematics
    result = json.loads(metric.read_text())
    _, material, coarse = setup(dt=.0025, geometry_scales=[1, 1])
    parameters = dict(coarse.parameters); parameters['function_space'] = 'P_2'
    quadratic = LVProblem(coarse.geometry, material, parameters=parameters,
                          solver_parameters={'linear_solver': 'mumps',
                                             'petsc': {'ksp_type': 'preonly', 'pc_type': 'none'}})
    fine_geo, _, fine = setup(dt=.0025, geometry_scales=[1, 1], refinement_level=1)
    common = df.VectorFunctionSpace(fine_geo.mesh, 'CG', 2)
    saved_inputs = np.load(root/'inputs.npz')
    np.testing.assert_array_equal(coarse.geometry.mesh.cells(), saved_inputs['cells'])
    np.testing.assert_allclose(quadratic.u_space.tabulate_dof_coordinates(),
                               saved_inputs['dof_coordinates'], rtol=0, atol=1e-14)
    kinematics = QuadraticKinematics(quadratic.u_space, saved_inputs['qpoints'],
                                     saved_inputs['qweights'])
    saved_fine = np.load(P/'data/processed/cardiac_dynamic_v1/convergence_refined_dt00025/geometry.npz')
    np.testing.assert_array_equal(fine_geo.mesh.cells(), saved_fine['cells'])
    np.testing.assert_allclose(fine_geo.mesh.coordinates(), saved_fine['coords'],
                               rtol=0, atol=1e-14)
    coefficients = np.load(root / 'displacement_coefficients.npy', mmap_mode='r')
    common_times = np.load(P/'data/processed/cardiac_dynamic_v1/convergence_dt0.0025/trajectory.npz')['time']
    common_indices = np.searchsorted(saved_inputs['time'], common_times)
    np.testing.assert_allclose(saved_inputs['time'][common_indices], common_times,
                               rtol=0, atol=1e-14)
    fine_fields = np.load(P / 'data/processed/cardiac_dynamic_v1/convergence_refined_dt00025/trajectory.npz')['u']
    coarse_fields = np.load(P / 'data/processed/cardiac_dynamic_v1/convergence_dt0.0025/trajectory.npz')['u']
    checks = []
    determinant_checks = []
    for i in ([0, 9] if args.pilot else [0, 99, 199, 399]):
        quadratic.u.vector().set_local(coefficients[common_indices[i]]); quadratic.u.vector().apply('insert')
        minimum, determinant_integral = kinematics.audit(coefficients[common_indices[i]])
        native_determinant = float(df.assemble(df.det(df.Identity(3)+df.grad(quadratic.u))*df.dx))
        np.testing.assert_allclose(determinant_integral, native_determinant,
                                   rtol=1e-10, atol=1e-13)
        determinant_checks.append(dict(step=i, sampled_minimum_detF=minimum,
            relative_integral_discrepancy=abs(determinant_integral-native_determinant)
                                          /max(abs(native_determinant), 1e-30)))
        p2fine = df.interpolate(quadratic.u, common)
        assign_nodes(fine, fine_fields[i])
        p1fine = df.interpolate(fine.u, common)
        difference = df.Function(common)
        difference.vector()[:] = p2fine.vector() - p1fine.vector()
        d2 = float(df.assemble(df.inner(difference, difference) * df.dx(
            domain=fine_geo.mesh, metadata={'quadrature_degree': 4})))
        assign_nodes(coarse, coarse_fields[i])
        p1coarse = df.interpolate(coarse.u, quadratic.u_space)
        dc = df.Function(quadratic.u_space)
        dc.vector()[:] = quadratic.u.vector() - p1coarse.vector()
        dc2 = float(df.assemble(df.inner(dc, dc) * df.dx(
            domain=coarse.geometry.mesh, metadata={'quadrature_degree': 4})))
        for key, native in [('refined_P1', d2), ('coarse_P1', dc2)]:
            algebraic = result['comparisons'][key]['frames'][i]['difference_volume_L2_squared']
            np.testing.assert_allclose(algebraic, native, rtol=1e-7, atol=1e-24)
            checks.append(dict(step=i, comparison=key, algebraic=algebraic, native=native,
                               relative_discrepancy=abs(algebraic-native)/max(abs(native), 1e-30)))
        print('NATIVE P2 NORM VERIFIED', i, flush=True)
    result['native_norm_verification'] = dict(passed=True, records=checks,
        source_sha256=ENTRY_SOURCE_SHA256,
        scope='Independent native interpolation to common P2 space, then exact degree-4 UFL integration.')
    result['native_kinematics_verification'] = dict(passed=True, records=determinant_checks,
        scope='Actual nonlinear P2-field determinant integration versus native UFL at fixed frames.')
    result['complete'] = True
    metric.write_text(json.dumps(result, indent=2, allow_nan=False))
    sys.exit(0)

while True:
    try:
        manifest = json.loads((root / 'manifest.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        manifest = {}
    needed_steps = int(round(.025/reference_config['dt_s']))
    if (manifest.get('complete') and manifest.get('output_sha256')) or (args.pilot and manifest.get('completed_steps', 0) >= needed_steps):
        break
    if (root/'failure.json').exists():
        raise RuntimeError('Reference attempt rejected; inspect its failure.json; no full-cycle analysis executed')
    if not args.wait:
        raise RuntimeError('Full quadratic reference is not complete; no full-cycle result claimed')
    print('WAIT QUADRATIC REFERENCE', manifest.get('completed_steps', 0),
          '/', int(round(reference_config['duration_s']/reference_config['dt_s'])), flush=True)
    time.sleep(30)

if manifest.get('complete'):
    for name, digest in manifest['output_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    (P/'experiments/manifests'/f'{reference_config["name"]}.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False))
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
inputs = dict(np.load(root / 'inputs.npz'))
fine_root = P / 'data/processed/cardiac_dynamic_v1/convergence_refined_dt00025'
coarse_root = P / 'data/processed/cardiac_dynamic_v1/convergence_dt0.0025'
fine_geo = dict(np.load(fine_root / 'geometry.npz'))
fine = dict(np.load(fine_root / 'trajectory.npz'))
coarse = dict(np.load(coarse_root / 'trajectory.npz'))
common_indices = np.searchsorted(inputs['time'], fine['time'])
np.testing.assert_allclose(inputs['time'][common_indices], fine['time'], rtol=0, atol=1e-14)
np.testing.assert_array_equal(coarse['time'], fine['time'])
X = fine_geo['coords']; cells = fine_geo['cells']; nnode = len(X)
maps = inputs['canonical_cell_dofs']
distance, canonical_nodes = cKDTree(X).query(inputs['dof_coordinates'][maps[..., 0]])
assert distance.max() < 1e-12
lookup = np.full((nnode, 3), -1, dtype=int)
lookup[canonical_nodes] = maps
assert lookup.min() >= 0
np.testing.assert_array_equal(lookup[canonical_nodes], maps)

memberships = [set() for _ in range(nnode)]
for parent, indices in enumerate(canonical_nodes):
    for index in indices:
        memberships[index].add(parent)
parents = []
for cell in cells:
    candidates = set.intersection(*(memberships[index] for index in cell))
    assert len(candidates) == 1
    parents.append(candidates.pop())
parents = np.asarray(parents)
del memberships
coarse_points = inputs['coords'][inputs['cells']]
coarse_D = (coarse_points[:, 1:] - coarse_points[:, :1]).transpose(0, 2, 1)
inverse = np.linalg.inv(coarse_D)
fine_points = X[cells]
fine_D = (fine_points[:, 1:] - fine_points[:, :1]).transpose(0, 2, 1)
fine_det = abs(np.linalg.det(fine_D))
coarse_det = abs(np.linalg.det(coarse_D))
edges = list(itertools.combinations(range(4), 2))
q_bary = np.c_[1-inputs['qpoints'].sum(1), inputs['qpoints']]

def basis(bary):
    return np.concatenate((bary*(2*bary-1), np.stack(
        [4*bary[..., i]*bary[..., j] for i, j in edges], axis=-1)), axis=-1)

def sparse(blocks, row_nodes, col_nodes):
    rows = np.broadcast_to(row_nodes[:, :, None], blocks.shape)
    cols = np.broadcast_to(col_nodes[:, None, :], blocks.shape)
    return coo_matrix((blocks.ravel(), (rows.ravel(), cols.ravel())),
                      shape=(nnode, nnode)).tocsr()

reference_basis = basis(q_bary)
reference_mass = np.einsum('q,qi,qj->ij', inputs['qweights'], reference_basis, reference_basis)
M2 = sparse(coarse_det[:, None, None]*reference_mass, canonical_nodes, canonical_nodes)
M1 = sparse(fine_det[:, None, None]/120*(np.eye(4)+np.ones((4, 4))), cells, cells)
cross_blocks = np.zeros((len(cells), 10, 4))
for lam, weight in zip(q_bary, inputs['qweights']):
    points = np.einsum('k,ekj->ej', lam, fine_points)
    xyz = np.einsum('eij,ej->ei', inverse[parents], points-coarse_points[parents, 0])
    parent_bary = np.c_[1-xyz.sum(1), xyz]
    assert parent_bary.min() > -1e-10
    cross_blocks += fine_det[:, None, None]*weight*basis(parent_bary)[:, :, None]*lam[None, None, :]
B = sparse(cross_blocks, canonical_nodes[parents], cells)
del cross_blocks
volume = fine_det.sum()/6
ones = np.ones(nnode)
for matrix in [M1, M2, B]:
    np.testing.assert_allclose(ones @ matrix @ ones, volume, rtol=1e-10)
affine = X @ np.array([[.01, -.03, 0], [.02, 0, .02], [0, .01, -.01]])
affine_terms = [float(np.sum(affine*(matrix@affine))) for matrix in [M1, M2, B]]
assert abs(affine_terms[0]+affine_terms[1]-2*affine_terms[2]) < 1e-12*sum(abs(x) for x in affine_terms)
coeff = np.load(root / 'displacement_coefficients.npy', mmap_mode='r')
comparisons = {key: [] for key in ['coarse_P1', 'refined_P1']}
n_endpoints = 10 if args.pilot else len(common_indices)
for i in range(n_endpoints):
    p2 = coeff[common_indices[i]][lookup]
    p2norm = float(np.sum(p2*(M2@p2)))
    base = coarse['u'][i][inputs['cells']]
    lifted_values = np.concatenate((base, np.stack([(base[:, a]+base[:, b])/2
                                                    for a, b in edges], axis=1)), axis=1)
    lifted = np.empty_like(p2); lifted[canonical_nodes] = lifted_values
    for key, field in [('coarse_P1', lifted), ('refined_P1', fine['u'][i])]:
        p1norm = float(np.sum(field*(M1@field)))
        cross = float(np.sum(p2*(B@field)))
        raw = p2norm+p1norm-2*cross
        assert raw >= -1e-12*max(p2norm+p1norm, 1e-30)
        difference = max(raw, 0.)
        comparisons[key].append(dict(time_s=float(fine['time'][i]),
            difference_volume_L2_squared=difference, P2_volume_L2_squared=p2norm,
            P1_volume_L2_squared=p1norm, relative_difference_vs_P2=np.sqrt(difference/max(p2norm, 1e-30))))
result = dict(complete=False, source_sha256=ENTRY_SOURCE_SHA256,
              reference_config=reference_config, reference_dataset=str(root.relative_to(P)),
              reference_dt_s=reference_config['dt_s'], linear_reference_dt_s=.0025,
              reference_output_sha256=manifest.get('output_sha256'),
              reference_summary=dict(completed_steps=manifest['completed_steps'],
                dofs=manifest['dofs'], tetrahedra=manifest['tetrahedra'],
                minimum_endpoint_sampled_detF=min(r['minimum_endpoint_sampled_detF'] for r in manifest['steps']),
                minimum_stress_sampled_detF=min(r['minimum_stress_sampled_detF'] for r in manifest['steps']),
                maximum_residual_l2_N=max(r['residual_l2_N'] for r in manifest['steps']),
                native_integral_checked_steps=manifest.get('native_integral_checked_steps', [0]),
                positivity_scope=manifest['positivity_scope']),
              full_cycle=not args.pilot, n_endpoints=n_endpoints, dt_s=.0025, domain_volume_m3=float(volume),
              comparisons={}, scope=('Ten-step startup pilot only. ' if args.pilot else 'Full cycle. ')
                + 'Discretization sensitivity on one passive '
                'geometry/loading: P2 on 17625 tetrahedra versus P1 on 17625 and141000 tetrahedra. '
                'Different timesteps change BOTH temporal and spatial discretization. '
                'P2 is a comparison reference, not assumed continuum truth.',
              limitations=['Not a three-level asymptotic convergence study.',
                           'No claim of continuum-calibrated neural intervals.',
                           'Does not modify frozen models or their held-out results.'])
for key, frames in comparisons.items():
    d = sum(f['difference_volume_L2_squared'] for f in frames)
    ref = sum(f['P2_volume_L2_squared'] for f in frames)
    ratio = float(np.sqrt(d/ref))
    result['comparisons'][key] = dict(relative_spacetime_volume_L2_difference_vs_P2=ratio,
        volume_time_component_rmse_m=float(np.sqrt(d/(len(frames)*3*volume))),
        within_1percent_tolerance=ratio <= .01, frames=frames)
metric.write_text(json.dumps(result, indent=2, allow_nan=False))
env = os.environ.copy()
env.update(FI_PROVIDER='tcp', FI_TCP_IFACE='en0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           XDG_CACHE_HOME=str(P/'.cache'), MAMBA_ROOT_PREFIX=str(P/'.mamba'),
           DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'))
subprocess.run([str(P/'.tools/bin/micromamba'), 'run', '-p', str(P/'.fenics'),
                'python', str(Path(__file__).resolve()), '--native-verify']
               + ['--config', args.config]
               + (['--pilot'] if args.pilot else []), cwd=P, env=env, check=True)
print('QUADRATIC STARTUP NORM PILOT COMPLETE' if args.pilot
      else 'QUADRATIC FULL-CYCLE COMPARISON COMPLETE', flush=True)
