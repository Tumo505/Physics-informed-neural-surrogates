"""Audit the limiting P2 cell using independent Cartesian polynomial interpolation."""
import argparse
from decimal import Decimal, localcontext
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lv_surrogate.physics.bernstein import BernsteinAudit
from registry import artifact, atomic, sha

ELEMENT = 813578
EDGES = list(combinations(range(4), 2))


def monomials(x):
    x = np.asarray(x)
    return np.column_stack((np.ones(len(x)), x, x[:, 0] ** 2, x[:, 1] ** 2,
                            x[:, 2] ** 2, x[:, 0] * x[:, 1],
                            x[:, 0] * x[:, 2], x[:, 1] * x[:, 2]))


def cartesian_deformation(vertices, nodal_displacement, points):
    """Fit a complete quadratic in scaled Cartesian coordinates and differentiate."""
    center = vertices.mean(axis=0)
    scale = np.max(np.linalg.norm(vertices - center, axis=1))
    nodes = np.vstack((vertices, [(vertices[i] + vertices[j]) / 2 for i, j in EDGES]))
    coefficients = np.linalg.solve(monomials((nodes - center) / scale), nodal_displacement)
    x, y, z = ((np.asarray(points) - center) / scale).T
    zeros, ones = np.zeros(len(x)), np.ones(len(x))
    derivatives = np.stack((
        np.column_stack((zeros, ones, zeros, zeros, 2*x, zeros, zeros, y, z, zeros)),
        np.column_stack((zeros, zeros, ones, zeros, zeros, 2*y, zeros, x, zeros, z)),
        np.column_stack((zeros, zeros, zeros, ones, zeros, zeros, 2*z, zeros, x, y)),
    ), axis=2) / scale
    return np.eye(3) + np.einsum('qkj,ki->qij', derivatives, coefficients)


def decimal_determinant(matrix):
    with localcontext() as context:
        context.prec = 60
        a = [[Decimal.from_float(float(v)) for v in row] for row in matrix]
        return str(a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
                   - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
                   + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))


def inspect_field(audit, deformation, independent=None):
    points = np.vstack((audit.lattice, np.random.default_rng(555).dirichlet(np.ones(4), 128)))
    matrices = np.einsum('qv,vij->qij', points, deformation)
    values = np.linalg.det(matrices)
    coefficients = np.linalg.det(np.einsum('qv,vij->qij', audit.lattice, deformation)) @ audit.transform.T
    reconstruction = audit.basis(points) @ coefficients
    result = dict(sampled_min=float(values.min()),
                  polynomial_identity_max_abs_error=float(np.max(abs(reconstruction - values))))
    if independent is not None:
        other = independent(points @ audit.X[0])
        result['independent_gradient_max_abs_difference'] = float(np.max(abs(other - matrices)))
        result['independent_det_max_abs_difference'] = float(np.max(abs(np.linalg.det(other) - values)))

    def determinant(barycentric):
        return float(np.linalg.det(np.einsum('v,vij->ij', barycentric, deformation)))

    best_point = points[values.argmin()]
    best = determinant(best_point)
    successes = 0
    # Includes vertex and edge starts, so boundary minima can be found.
    starts = np.vstack((audit.lattice, np.full((1, 4), .25)))
    for start in starts:
        trial = minimize(determinant, start, method='SLSQP', bounds=[(0., 1.)]*4,
                         constraints=[dict(type='eq', fun=lambda b: b.sum()-1)],
                         options=dict(maxiter=150, ftol=1e-14))
        successes += int(trial.success)
        if np.isfinite(trial.x).all() and trial.x.min() >= -1e-10 and abs(trial.x.sum()-1) <= 1e-8:
            point = np.maximum(trial.x, 0)
            point /= point.sum()
            value = determinant(point)
            if value < best:
                best_point, best = point, value
    matrix = np.einsum('v,vij->ij', best_point, deformation)
    result.update(optimized_minimum_estimate=best, minimum_barycentric=best_point.tolist(),
                  minimum_cartesian=(best_point @ audit.X[0]).tolist(),
                  successful_optimizer_starts=successes,
                  determinant_60_digit_of_float_matrix=decimal_determinant(matrix),
                  singular_values_at_minimum=np.linalg.svd(matrix, compute_uv=False).tolist(),
                  minimum_search_scope='Multistart numerical estimate on the closed simplex',
                  bernstein=[])
    for depth in (12, 16, 20, 24):
        audit.max_depth = depth
        try:
            proof = audit.resolve(deformation, audit.X[0])
            entry = dict(depth=depth, outcome='certified_positive', **proof)
        except ValueError as exc:
            entry = dict(depth=depth, outcome='nonpositive_witness' if 'nonpositive' in str(exc)
                         else 'unresolved', reason=str(exc))
        result['bernstein'].append(entry)
        if entry['outcome'] != 'unresolved':
            break
    return result


def run(output):
    target = ROOT / 'data/processed/cardiac_hpc_cmbbe-cofactor-ref-03'
    warm = ROOT / 'experiments/hpc/cmbbe-cuda-production-10/initial-newton.npz'
    replay = ROOT / 'experiments/hpc/cmbbe-positivity-555-02/slice-001'
    manifest_path = target / 'manifest.json'
    inputs_path = target / 'inputs.npz'
    restart = target / 'restart.npz'
    manifest_digest, restart_digest = sha(manifest_path), sha(restart)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['completed_steps'] == len(manifest['steps']) == 554
    assert sha(inputs_path) == manifest['source_inputs_sha256']
    fields = np.load(target / 'displacement_coefficients.npy', mmap_mode='r')
    with np.load(inputs_path, allow_pickle=False) as inputs:
        cells = inputs['cells']
        maps = inputs['canonical_cell_dofs'][ELEMENT]
        vertices = inputs['coords'][cells[ELEMENT]]
        dof_coords = inputs['dof_coordinates'][maps]
        expected = np.vstack((vertices, [(vertices[i]+vertices[j])/2 for i, j in EDGES]))
        coordinate_error = float(np.max(abs(dof_coords - expected[:, None, :])))
        np.testing.assert_allclose(dof_coords, np.broadcast_to(expected[:, None, :], dof_coords.shape),
                                   rtol=0, atol=1e-12)
        audit = BernsteinAudit(dict(coords=vertices, cells=np.array([[0, 1, 2, 3]]),
                                     canonical_cell_dofs=maps[None]))
    edge_matrix = (vertices[1:] - vertices[0]).T
    result = dict(element=ELEMENT, accepted_states=554, margin=1e-9,
                  scope='Single-cell audit. Numerical minimization supplies estimates and witnesses. '
                        'Whole-cell lower bounds use the existing Bernstein subdivision.',
                  geometry=dict(vertices=vertices.tolist(), vertex_ids=cells[ELEMENT].tolist(),
                                reference_signed_volume=float(np.linalg.det(edge_matrix)/6),
                                edge_matrix_condition=float(np.linalg.cond(edge_matrix)),
                                canonical_map=maps.tolist(), unique_dofs=int(len(np.unique(maps))),
                                nodal_coordinate_error=coordinate_error),
                  fields={})
    assert len(np.unique(maps)) == 30
    for state in (553, 554):
        journal = json.loads((target / f'step_journal/{state-1:06d}.json').read_text())
        assert journal['record'] == manifest['steps'][state-1]
        import hashlib
        assert hashlib.sha256(np.asarray(fields[state-1]).tobytes()).hexdigest() == journal['coefficient_sha256']
    assert journal['accepted_checkpoint_sha256'] == restart_digest
    with np.load(restart, allow_pickle=False) as saved:
        assert int(saved['completed_steps']) == 554
        np.testing.assert_array_equal(saved['u_old'], fields[553])
    with np.load(warm, allow_pickle=False) as saved:
        assert int(saved['step']) == 554 and saved['checkpoint_sha256'].item() == restart_digest
        warm_values = saved['u'][maps]
    nodal_fields = dict(accepted_553=fields[552][maps], accepted_554=fields[553][maps],
                        original_warm_start=warm_values)
    for label, nodal in nodal_fields.items():
        gradient = np.stack([
            np.concatenate(((4*b[:, None]-1)*audit.gl[0],
                            np.array([4*(b[i]*audit.gl[0,j]+b[j]*audit.gl[0,i])
                                      for i,j in EDGES])), axis=0)
            for b in np.eye(4)])
        deformation = np.eye(3) + np.einsum('ki,vkj->vij', nodal, gradient)
        result['fields'][label] = inspect_field(
            audit, deformation, lambda points, nodal=nodal: cartesian_deformation(vertices, nodal, points))

    replay_report = json.loads((replay / 'positivity-replay.json').read_text())
    assert replay_report['reproduced'] and replay_report['prefix_matches']
    assert replay_report['element'] == ELEMENT and replay_report['iteration'] == 3
    assert replay_report['stage'] == 'endpoint'
    with np.load(replay / 'rejected-cell.npz', allow_pickle=False) as rejected:
        np.testing.assert_array_equal(rejected['vertices'], vertices)
        deformation = rejected['deformation']
    # Refit each affine deformation component in Cartesian coordinates for the captured trial.
    coefficients = np.linalg.solve(np.column_stack((np.ones(4), vertices)), deformation.reshape(4, 9))
    result['fields']['rejected_newton_3_full_step'] = inspect_field(
        audit, deformation,
        lambda points: (np.column_stack((np.ones(len(points)), points)) @ coefficients).reshape(-1,3,3))
    result['fields']['rejected_newton_3_full_step']['independence_scope'] = (
        'Cartesian refit of captured deformation gradients. Rejected nodal displacement was not saved.')
    assert sha(manifest_path) == manifest_digest and sha(restart) == restart_digest
    output.mkdir()
    atomic(output / 'result.json', result)
    for path in (Path(__file__), manifest_path, restart, inputs_path, warm,
                 replay / 'positivity-replay.json', replay / 'rejected-cell.npz'):
        artifact(ROOT, output, path, 'Single-cell audit input')
    artifact(ROOT, output, output / 'result.json', 'Independent interpolation and determinant audit')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='cmbbe-cell-audit-555-01')
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        parser.error('Invalid run ID')
    run(ROOT / 'experiments/hpc' / args.run_id)
