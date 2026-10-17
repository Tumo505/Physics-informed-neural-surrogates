"""Sufficient whole-element P2 Jacobian positivity via cubic Bernstein bounds.

This is a floating-point algebraic bound with a declared margin, not an interval-
arithmetic proof. Nonpositive Bernstein coefficients mean unresolved, not inverted.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--config', default='configs/physics/dynamic_quadratic_reference_dt00125.json')
ap.add_argument('--max-steps', type=int)
args = ap.parse_args()
config_path = P / args.config
cfg = json.loads(config_path.read_text())
root = P / 'data/processed' / cfg['name']
m = json.loads((root / 'manifest.json').read_text())
assert m['complete'] and m['config_sha256'] == hashlib.sha256(config_path.read_bytes()).hexdigest()
for name, digest in m['output_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
data = np.load(root / 'inputs.npz')
u = np.load(root / 'displacement_coefficients.npy', mmap_mode='r')
X = data['coords'][data['cells']]
inverse = np.linalg.inv((X[:, 1:]-X[:, :1]).transpose(0, 2, 1))
grad_lambda = np.concatenate((-inverse.sum(1, keepdims=True), inverse), axis=1)
edges = list(itertools.combinations(range(4), 2))
gradients = []
for bary in np.eye(4):
    gradients.append(np.concatenate(((4*bary[None, :, None]-1)*grad_lambda,
        np.stack([4*(bary[i]*grad_lambda[:, j]+bary[j]*grad_lambda[:, i])
                  for i, j in edges], axis=1)), axis=1))
gradients = np.stack(gradients, axis=1)
multi = np.array([a for a in itertools.product(range(4), repeat=4) if sum(a) == 3])
lattice = multi/3
multinomial = np.array([math.factorial(3)/math.prod(math.factorial(int(i)) for i in a)
                        for a in multi])


def basis(bary):
    return multinomial[None, :]*np.prod(np.asarray(bary)[:, None, :]**multi[None, :, :], axis=-1)


B = basis(lattice)
transform = np.linalg.inv(B)
assert np.linalg.cond(B) < 10
np.testing.assert_allclose(B.sum(1), 1, rtol=0, atol=1e-14)
check_bary = np.random.default_rng(2027).dirichlet(np.ones(4), size=7)
check_basis = basis(check_bary)
limit = min(len(u), args.max_steps) if args.max_steps else len(u)
margin = 1e-9
records = []
previous = np.zeros(u.shape[1])
for step in range(limit):
    current = u[step]
    record = dict(step=step, time_s=float(data['time'][step]))
    for name, coefficients in [('endpoint', current), ('stress', .4*previous+.6*current)]:
        values = np.asarray(coefficients)[data['canonical_cell_dofs']]
        vertex_F = np.eye(3)+np.einsum('eki,evkj->evij', values, gradients)
        lattice_F = np.einsum('qv,evij->eqij', lattice, vertex_F)
        determinant = np.linalg.det(lattice_F)
        bernstein = determinant @ transform.T
        lower = bernstein.min(axis=1)
        assert np.isfinite(bernstein).all()
        record[name+'_minimum_Bernstein_lower_bound'] = float(lower.min())
        record[name+'_unresolved_elements'] = int(np.count_nonzero(lower <= margin))
        if step in {0, limit//4, limit//2, limit-1}:
            direct = np.linalg.det(np.einsum('qv,evij->eqij', check_bary, vertex_F))
            reconstructed = bernstein @ check_basis.T
            np.testing.assert_allclose(direct, reconstructed, rtol=1e-11, atol=1e-11)
            # Integral of each degree-three Bernstein basis over a tetrahedron is V/20.
            volume = abs(np.linalg.det((X[:, 1:]-X[:, :1]).transpose(0, 2, 1)))/6
            integral = float(volume @ bernstein.mean(axis=1))
            qb = np.c_[1-data['qpoints'].sum(1), data['qpoints']]
            quadrature_det = np.linalg.det(np.einsum('qv,evij->eqij', qb, vertex_F))
            quadrature_integral = float((6*volume) @ (quadrature_det @ data['qweights']))
            np.testing.assert_allclose(integral, quadrature_integral, rtol=1e-11, atol=1e-13)
            record[name+'_polynomial_reconstruction_max_abs_error'] = float(abs(direct-reconstructed).max())
            record[name+'_integral_discrepancy'] = abs(integral-quadrature_integral)
    records.append(record)
    previous = current
    if (step+1) % 25 == 0:
        print('BERNSTEIN AUDIT', step+1, '/', limit, flush=True)
complete = limit == len(u)
result = dict(complete=complete, reference_name=cfg['name'],
    config_sha256=m['config_sha256'], output_sha256=m['output_sha256'],
    executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    checked_steps=limit, tetrahedra=len(X), positive_margin=margin,
    minimum_endpoint_lower_bound=min(r['endpoint_minimum_Bernstein_lower_bound'] for r in records),
    minimum_stress_lower_bound=min(r['stress_minimum_Bernstein_lower_bound'] for r in records),
    unresolved_endpoint_element_states=sum(r['endpoint_unresolved_elements'] for r in records),
    unresolved_stress_element_states=sum(r['stress_unresolved_elements'] for r in records),
    records=records,
    scope='P2 displacement has affine deformation gradient and cubic determinant. Its cubic tetrahedral Bernstein basis is nonnegative and partitions unity, so the minimum coefficient bounds detF throughout the element in exact arithmetic. Implemented in float64 with 1e-9 positive margin; random-point polynomial reconstruction and independent source-quadrature integration are checked at four frames. Not an interval-arithmetic certificate, nor a mechanics-accuracy or mesh-convergence result.')
result['whole_element_positivity_sufficient_condition_passed'] = bool(complete and
    result['unresolved_endpoint_element_states'] == 0 and result['unresolved_stress_element_states'] == 0)
suffix = '' if complete else '_pilot'
target = P/f'experiments/metrics/{cfg["name"]}_bernstein{suffix}.json'
target.write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps({k: v for k, v in result.items() if k not in ['records', 'scope', 'output_sha256']}, indent=2), flush=True)
