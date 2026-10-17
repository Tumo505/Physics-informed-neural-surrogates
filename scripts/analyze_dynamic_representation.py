"""Post-hoc oracle diagnostic only: frozen POD span and coefficient-bound floors."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from scipy.optimize import lsq_linear

P = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
frozen = json.loads((P/'experiments/manifests/dynamic_final_freeze.json').read_text())
confirmation = json.loads((P/'experiments/metrics/dynamic_confirmation_evaluation.json').read_text())
training = json.loads((P/'configs/physics/dynamic_training.json').read_text())
buffers = {}
for path, digest in frozen['checkpoint_sha256'].items():
    file = P/path
    assert hashlib.sha256(file.read_bytes()).hexdigest() == digest
    state = torch.load(file, map_location='cpu', weights_only=False)['model']
    regime = 'limited' if '_limited/' in path else 'full'
    values = {key: state[key].double().numpy() for key in
              ['basis', 'coefficient_scale', 'coefficient_bound']}
    if regime in buffers:
        for key in values: np.testing.assert_array_equal(values[key], buffers[regime][key])
    else: buffers[regime] = values
cases = confirmation['models']['full/data']['pooled_primary_and_confirmation']['per_case']
ids = [case['case'] for case in cases]
assert len(ids) == len(set(ids)) == 20
result = dict(complete=True, diagnostic_only=True, n_geometries=20, regimes={}, models={},
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    frozen_checkpoint_sha256=frozen['checkpoint_sha256'],
    scope='Post-hoc projection of already evaluated held-out labels. Oracle coefficients are '
          'not deployable predictions, physically certified fields or additional training data. '
          'No architecture, weights, rank, bounds or model selection is changed. '
          'All errors still target the coarse P1/5ms reference.')
for regime, values in buffers.items():
    basis = values['basis']; gram = basis@basis.T
    chol = np.linalg.cholesky(gram); A = chol.T
    inverse = np.linalg.inv(gram)
    bound = values['coefficient_scale']*values['coefficient_bound']
    assert np.all(bound>0)
    per_case = []
    for id in ids:
        root = P/f'data/processed/cardiac_dynamic_v1/{id}'
        with np.load(root/'trajectory.npz') as data:
            truth = data['u'].reshape(200, -1)
        coefficients = truth@basis.T@inverse
        residual = truth-coefficients@basis
        assert np.linalg.norm(residual@basis.T) < 1e-8*max(np.linalg.norm(truth), 1e-30)
        span_mse = float(np.mean(residual**2))
        constrained = coefficients.copy()
        outside = (abs(coefficients)>bound).any(1)
        for frame in np.flatnonzero(outside):
            fit = lsq_linear(A, A@coefficients[frame], bounds=(-bound, bound),
                             method='bvls', tol=1e-12, max_iter=200)
            assert fit.success and fit.optimality < 1e-9
            constrained[frame] = fit.x
        delta = constrained-coefficients
        penalty = float(np.mean(np.sum(delta*(delta@gram), axis=1))/truth.shape[1])
        per_case.append(dict(case=id, span_oracle_component_mse_m2=span_mse,
            bounded_oracle_component_mse_m2=span_mse+penalty,
            unconstrained_projection_frames_outside_readout_box=int(outside.sum())))
    result['regimes'][regime] = dict(rank=len(basis),
        basis_Gram_condition=float(np.linalg.cond(gram)), per_case=per_case,
        mean_span_oracle_rmse_m=float(np.mean([np.sqrt(r['span_oracle_component_mse_m2']) for r in per_case])),
        mean_bounded_oracle_rmse_m=float(np.mean([np.sqrt(r['bounded_oracle_component_mse_m2']) for r in per_case])))
    floor = {r['case']:r for r in per_case}
    for key, model in confirmation['models'].items():
        if not key.startswith(regime+'/'): continue
        rows = model['pooled_primary_and_confirmation']['per_case']
        total = np.array([r['component_rmse_m']**2 for r in rows])
        oracle = np.array([floor[r['case']]['bounded_oracle_component_mse_m2'] for r in rows])
        assert np.all(total>=oracle-1e-12)
        result['models'][key] = dict(mean_recorded_rmse_m=float(np.mean(np.sqrt(total))),
            bounded_oracle_fraction_of_geometry_mean_MSE=float(oracle.sum()/total.sum()),
            excess_geometry_mean_MSE_above_bounded_oracle_m2=float(np.mean(total-oracle)))
(P/'experiments/metrics/dynamic_representation_diagnostic.json').write_text(
    json.dumps(result, indent=2, allow_nan=False))
print('FROZEN REPRESENTATION DIAGNOSTIC COMPLETE', result['models'], flush=True)
