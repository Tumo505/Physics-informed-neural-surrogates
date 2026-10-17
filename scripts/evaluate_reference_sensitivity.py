"""Frozen-model sensitivity to a changed FE target on one known baseline case."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'src'))
from lv_surrogate.models.dynamic_graph import DynamicGraph
torch.set_num_threads(2)
comparison = json.loads((P/'experiments/metrics/dynamic_quadratic_comparison.json').read_text())
assert comparison['complete'] and comparison['native_norm_verification']['passed']
freeze = json.loads((P/'experiments/manifests/dynamic_final_freeze.json').read_text())
for path, digest in freeze['checkpoint_sha256'].items():
    assert hashlib.sha256((P/path).read_bytes()).hexdigest() == digest
training = json.loads((P/'configs/physics/dynamic_training.json').read_text())
evaluation = json.loads((P/'experiments/metrics/dynamic_final_evaluation.json').read_text())
base = P/'data/processed/cardiac_dynamic_v1/convergence_dt0.005'
trajectory = dict(np.load(base/'trajectory.npz'))
geometry = dict(np.load(base/'geometry.npz'))
qroot = P/comparison['reference_dataset']
qinputs = dict(np.load(qroot/'inputs.npz'))
np.testing.assert_array_equal(geometry['cells'], qinputs['cells'])
np.testing.assert_allclose(geometry['coords'], qinputs['coords'], rtol=0, atol=1e-14)
indices = np.searchsorted(qinputs['time'], trajectory['time'])
np.testing.assert_allclose(qinputs['time'][indices], trajectory['time'], rtol=0, atol=1e-14)
quadratic = np.load(qroot/'displacement_vertices.npy', mmap_mode='r')[indices]
truths = {'P1_dt005': trajectory['u'], 'P2_changed_reference': quadratic}
top = torch.tensor(np.load(P/'data/processed/cardiac_geometry_ml/topology.npy'))
labels = np.load(P/'data/processed/cardiac_geometry_ml/cluster_labels.npy')
counts = np.bincount(labels, minlength=64)
points = np.c_[geometry['coords']/.1, geometry['fiber'], geometry['sheet']]
pooled = np.zeros((64, 9)); np.add.at(pooled, labels, points); pooled /= counts[:, None]
delta = pooled[top[:, 1], :3]-pooled[top[:, 0], :3]
b = [torch.tensor(x[None], dtype=torch.float64) for x in [pooled, np.zeros(3),
     np.c_[delta, np.linalg.norm(delta, axis=-1)], trajectory['pressure']/16070.954]]
records = {}
for key in evaluation['models']:
    regime, label = key.split('/')
    name = training['name']+'_'+regime
    stats = dict(np.load(P/f'data/processed/{training["data_tag"]}_{regime}/normalizers.npz'))
    members = []
    with torch.inference_mode():
        for seed in training['seeds']:
            net = DynamicGraph(stats).double()
            net.load_state_dict(torch.load(P/f'experiments/checkpoints/{name}/{label}_seed{seed}.pt',
                                          map_location='cpu', weights_only=False)['model'])
            net.eval()
            members.append(net.decode(net(*b[:3], top, b[3])[0]).reshape(200, -1, 3))
        prediction = torch.stack(members).mean(0).numpy()
    record = {}
    for target, truth in truths.items():
        error = prediction-truth
        joint = float(np.linalg.norm(error, axis=-1).max())
        balls = evaluation['models'][key]['splits']['test']['coverage']
        record[target] = dict(component_rmse_m=float(np.sqrt(np.mean(error**2))),
            maximum_nodal_L2_error_over_complete_cycle_m=joint,
            original_coarse_reference_balls=[dict(nominal=c['nominal'], radius_m=c['radius_m'],
                contains_this_single_case=joint<=c['radius_m']) for c in balls])
    records[key] = record
result = dict(complete=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    frozen_checkpoint_sha256=freeze['checkpoint_sha256'], n_cases=1, n_common_endpoints=200,
    reference_change_component_rmse_m=float(np.sqrt(np.mean((trajectory['u']-quadratic)**2))),
    models=records, quadratic_reference_dt_s=comparison['reference_dt_s'],
    preferred_from_original_validation=evaluation['preferred_from_validation_only'],
    scope='One known baseline numerical geometry/waveform, fixed models, no selection/refitting. '
          'Combined element-order AND timestep reference change; not an independent population test. '
          'Original conformal radii target coarse P1/5ms labels; their containment here does not '
          'establish coverage against the changed P2 reference or continuum truth. '
          'No cases are added to the20-test cohort.')
(P/'experiments/metrics/dynamic_reference_sensitivity.json').write_text(json.dumps(result, indent=2,
                                                                                  allow_nan=False))
print('FROZEN REFERENCE SENSITIVITY COMPLETE', flush=True)
