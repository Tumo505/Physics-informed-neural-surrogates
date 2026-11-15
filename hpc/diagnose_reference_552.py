"""Compare accepted displacement increments around sampled state-552 predictor inversions."""
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'src'))
from lv_surrogate.physics.bernstein import BernsteinAudit
from registry import sha

TARGET = P / 'data/processed/cardiac_hpc_cmbbe-cofactor-ref-03'
ACCEPTED = 551
WINDOW = 8


def predictor_witnesses(audit, old, predicted, alpha):
    """Evaluate the production lattice and retain one direct witness per sampled cell."""
    stages = {'endpoint': predicted, 'stress': alpha * old + (1 - alpha) * predicted}
    witnesses = {name: [] for name in stages}
    derivatives = np.eye(4)
    for start in range(0, len(audit.maps), audit.chunk):
        stop = min(start + audit.chunk, len(audit.maps))
        gl = audit.gl[start:stop]
        grad = np.stack([
            np.concatenate(((4 * b[None, :, None] - 1) * gl,
                            np.stack([4 * (b[i] * gl[:, j] + b[j] * gl[:, i])
                                      for i, j in audit.edges], axis=1)), axis=1)
            for b in derivatives
        ], axis=1)
        maps = audit.maps[start:stop]
        for name, field in stages.items():
            deformation = np.eye(3) + np.einsum('eki,evkj->evij', field[maps], grad)
            samples = np.linalg.det(np.einsum('qv,evij->eqij', audit.lattice, deformation))
            assert np.isfinite(samples).all(), f'Nonfinite determinant samples in {name}'
            minima = samples.min(axis=1)
            for index in np.flatnonzero(minima <= 0):
                point = int(samples[index].argmin())
                witnesses[name].append((start + int(index), float(minima[index]), point))
    return witnesses


def increment_stats(field_delta, maps, cells):
    if not len(cells):
        return None
    maxima = np.empty(len(cells))
    for start in range(0, len(cells), 8192):
        selection = cells[start:start + 8192]
        maxima[start:start + len(selection)] = np.linalg.norm(
            field_delta[maps[selection]], axis=2).max(axis=1)
    return dict(median_cell_max=float(np.median(maxima)), max_cell_max=float(maxima.max()),
                max_cell=int(cells[maxima.argmax()]))


def diagnose(target):
    manifest = json.loads((target / 'manifest.json').read_text())
    assert manifest['completed_steps'] == len(manifest['steps']) == ACCEPTED
    assert manifest['steps'][-1]['recovery_admission']
    journal = json.loads((target / 'step_journal/000550.json').read_text())
    checkpoint_path = target / 'restart.npz'
    inputs_path = target / 'inputs.npz'
    assert journal['record'] == manifest['steps'][-1]
    assert sha(checkpoint_path) == journal['accepted_checkpoint_sha256']
    assert sha(inputs_path) == manifest['source_inputs_sha256']
    fields = np.load(target / 'displacement_coefficients.npy', mmap_mode='r')
    with np.load(checkpoint_path) as checkpoint, np.load(inputs_path) as inputs:
        assert int(checkpoint['completed_steps']) == ACCEPTED
        old = checkpoint['u_old']
        np.testing.assert_array_equal(old, fields[ACCEPTED - 1])
        predicted = old + manifest['config']['dt_s'] * checkpoint['v_old']
        audit = BernsteinAudit(inputs, chunk=2048)
    # The native reference uses alpha_f=0.4 for its stress evaluation.
    witnesses = predictor_witnesses(audit, old, predicted, alpha=.4)
    inverted = np.array(sorted({row[0] for rows in witnesses.values() for row in rows}), dtype=int)
    all_cells = np.arange(len(audit.maps))
    increments = []
    for index in range(ACCEPTED - WINDOW, ACCEPTED):
        delta = fields[index] - fields[index - 1]
        increments.append(dict(from_state=index, to_state=index + 1,
                               all_cells=increment_stats(delta, audit.maps, all_cells),
                               sampled_witness_cells=increment_stats(delta, audit.maps, inverted)))
    stages = {}
    for stage, rows in witnesses.items():
        worst = sorted(rows, key=lambda row: row[1])[:10]
        stages[stage] = dict(cells_with_sampled_witness=len(rows), worst=[
            dict(cell=cell, determinant=value, barycentric_point=audit.lattice[point].tolist())
            for cell, value, point in worst])
    return dict(accepted_states=ACCEPTED, predictor_target=ACCEPTED + 1,
                checkpoint_sha256=journal['accepted_checkpoint_sha256'],
                cell_count=len(all_cells), witness_cell_count=len(inverted),
                determinant_sampling='Direct samples on the production cubic lattice. '
                                     'Subdivision-only witnesses may be missed.',
                increment_metric='Per-cell maximum Euclidean displacement increment across P2 nodes.',
                stages=stages, recent_increments=increments)


WITNESS_CELLS = (813768, 813772, 826438, 770432)


def face_neighborhood(cells, seeds):
    # A tetrahedron shares a face when it contains the same three vertex IDs.
    neighborhood = set(seeds)
    for seed in seeds:
        for face in combinations(cells[seed], 3):
            hits = np.flatnonzero(np.isin(cells, face).sum(axis=1) == 3)
            neighborhood.update(map(int, hits))
    return sorted(neighborhood)


def local_kinematics(audit, fields, cell_ids):
    gl = audit.gl
    grad = np.stack([
        np.concatenate(((4 * b[None, :, None] - 1) * gl,
                        np.stack([4 * (b[i] * gl[:, j] + b[j] * gl[:, i])
                                  for i, j in audit.edges], axis=1)), axis=1)
        for b in np.eye(4)
    ], axis=1)
    results = {}
    previous = None
    for label, field in fields.items():
        gradient = np.einsum('eki,evkj->evij', field[audit.maps], grad)
        deformation = np.eye(3) + gradient
        samples = np.linalg.det(np.einsum('qv,evij->eqij', audit.lattice, deformation))
        rows = []
        for i, cell in enumerate(cell_ids):
            point = int(samples[i].argmin())
            minimum = float(samples[i, point])
            bound = None
            if label.startswith('accepted_'):
                coefficients = samples[i] @ audit.transform.T
                bound = float(coefficients.min())
                if bound <= audit.margin:
                    bound = audit.resolve(deformation[i], audit.X[i])['lower_bound']
            rows.append(dict(cell=cell, sampled_min_det=minimum,
                             barycentric_point=audit.lattice[point].tolist(),
                             whole_element_lower_bound=bound,
                             max_grad_u_frobenius=float(np.linalg.norm(gradient[i], axis=(1, 2)).max()),
                             max_grad_change_frobenius=None if previous is None else float(
                                 np.linalg.norm(gradient[i] - previous[i], axis=(1, 2)).max())))
        results[label] = rows
        if label.startswith('accepted_'):
            previous = gradient
    return results


def compare_neighborhood(target):
    manifest = json.loads((target / 'manifest.json').read_text())
    assert manifest['completed_steps'] == len(manifest['steps']) == ACCEPTED
    assert manifest['steps'][-1]['recovery_admission']
    restart = target / 'restart.npz'
    inputs_path = target / 'inputs.npz'
    assert sha(inputs_path) == manifest['source_inputs_sha256']
    fields = np.load(target / 'displacement_coefficients.npy', mmap_mode='r')
    accepted = {}
    checkpoint_hash = sha(restart)
    for step in range(547, ACCEPTED):
        journal = json.loads((target / f'step_journal/{step:06d}.json').read_text())
        assert journal['record'] == manifest['steps'][step]
        assert hashlib.sha256(fields[step].tobytes()).hexdigest() == journal['coefficient_sha256']
        accepted[f'accepted_{step + 1}'] = fields[step]
    assert checkpoint_hash == journal['accepted_checkpoint_sha256']
    with np.load(restart) as checkpoint, np.load(inputs_path) as inputs:
        assert int(checkpoint['completed_steps']) == ACCEPTED
        old = checkpoint['u_old']
        np.testing.assert_array_equal(old, fields[ACCEPTED - 1])
        predicted = old + manifest['config']['dt_s'] * checkpoint['v_old']
        cells = inputs['cells']
        neighborhood = face_neighborhood(cells, WITNESS_CELLS)
        audit = BernsteinAudit(dict(coords=inputs['coords'], cells=cells[neighborhood],
                                    canonical_cell_dofs=inputs['canonical_cell_dofs'][neighborhood]))
    results = local_kinematics(audit, dict(accepted, velocity_endpoint=predicted,
                                         velocity_stress=.4 * old + .6 * predicted), neighborhood)
    return dict(accepted_states=ACCEPTED, checkpoint_sha256=checkpoint_hash,
                witness_cells=list(WITNESS_CELLS), neighborhood_cells=neighborhood,
                determinant_sampling='Production lattice. Accepted-state lower bounds use Bernstein subdivision as needed.',
                gradient_comparison='Gradient change is relative to the preceding accepted state, or state 551 for either predictor stage.',
                fields=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neighborhood', action='store_true',
                        help='Compare accepted states 548–551 and the velocity predictor locally')
    args = parser.parse_args()
    result = compare_neighborhood(TARGET) if args.neighborhood else diagnose(TARGET)
    print(json.dumps(result, indent=2, allow_nan=False))
