"""Inspect P2 coordinate/component correspondence using inputs only, no labels."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

P = Path(__file__).resolve().parents[1]
protocol = json.loads((P/'configs/physics/dynamic_quadratic_geometry_controls.json').read_text())
records = []
tolerance = 1e-12

def load(name):
    root = P/'data/processed'/name
    manifest = json.loads((root/'manifest.json').read_text())
    cfg = manifest['config']; data = np.load(root/'inputs.npz')
    scale = np.array([cfg['geometry_scales'][1], cfg['geometry_scales'][0], cfg['geometry_scales'][0]])
    maps = data['canonical_cell_dofs']
    components = np.full(len(data['dof_coordinates']), -1, dtype=np.int8)
    for c in range(3):
        indices = np.unique(maps[..., c])
        assert (components[indices] == -1).all()
        components[indices] = c
    assert (components >= 0).all()
    return data, components, scale

for level, baseline in [(0, 'cardiac_quadratic_reference_gamg_v1'),
                        (1, 'cardiac_quadratic_reference_refined_v1')]:
    a, ac, scale_a = load(baseline)
    for case in protocol['case_ids']:
        name = f'cardiac_quadratic_{case}_r{level}_q{protocol["quadrature_degree"]}_v1'
        b, bc, scale_b = load(name)
        vertex_distance, vertex_map = cKDTree(a['coords']/scale_a).query(b['coords']/scale_b)
        vertex_match = bool(vertex_distance.max() < tolerance and len(np.unique(vertex_map)) == len(a['coords']) == len(b['coords']))
        same_cells = False
        if vertex_match:
            cells_a = np.sort(a['cells'], axis=1)
            cells_b = np.sort(vertex_map[b['cells']], axis=1)
            def sorted_rows(x):
                return x[np.lexsort(x.T[::-1])]
            same_cells = bool(np.array_equal(sorted_rows(cells_a), sorted_rows(cells_b)))
        dof_map = np.full(len(bc), -1, dtype=np.int64)
        max_distance = 0.; unmatched = 0
        for c in range(3):
            ai = np.flatnonzero(ac == c); bi = np.flatnonzero(bc == c)
            distance, nearest = cKDTree(a['dof_coordinates'][ai]/scale_a).query(b['dof_coordinates'][bi]/scale_b)
            dof_map[bi] = ai[nearest]
            max_distance = max(max_distance, float(distance.max()))
            unmatched += int((distance >= tolerance).sum())
        bijective = bool(len(np.unique(dof_map)) == len(ac) == len(bc))
        all_dofs_match = bool(unmatched == 0 and bijective)
        records.append(dict(case=case, refinement_level=level, baseline=baseline, candidate=name,
            baseline_dofs=len(ac), candidate_dofs=len(bc),
            maximum_canonical_vertex_distance_m=float(vertex_distance.max()),
            canonical_vertices_bijectively_match=vertex_match,
            tetrahedral_connectivity_matches_after_vertex_permutation=same_cells,
            maximum_canonical_DOF_distance_m=max_distance, unmatched_component_DOFs=unmatched,
            component_DOF_mapping_bijective=bijective,
            all_component_DOFs_match=all_dofs_match,
            candidate_to_baseline_DOF_permutation_sha256=hashlib.sha256(dof_map.tobytes()).hexdigest() if all_dofs_match else None,
            input_sha256={n:hashlib.sha256((P/'data/processed'/n/'inputs.npz').read_bytes()).hexdigest() for n in [baseline,name]}))
result = dict(complete=True, coordinates_only=True, tolerance_m=tolerance, records=records,
    executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Input-only check on two validation geometries at two mesh levels; no displacement arrays, labels, calibration/test outcomes or model weights read. Undo the exact affine geometry scaling, then compare coordinates and physical vector components. Reordering is required even when DOF sets match. Matching coordinate/component DOF sets permit shared coefficient representation only after explicit permutation; different element connectivity requires case-specific mechanics and graph operators. Nonmatching DOF sets would require interpolation or another representation. Does not establish mechanics accuracy, numerical convergence or population correspondence.')
(P/'experiments/metrics/dynamic_quadratic_geometry_correspondence.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result,indent=2))
