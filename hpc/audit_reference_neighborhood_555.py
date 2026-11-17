"""Inspect endpoint and stress-stage kinematics in cells supporting elastic-force spikes."""
import argparse
import hashlib
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

from audit_reference_cell_555 import cartesian_deformation, inspect_field
from audit_reference_transition_555 import ROOT, TARGET, ELEMENT, load
from registry import atomic, sha


def supporting_cells(maps, dofs):
    """Find cell support of the selected canonical degrees of freedom."""
    selection = []
    for start in range(0, len(maps), 8192):
        hit = np.isin(maps[start:start + 8192], list(dofs)).any(axis=(1, 2))
        selection.extend((start + np.flatnonzero(hit)).tolist())
    return selection


def point_metrics(deformation, fiber, sheet, parameters):
    determinant = np.linalg.det(deformation)
    stretches = np.linalg.svd(deformation, compute_uv=False)
    C = np.swapaxes(deformation, 1, 2) @ deformation
    I4f = np.einsum('qi,qij,qj->q', fiber, C, fiber)
    I4s = np.einsum('qi,qij,qj->q', sheet, C, sheet)
    I8fs = np.einsum('qi,qij,qj->q', fiber, C, sheet)
    positive = determinant > 0
    I1bar = np.full(len(determinant), np.nan)
    I1bar[positive] = determinant[positive] ** (-2/3) * np.trace(C[positive], axis1=1, axis2=2)
    return dict(
        determinant=determinant, smallest_stretch=stretches[:, -1], largest_stretch=stretches[:, 0],
        I1bar=I1bar, I4f=I4f, I4s=I4s, I8fs=I8fs,
        fiber_norm=np.linalg.norm(fiber, axis=1), sheet_norm=np.linalg.norm(sheet, axis=1),
        isotropic_exponent=parameters['b'] * (I1bar - 3),
        fiber_exponent=parameters['b_f'] * (I4f - 1)**2,
        sheet_exponent=parameters['b_s'] * (I4s - 1)**2,
        coupling_exponent=parameters['b_fs'] * I8fs**2,
    )


def summarize(metrics, barycentric):
    result = {}
    for name, values in metrics.items():
        finite = np.isfinite(values)
        if finite.any():
            low = int(np.flatnonzero(finite)[np.argmin(values[finite])])
            high = int(np.flatnonzero(finite)[np.argmax(values[finite])])
            result[name] = dict(min=float(values[low]), max=float(values[high]),
                                minimum_barycentric=barycentric[low].tolist(),
                                maximum_barycentric=barycentric[high].tolist(),
                                nonfinite_count=int((~finite).sum()))
        else:
            result[name] = dict(min=None, max=None, nonfinite_count=len(values))
    return result


def audit(run):
    from lv_surrogate.physics.parallel_fem import ParallelP2
    from lv_surrogate.physics.bernstein import BernsteinAudit

    plan = load(run / 'plan.json')
    assert plan['neighborhood']
    manifest = load(TARGET / 'manifest.json')
    assert manifest['completed_steps'] == 554
    protocol = load(Path(plan['protocol']))
    transition_path = ROOT / 'experiments/hpc/cmbbe-transition-audit-555-01/result.json'
    transition = load(transition_path)
    assert sha(transition_path) == plan['frozen'][str(transition_path)]
    hotspots = {}
    for row in transition['records']:
        if row['state'] < 552 or row['candidate'] not in ('previous_accepted_displacement', 'checkpoint_displacement'):
            continue
        for entry in row['components']['elastic']['largest_entries'][:3]:
            hotspots.setdefault(entry['dof'], []).append(dict(
                state=row['state'], candidate=row['candidate'], value_N=entry['value_N'],
                coordinate=entry['coordinate']))
    with np.load(TARGET / 'inputs.npz', allow_pickle=False) as inputs:
        cells = inputs['cells']
        maps = inputs['canonical_cell_dofs']
        selected = set(supporting_cells(maps, hotspots))
        # Include the limiting cell and its face neighbors.
        neighbors = np.flatnonzero(np.isin(cells, cells[ELEMENT]).sum(axis=1) >= 3)
        selected.update(neighbors.tolist())
        selected.add(ELEMENT)
        ids = np.array(sorted(selected), dtype=int)
        selected_maps, selected_cells = maps[ids], cells[ids]
        selected_vertices = inputs['coords'][selected_cells]
        times, pressures = inputs['time'], inputs['pressure']
        xyz = inputs['qpoints']
        quadrature = np.column_stack((1-xyz.sum(axis=1), xyz))
    del cells, maps
    assert np.all(quadrature >= -1e-12) and np.allclose(quadrature.sum(axis=1), 1.)
    adapter = ParallelP2(ROOT, protocol['source_reference'], manifest['config'])
    assert adapter.comm.size == 1
    mesh = adapter.geo.mesh
    tree = cKDTree(mesh.coordinates())
    distances, native_vertices = tree.query(selected_vertices.reshape(-1, 3))
    assert distances.max() < 1e-12
    count = len(mesh.coordinates())
    fiber = adapter.geo.f0.compute_vertex_values(mesh).reshape(3, count).T[native_vertices].reshape(-1, 4, 3)
    sheet = adapter.geo.s0.compute_vertex_values(mesh).reshape(3, count).T[native_vertices].reshape(-1, 4, 3)
    parameters = {name: float(value) for name, value in adapter.problem.material.parameters.items()}
    fields = np.load(TARGET / 'displacement_coefficients.npy', mmap_mode='r')
    with np.load(plan['seed'], allow_pickle=False) as seed:
        assert seed['checkpoint_sha256'].item() == plan['checkpoint_sha256']
        warm = seed['u'][selected_maps]
    report = dict(
        complete=False, accepted_states=554, limiting_element=ELEMENT, cells=ids.tolist(),
        hotspots={str(dof): entries for dof, entries in hotspots.items()}, material_parameters=parameters,
        native_vertex_coordinate_error=float(distances.max()), records=[], overview=[],
        reference_state_accepted=False,
        scope='Cell patch supporting the largest elastic entries plus limiting-cell face neighbors. '
              'Invariants are evaluated by independent Cartesian P2 interpolation with cached P1 fiber fields. '
              'Sampled extrema are estimates. Bernstein bounds apply to each examined cell.',
        exponent_scope='Arguments of the material exponential factors before fiber/sheet activation weights.')
    edges = list(combinations(range(4), 2))
    edge_points = []
    for i, j in edges:
        for weight in np.linspace(0., 1., 33):
            point = np.zeros(4)
            point[i], point[j] = weight, 1-weight
            edge_points.append(point)

    def evaluate(state, candidate, nodal, previous):
        for stage, local_values in (('endpoint', nodal), ('stress', .4*previous + .6*nodal)):
            records = []
            for index, element in enumerate(ids):
                vertices = selected_vertices[index]
                local = BernsteinAudit(dict(coords=vertices, cells=np.array([[0,1,2,3]]),
                                             canonical_cell_dofs=selected_maps[index:index+1]))
                barycentric = np.vstack((quadrature, local.lattice, edge_points))
                deformation = cartesian_deformation(vertices, local_values[index], barycentric @ vertices)
                fiber_values = barycentric @ fiber[index]
                sheet_values = barycentric @ sheet[index]
                metrics = point_metrics(deformation, fiber_values, sheet_values, parameters)
                vertex_F = cartesian_deformation(vertices, local_values[index], vertices)
                extrema = inspect_field(local, vertex_F)
                quadrature_summary = summarize(
                    {name: value[:len(quadrature)] for name, value in metrics.items()}, quadrature)
                records.append(dict(
                    element=int(element), vertex_ids=selected_cells[index].tolist(),
                    supports_hotspot_dofs=[int(dof) for dof in hotspots if dof in selected_maps[index]],
                    reference_edge_condition=float(np.linalg.cond((vertices[1:]-vertices[0]).T)),
                    nodal_increment_max_abs=float(np.max(abs(nodal[index]-previous[index]))),
                    extrema=summarize(metrics, barycentric), quadrature=quadrature_summary,
                    optimized_minimum_estimate=extrema['optimized_minimum_estimate'],
                    minimum_barycentric=extrema['minimum_barycentric'],
                    bernstein=extrema['bernstein']))
            row = dict(state=state, candidate=candidate, stage=stage,
                       time_s=float(times[state-1]), pressure_Pa=float(pressures[state-1]), cells=records)
            report['records'].append(row)
            overview = dict(state=state, candidate=candidate, stage=stage)
            for name in ('isotropic_exponent', 'fiber_exponent', 'sheet_exponent', 'coupling_exponent'):
                finite = [item for item in records if item['quadrature'][name]['max'] is not None]
                if finite:
                    largest = max(finite, key=lambda item: item['quadrature'][name]['max'])
                    overview[name] = dict(cell=largest['element'], value=largest['quadrature'][name]['max'])
            minimum = min(records, key=lambda item: item['extrema']['determinant']['min'])
            overview['sampled_min_determinant'] = dict(
                cell=minimum['element'], value=minimum['extrema']['determinant']['min'])
            report['overview'].append(overview)
            atomic(run / 'result.json', report)
            atomic(run / 'overview.json', dict(complete=False, rows=report['overview']))
            worst = max(records, key=lambda item: item['quadrature']['isotropic_exponent']['max']
                        if item['quadrature']['isotropic_exponent']['max'] is not None else float('-inf'))
            print('PATCH', state, candidate, stage, 'cells', len(records),
                  'worst_isotropic_exponent_cell', worst['element'],
                  'exponent', worst['quadrature']['isotropic_exponent']['max'], flush=True)

    for state in range(550, 555):
        journal = load(TARGET / f'step_journal/{state-1:06d}.json')
        assert journal['record'] == manifest['steps'][state-1]
        assert hashlib.sha256(np.asarray(fields[state-1]).tobytes()).hexdigest() == journal['coefficient_sha256']
        previous = fields[state-2][selected_maps]
        current = fields[state-1][selected_maps]
        evaluate(state, 'previous_accepted_displacement', previous, previous)
        evaluate(state, 'accepted_displacement', current, previous)
    old = fields[553][selected_maps]
    evaluate(555, 'checkpoint_displacement', old, old)
    evaluate(555, 'original_warm_start', warm, old)
    report['complete'] = True
    atomic(run / 'result.json', report)
    atomic(run / 'overview.json', dict(complete=True, rows=report['overview']))
    print('NEIGHBORHOOD AUDIT COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit',))
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    from re import fullmatch
    if not fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('Invalid run ID')
    audit(ROOT / 'experiments/hpc' / args.run_id)
