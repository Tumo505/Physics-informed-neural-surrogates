"""Check statistical units, timing units and rejection of incomplete artifacts."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import report_data_efficiency as report
from data_efficiency import digest, write


def example_run(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'evaluation').mkdir()
    write(run / 'protocol.json', dict(sizes=[64], seeds=[11, 22, 33]))
    write(run / 'state.json', dict(status='complete'))
    grouped, records = {}, []
    seed_means = {}
    for kind, offset in [('gnn', 0.), ('point', .5)]:
        seed_means[kind] = []
        total_seconds = 0.
        for index, seed in enumerate([11, 22, 33]):
            errors = np.array([1., 2., 4.]) + index * .1 + offset
            grouped[kind, str(seed)] = errors
            seed_means[kind].append(float(errors.mean()))
            seconds = 60 * (index + 1)
            total_seconds += seconds
            write(run / 'evaluation' / f'64-{kind}-{seed}.json', dict(
                parameters=100, mean_nodal_mm=float(errors.mean()),
                training_seconds=seconds, inference_seconds=.002,
                timing_scope='device resident; transfers excluded'))
        errors = np.mean([grouped[kind, str(s)] for s in [11, 22, 33]], axis=0) * .8
        grouped[kind, 'ensemble'] = errors
        write(run / 'evaluation' / f'64-{kind}-ensemble.json', dict(
            members=3, mean_nodal_mm=float(errors.mean()), training_seconds=total_seconds,
            inference_seconds=.007, timing_scope='sequential members on one GPU'))
    comparisons = []
    for mode in ('single', 'ensemble'):
        means = {kind: float(np.mean(seed_means[kind])) if mode == 'single' else
                 float(grouped[kind, 'ensemble'].mean()) for kind in ('gnn', 'point')}
        delta = means['gnn'] - means['point']
        comparisons.append(dict(size=64, comparison=mode, gnn_mean_mm=means['gnn'],
            point_mean_mm=means['point'], gnn_minus_point_mm=delta,
            paired_geometry_bootstrap_95_ci_mm=[delta - .1, delta + .1], per_seed_mean_mm=seed_means))
    result = dict(complete=True, comparisons=comparisons, seeds=[11, 22, 33],
                  test_geometries=3, ensemble_members=3, protocol_sha256=digest(run / 'protocol.json'))
    write(run / 'evaluation/result.json', result)
    for (kind, seed), values in grouped.items():
        for geometry, value in enumerate(values):
            records.append(dict(size=64, model=kind, seed=seed, geometry=geometry,
                                mean_nodal_mm=value, p95_nodal_mm=2 * value, max_nodal_mm=3 * value))
    report.csv_write(run / 'evaluation/per-geometry.csv', records)
    return run, result


def test_full_report_measured_units_and_outputs(tmp_path):
    run, _ = example_run(tmp_path)
    out = tmp_path / 'report'
    saved = report.build(out, run=run)
    assert saved['status'] == 'complete'
    assert saved['missing_measurements'] == []
    assert len(saved['cost_and_tail']) == 4
    single = next(r for r in saved['cost_and_tail'] if r['model'] == 'gnn' and r['comparison'] == 'single')
    ensemble = next(r for r in saved['cost_and_tail'] if r['model'] == 'gnn' and r['comparison'] == 'ensemble')
    assert single['training_worker_minutes'] == 2
    assert single['inference_ms'] == 2
    assert ensemble['training_worker_minutes'] == 6
    assert ensemble['inference_ms'] == 7
    assert single['parameters'] == 100 and ensemble['parameters'] == 300
    assert saved['total_training_worker_hours'] == .2
    assert np.isclose(single['mean_seed_geometry_p95_error_mm'], 3.9)
    assert single['worst_nodal_error_all_members_mm'] == pytest.approx(12.6)
    assert len(list(out.glob('*.svg'))) == 4
    assert len(list(out.glob('*.png'))) == 4
    assert 'sample SD' in (out / 'captions-and-limitations.txt').read_text()


def test_summary_only_does_not_invent_missing_measurements(tmp_path):
    run, _ = example_run(tmp_path)
    out = tmp_path / 'summary'
    saved = report.build(out, summary=run / 'evaluation/result.json')
    assert saved['status'] == 'accuracy_only'
    assert saved['total_training_worker_hours'] is None
    assert len(saved['missing_measurements']) == 2
    assert len(list(out.glob('*.svg'))) == 2
    assert not (out / 'cost-and-tail-table.csv').exists()
    with pytest.raises(FileExistsError):
        report.build(out, summary=run / 'evaluation/result.json')


@pytest.mark.parametrize('damage', ['duplicate', 'missing', 'different_mean', 'timing', 'protocol'])
def test_reject_inconsistent_artifacts_before_report_creation(tmp_path, damage):
    run, _ = example_run(tmp_path)
    path = run / 'evaluation/per-geometry.csv'
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if damage == 'duplicate':
        report.csv_write(path, rows + [rows[0]])
    elif damage == 'missing':
        report.csv_write(path, rows[1:])
    elif damage == 'different_mean':
        rows[0]['mean_nodal_mm'] = 2.
        report.csv_write(path, rows)
    elif damage == 'timing':
        metric = run / 'evaluation/64-gnn-11.json'
        value = report.read(metric)
        value['inference_seconds'] = -1
        write(metric, value)
    else:
        write(run / 'protocol.json', dict(sizes=[64], seeds=[1, 2, 3]))
    out = tmp_path / 'invalid-report'
    with pytest.raises(ValueError):
        report.build(out, run=run)
    assert not out.exists()


def test_accuracy_seed_sd_and_sign(tmp_path):
    _, result = example_run(tmp_path)
    rows = report.accuracy_rows(result)
    single = rows[0]
    assert single['graph_minus_pointwise_mm'] < 0
    assert single['graph_relative_error_reduction_pct'] > 0
    assert single['graph_seed_sd_mm'] == pytest.approx(.1)
    assert rows[1]['graph_seed_sd_mm'] == ''
