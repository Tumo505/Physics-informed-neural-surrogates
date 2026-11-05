"""Proof equivalence and bounded submission checks on tiny synthetic fields only."""
import copy
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'hpc')]
from proof_benchmark import timed_pair, same_proof, worker_summary, WORKERS
from registry import atomic, artifact, sha
import submit_proof_benchmark as launcher


def curved_fixture():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    nodes = np.concatenate([vertices, np.array([(vertices[i] + vertices[j]) / 2
                            for i, j in itertools.combinations(range(4), 2)])])
    x, y, z = nodes.T
    # det(F)=(x-.5)^2+.1*y+.01 > 0, with inconclusive parent
    # Bernstein bounds requiring genuine subdivisions of this curved P2 map.
    mapped = np.stack([.5*x*x - .5*x - .1*y, x*y + .1*x - .5*y, z], axis=1)
    u = (mapped - nodes).ravel()
    arrays = dict(coords=vertices, cells=np.tile(np.arange(4), (32, 1)),
                  canonical_cell_dofs=np.tile(np.arange(30).reshape(1, 10, 3), (32, 1, 1)))
    return arrays, u


def test_actual_curved_proof_subdivision_identical_across_worker_counts(tmp_path):
    arrays, u = curved_fixture()
    baseline, _ = timed_pair(arrays, u, u, 4)
    assert baseline['endpoint']['original_unresolved_elements'] == 32
    records = []
    for workers in [1, 2, 4, 8, 16]:
        proof, row = timed_pair(arrays, u, u, workers)
        same_proof(baseline, proof)
        assert len(row['worker_calls']) == 2 * workers
        assert row['process_cpu_seconds'] >= 0 and row['pair_seconds'] > 0
        records.append(row)
    profiles = tmp_path / 'profiles'
    profiles.mkdir()
    proof, row = timed_pair(arrays, u, u, 4, profile_dir=profiles)
    same_proof(baseline, proof)
    records.append(row)
    assert len(list(profiles.glob('*.prof'))) == len(list(profiles.glob('*.json'))) == 8
    assert len(worker_summary(records)) == 5
    assert all(r['repetitions'] == 1 for r in worker_summary(records))
    assert WORKERS[:5] == list(reversed(WORKERS[5:]))


def test_invalid_field_and_changed_subdivision_rejected():
    arrays, u = curved_fixture()
    with pytest.raises(ValueError, match='nonpositive'):
        # A linear reflection has a negative determinant everywhere.
        v = np.zeros((10, 3))
        v[:, 0] = -2 * np.concatenate([arrays['coords'], np.array([
            (arrays['coords'][i]+arrays['coords'][j])/2 for i, j in itertools.combinations(range(4), 2)])])[:, 0]
        timed_pair(arrays, v.ravel(), v.ravel(), 4)
    proof, _ = timed_pair(arrays, u, u, 4)
    other = copy.deepcopy(proof)
    other['endpoint']['refined_elements'][0]['maximum_depth'] += 1
    with pytest.raises(AssertionError, match='topology'):
        same_proof(proof, other)
    other = copy.deepcopy(proof)
    other['stress']['minimum_Bernstein_bound'] = 0
    with pytest.raises(AssertionError):
        same_proof(proof, other)


def test_full_worker_protocol_on_fixed_synthetic_archived_fields(monkeypatch, tmp_path):
    import proof_benchmark as benchmark
    monkeypatch.setattr(benchmark, 'P', tmp_path)
    arrays, u = curved_fixture()
    root = tmp_path / 'data/processed/source'
    root.mkdir(parents=True)
    np.savez(root / 'inputs.npz', **arrays)
    np.save(root / 'displacement_coefficients.npy', np.stack([u] * 4))
    proof, _ = timed_pair(arrays, u, u, 4)
    atomic(root / 'positivity.json', [dict(step=i, **proof) for i in range(4)])
    np.savez(tmp_path / 'start.npz', completed_steps=2, u_old=u)
    import hashlib
    digest = hashlib.sha256(u.tobytes()).hexdigest()
    atomic(tmp_path / 'snapshot.json', dict(reference='source', files={
           str(path.relative_to(tmp_path)): sha(path) for path in [root / 'inputs.npz', root / 'positivity.json']}))
    atomic(tmp_path / 'plan.json', dict(snapshot='snapshot.json', snapshot_sha256=sha(tmp_path / 'snapshot.json'),
           scope='Synthetic fixture only', known_start_checkpoint='start.npz',
           fixed_case=dict(step=2, previous_step=1, field_sha256=digest, previous_field_sha256=digest)))
    output = tmp_path / 'out'
    output.mkdir()
    before = {path: sha(path) for path in root.iterdir()}
    benchmark.run(tmp_path / 'plan.json', output)
    result = json.loads((output / 'result.json').read_text())
    assert result['complete'] and result['passed'] and len(result['records']) == 10
    assert len(result['worker_summary']) == 5
    assert all(r['repetitions'] == 2 for r in result['worker_summary'])
    assert not result['reference_continuation_allowed'] and not result['physical_new_state_generated']
    assert result['worker_profile']['profiled']
    assert before == {path: sha(path) for path in before}
    with pytest.raises(AssertionError, match='overwrite'):
        benchmark.run(tmp_path / 'plan.json', output)


def profile_parent(root):
    parent = root / 'experiments/hpc/profile-01'
    original = root / 'experiments/hpc/original'
    parent.mkdir(parents=True)
    original.mkdir()
    atomic(original / 'state.json', dict(platform_output='experiments/hpc/original/audit.json'))
    atomic(original / 'audit.json', dict(passed=True, audit_complete=True, checked_steps=4,
           records=[dict(step=i, field_sha256=f'fixture-{i}') for i in range(4)]))
    atomic(root / 'snapshot.json', dict(accepted_steps=4))
    paths = [original / 'state.json', original / 'audit.json', root / 'snapshot.json']
    plan = dict(parent_run='original', ranks=12, cpus=28, memory_gb=240, queue='workq',
                software='PBSJobApp', snapshot='snapshot.json', known_start_checkpoint='start.npz',
                cumulative_seconds_before=34433.3, cumulative_pipeline_budget_hours=192,
                scientific_sources={}, diagnostic_sources={},
                parent_files_sha256={str(p.relative_to(root)): sha(p) for p in paths})
    atomic(parent / 'plan.json', plan)
    atomic(parent / 'summary.json', dict(diagnostic_complete=True, stop_reason=None,
           rank_reports_received=12, expected_ranks=12, recorded_custom_iterations=7,
           known_pair_qualified=False, recorded_linear_seconds=118.18,
           cumulative_seconds=37078.3))
    atomic(parent / 'rank-000.json', dict(status='solver_timeout', histories_advanced=False,
           functions=[dict(file='bernstein.py', function='pair', inclusive_seconds=1609.52)]))
    atomic(parent / 'state.json', dict(status='diagnostic_complete'))
    for name in ['plan.json', 'summary.json', 'rank-000.json']:
        artifact(root, parent, parent / name, 'Synthetic profile fixture')
    return parent


def test_measured_profile_required_and_cumulative_budget_preserved(tmp_path):
    parent = profile_parent(tmp_path)
    _, elapsed, case, hashes = launcher.evidence(tmp_path, parent)
    assert elapsed == 37078.3 and case['step'] == 2 and case['previous_step'] == 1
    assert 'experiments/hpc/profile-01/summary.json' in hashes
    report = json.loads((parent / 'rank-000.json').read_text())
    report['functions'][0]['inclusive_seconds'] = 1
    atomic(parent / 'rank-000.json', report)
    artifact(tmp_path, parent, parent / 'rank-000.json', 'Synthetic counterexample')
    with pytest.raises(AssertionError, match='bottleneck'):
        launcher.evidence(tmp_path, parent)


def test_single_submission_uses_no_mpi_and_never_changes_parent(monkeypatch, tmp_path):
    parent = profile_parent(tmp_path)
    for name in ['proof_benchmark.py', 'submit_proof_benchmark.py', 'submit_probe_profile.py']:
        path = tmp_path / 'hpc' / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('# test fixture\n')
    before = {p: sha(p) for p in parent.rglob('*') if p.is_file()}
    monkeypatch.setattr(launcher, 'P', tmp_path)
    monkeypatch.setattr(launcher, 'assert_idle', lambda root: '')
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *a, **k: 'fixture-revision\n')
    calls = []
    def qsub(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='456.pbs\n', stderr='')
    monkeypatch.setattr(launcher.subprocess, 'run', qsub)
    args = SimpleNamespace(run_id='proof-01', parent_run='profile-01')
    launcher.submit(args)
    assert len(calls) == 1 and 'walltime=03:00:00' in calls[0]
    assert 'select=1:ncpus=28:mem=240gb' in calls[0] and 'place=excl' in calls[0]
    plan = json.loads((tmp_path / 'experiments/hpc/proof-01/plan.json').read_text())
    assert plan['cumulative_seconds_before'] == 37078.3
    assert plan['outer_seconds'] == 7200 and plan['case_seconds'] == 900
    assert plan['physical_states_to_generate'] == 0 and not plan['reference_continuation_allowed']
    assert before == {p: sha(p) for p in before}
    with pytest.raises(FileExistsError):
        launcher.submit(args)
    assert len(calls) == 1
