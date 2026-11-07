"""Regression checks for the experimental determinant path."""
import numpy as np
import pytest
import hashlib

from test_hpc_proof_benchmark import curved_fixture, same_proof, timed_pair
from cofactor_proof import CofactorBernsteinAudit, bulk_coefficients
from registry import atomic, sha
import benchmark_cofactor


def test_bulk_coefficients_and_near_margin_fallback(monkeypatch):
    matrices = np.eye(3) + .1 * np.random.default_rng(6).normal(size=(64, 20, 3, 3))
    matrices[0] = np.diag([1., 1., 1e-9])
    original = np.linalg.det
    transform = np.eye(20)
    expected = original(matrices)
    calls = []

    def traced(values):
        calls.append(values.copy())
        return original(values)

    monkeypatch.setattr(np.linalg, 'det', traced)
    actual = bulk_coefficients(matrices, transform, 1e-9)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], matrices[:1])
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_candidate_preserves_curved_subdivision_and_rejects_inversion():
    arrays, u = curved_fixture()
    baseline, _ = timed_pair(arrays, u, .98*u, 8)
    candidate = CofactorBernsteinAudit(arrays, workers=8)
    try:
        proof = candidate.pair(u, .98*u)
        assert proof['endpoint']['original_unresolved_elements'] == 32
        same_proof(baseline, proof)
        x = np.array([0., 1., 0., 0., .5, 0., 0., .5, .5, 0.])
        reflection = np.zeros((10, 3))
        reflection[:, 0] = -2*x
        with pytest.raises(ValueError, match='nonpositive'):
            candidate.pair(reflection.ravel(), reflection.ravel())
    finally:
        candidate.close()


def test_comparison_binds_both_archived_fields_and_writes_proof_artifacts(tmp_path, monkeypatch):
    arrays, u = curved_fixture()
    source = tmp_path / 'data/processed/archive'
    source.mkdir(parents=True)
    np.savez(source / 'inputs.npz', **arrays)
    fields = np.array([.98*u, u, u])
    np.save(source / 'displacement_coefficients.npy', fields)
    atomic(source / 'positivity.json', [
        {}, timed_pair(arrays, fields[1], fields[0], 8)[0],
        timed_pair(arrays, fields[2], fields[1], 8)[0],
    ])
    snapshot = tmp_path / 'snapshot.json'
    atomic(snapshot, dict(reference='archive', files={
        str(path.relative_to(tmp_path)): sha(path) for path in source.iterdir()}))
    run = tmp_path / 'experiments/hpc/comparison'
    run.mkdir(parents=True)
    plan = dict(snapshot='snapshot.json', snapshot_sha256=sha(snapshot),
                workers=8, fixed_steps=[1, 2], field_hashes={
                    str(i): hashlib.sha256(field.tobytes()).hexdigest()
                    for i, field in enumerate(fields)})
    monkeypatch.setattr(benchmark_cofactor, 'P', tmp_path)
    monkeypatch.setattr(benchmark_cofactor, 'check_resources', lambda _: 0)
    result = dict(records=[])
    benchmark_cofactor.compare(run, plan, result)
    assert [row['step'] for row in result['records']] == [1, 2]
    assert all(row['proof_agreement'] for row in result['records'])
    assert len(list(run.glob('proof-*.json'))) == 4
    plan['field_hashes']['0'] = 'changed'
    with pytest.raises(AssertionError):
        benchmark_cofactor.compare(run, plan, dict(records=[]))
