import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
from diagnose_reference_552 import BernsteinAudit, increment_stats, predictor_witnesses


def test_endpoint_and_stress_witnesses_and_local_increment():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    nodes = np.vstack((vertices, [(vertices[i] + vertices[j]) / 2
                                 for i in range(4) for j in range(i + 1, 4)]))
    maps = np.arange(30).reshape(1, 10, 3)
    audit = BernsteinAudit(dict(coords=vertices, cells=[[0, 1, 2, 3]],
                                canonical_cell_dofs=maps))
    predicted = np.zeros((10, 3))
    predicted[:, 0] = -1.5 * nodes[:, 0]
    predicted = predicted.ravel()
    witnesses = predictor_witnesses(audit, np.zeros(30), predicted, alpha=.4)
    assert witnesses['stress'] == []
    assert len(witnesses['endpoint']) == 1
    cell, determinant, _ = witnesses['endpoint'][0]
    assert cell == 0
    np.testing.assert_allclose(determinant, -.5)
    assert increment_stats(predicted, maps, np.array([cell]))['max_cell_max'] == 1.5
