from decimal import Decimal, localcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))

from lv_surrogate.physics.adaptive_bernstein import (
    EDGES, GROUPS, TRIPLES, child_bounds, coefficient_bounds, resolve_positive,
)
from lv_surrogate.physics.bernstein import BernsteinAudit
from refined_positivity_policy import install_adaptive_proof


VERTICES = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])


def exact_coefficients(matrices):
    def det(a):
        return (a[0,0]*(a[1,1]*a[2,2]-a[1,2]*a[2,1])
                -a[0,1]*(a[1,0]*a[2,2]-a[1,2]*a[2,0])
                +a[0,2]*(a[1,0]*a[2,1]-a[1,1]*a[2,0]))

    values = []
    for triple in TRIPLES:
        mixed = np.stack([matrices[triple[j],:,j] for j in range(3)], axis=1)
        values.append(det(mixed))
    return [sum((values[i] for i in group), Decimal(0))/len(group) for group in GROUPS]


@pytest.mark.parametrize('cancellation', [False, True])
def test_interval_coefficients_and_every_child_enclose_high_precision_polar_form(cancellation):
    f = np.random.default_rng(813578).normal(size=(4,3,3))
    if cancellation:
        f[:,:,2] = f[:,:,0]+1e-12*f[:,:,2]
    low, high = coefficient_bounds(f)
    lows, highs = child_bounds(low, high)
    with localcontext() as context:
        context.prec = 200
        exact = np.array([Decimal.from_float(float(x)) for x in f.flat], dtype=object).reshape(f.shape)
        for lo, hi, value in zip(low, high, exact_coefficients(exact), strict=True):
            assert Decimal.from_float(lo) <= value <= Decimal.from_float(hi)
        for edge_index, (a,b) in enumerate(EDGES):
            for side, replace in enumerate((a,b)):
                restricted = exact.copy()
                restricted[replace] = (exact[a]+exact[b])/2
                for lo, hi, value in zip(lows[edge_index,side], highs[edge_index,side],
                                         exact_coefficients(restricted), strict=True):
                    assert Decimal.from_float(lo) <= value <= Decimal.from_float(hi)


def narrow_positive(epsilon, *, negative=False):
    # These affine vertex gradients come from a complete P2 displacement:
    # F11=F22=x-.37, F12=-sqrt(eps), F21=sqrt(eps)+y, F33=1.
    # Its determinant is (x-.37)^2+eps+sqrt(eps)*y, with minimum eps.
    s = np.sqrt(epsilon)
    f = np.zeros((4,3,3))
    f[:,0,0] = f[:,1,1] = VERTICES[:,0]-.37
    f[:,0,1] = s if negative else -s
    f[:,1,0] = s+VERTICES[:,1]
    f[:,2,2] = 1
    return f


@pytest.mark.parametrize('minimum', [1e-4, 1e-6, 1e-8, 2e-9])
def test_narrow_positive_p2_field_gets_valid_bound_above_unchanged_margin(minimum):
    proof = resolve_positive(narrow_positive(minimum), VERTICES)
    assert 1e-9 < proof['lower_bound'] <= minimum
    assert proof['nodes_visited'] <= 20000 and proof['maximum_depth'] <= 64


def test_hidden_inversion_with_positive_original_vertices_is_rejected():
    f = narrow_positive(1e-5, negative=True)
    assert np.linalg.det(f).min() > 0
    with pytest.raises(ValueError, match='witness'):
        resolve_positive(f, VERTICES)


def test_determinant_directed_split_resolves_a_positive_field_blocked_by_geometric_depth_32():
    f = narrow_positive(1e-8)
    old = BernsteinAudit(dict(coords=VERTICES, cells=np.array([[0,1,2,3]]),
                              canonical_cell_dofs=np.arange(30).reshape(1,10,3)), max_depth=32)
    with pytest.raises(ValueError, match='unresolved at maximum depth'):
        old.resolve(f, VERTICES)
    proof = resolve_positive(f, VERTICES)
    assert 1e-9 < proof['lower_bound'] <= 1e-8
    assert proof['maximum_depth'] <= 32


def test_positive_determinant_below_margin_is_rejected():
    f = np.broadcast_to(np.diag([1.,1.,8e-10]), (4,3,3))
    with pytest.raises(ValueError, match='margin violation witness'):
        resolve_positive(f, VERTICES)


@pytest.mark.parametrize('budget', [dict(max_nodes=1), dict(max_depth=1)])
def test_unfinished_subdivision_never_grants_acceptance(budget):
    with pytest.raises(ValueError, match='unresolved'):
        resolve_positive(narrow_positive(1e-8), VERTICES, **budget)


def test_adaptive_policy_only_falls_back_on_unresolved_geometry():
    def unresolved(*args):
        raise ValueError('Whole-element positivity remains unresolved at maximum depth')

    child = SimpleNamespace(resolve=unresolved)
    restore = install_adaptive_proof(SimpleNamespace(margin=1e-9, shards=[(0,child)]))
    proof = child.resolve(narrow_positive(1e-8), VERTICES)
    assert proof['lower_bound'] > 1e-9 and proof['fallback_from_depth'] == 32
    restore()
    assert child.resolve is unresolved


def test_adaptive_policy_does_not_swallow_existing_witnesses():
    def rejected(*args):
        raise ValueError('Direct nonpositive determinant witness')

    child = SimpleNamespace(resolve=rejected)
    install_adaptive_proof(SimpleNamespace(margin=1e-9, shards=[(0,child)]))
    with pytest.raises(ValueError, match='Direct nonpositive determinant witness'):
        child.resolve(np.broadcast_to(np.eye(3), (4,3,3)), VERTICES)
