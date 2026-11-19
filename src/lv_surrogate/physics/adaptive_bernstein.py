"""Determinant-directed cubic Bernstein subdivision with outward-rounded bounds.

The enclosures apply to the exact polynomial of the supplied float64 vertex
gradients. They do not enclose errors incurred when assembling those gradients.
"""
import itertools
import math

import numpy as np

MULTI = tuple(a for a in itertools.product(range(4), repeat=4) if sum(a) == 3)
INDEX = {a: i for i, a in enumerate(MULTI)}
EDGES = tuple(itertools.combinations(range(4), 2))
TRIPLES = np.array(list(itertools.product(range(4), repeat=3)))
GROUPS = [np.flatnonzero([tuple(np.bincount(t, minlength=4)) == a for t in TRIPLES]) for a in MULTI]
VERTEX_COEFFICIENTS = [INDEX[tuple(3 if i == j else 0 for i in range(4))] for j in range(4)]


def down(value):
    return np.nextafter(value, -np.inf)


def up(value):
    return np.nextafter(value, np.inf)


def multiply(a, b):
    products = np.stack((a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]))
    return down(products.min(axis=0)), up(products.max(axis=0))


def coefficient_bounds(deformation):
    """Use the symmetric determinant polar form, avoiding interpolation inversion."""
    f = np.asarray(deformation, dtype=np.float64)
    assert f.shape == (4,3,3) and np.isfinite(f).all()
    matrices = np.stack([f[TRIPLES[:, j], :, j] for j in range(3)], axis=2)
    low = np.zeros(64)
    high = np.zeros(64)
    for permutation in itertools.permutations(range(3)):
        sign = (-1)**sum(permutation[i] > permutation[j] for i in range(3) for j in range(i+1,3))
        term = (matrices[:,0,permutation[0]], matrices[:,0,permutation[0]])
        for row in (1,2):
            value = matrices[:,row,permutation[row]]
            term = multiply(term, (value, value))
        left, right = term if sign == 1 else (-term[1], -term[0])
        low, high = down(low+left), up(high+right)
    coefficients_low, coefficients_high = [], []
    for group in GROUPS:
        lo = hi = np.float64(0.)
        for index in group:
            lo, hi = down(lo+low[index]), up(hi+high[index])
        coefficients_low.append(down(lo/len(group)))
        coefficients_high.append(up(hi/len(group)))
    bounds = np.array(coefficients_low), np.array(coefficients_high)
    if not all(np.isfinite(value).all() for value in bounds):
        raise ValueError('Nonfinite determinant enclosure')
    return bounds


def restrictions():
    indices = np.zeros((6,2,20,4), dtype=int)
    weights = np.zeros((6,2,20,4))
    for edge_index, (a,b) in enumerate(EDGES):
        for side, (replace, other) in enumerate(((a,b),(b,a))):
            for index, multi in enumerate(MULTI):
                degree = multi[replace]
                for k in range(degree+1):
                    old = list(multi)
                    old[replace] = k
                    old[other] += degree-k
                    indices[edge_index,side,index,k] = INDEX[tuple(old)]
                    weights[edge_index,side,index,k] = math.comb(degree,k)/2**degree
    return indices, weights


RESTRICTION_INDICES, RESTRICTION_WEIGHTS = restrictions()


def child_bounds(low, high):
    """Restrict the cubic to both midpoint children for each of six edges."""
    shape = RESTRICTION_WEIGHTS.shape[:-1]
    lo, hi = np.zeros(shape), np.zeros(shape)
    for index in range(4):
        ids = RESTRICTION_INDICES[...,index]
        weights = RESTRICTION_WEIGHTS[...,index]
        lo = down(lo+down(weights*low[ids]))
        hi = up(hi+up(weights*high[ids]))
    return lo, hi


def resolve_positive(deformation, vertices, *, margin=1e-9, max_nodes=20000, max_depth=64):
    """Prove all leaves positive; unresolved work never grants acceptance."""
    assert margin == 1e-9 and max_nodes > 0 and max_depth > 0
    vertices = np.asarray(vertices, dtype=float)
    assert vertices.shape == (4,3) and np.isfinite(vertices).all()
    low, high = coefficient_bounds(deformation)
    stack = [(low, high, np.eye(4), 0)]
    minimum = float('inf')
    leaves = nodes = deepest = 0
    while stack:
        low, high, barycentric, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise ValueError('Whole-element positivity remains unresolved at adaptive node budget')
        deepest = max(deepest, depth)
        if np.any(high[VERTEX_COEFFICIENTS] <= 0):
            raise ValueError('Direct nonpositive determinant witness')
        if np.any(high[VERTEX_COEFFICIENTS] <= margin):
            raise ValueError('Direct determinant margin violation witness')
        lower = float(low.min())
        if lower > margin:
            minimum = min(minimum, lower)
            leaves += 1
            continue
        if depth == max_depth:
            raise ValueError('Whole-element positivity remains unresolved at maximum depth')
        lows, highs = child_bounds(low, high)
        physical = barycentric @ vertices
        # Split where the determinant bound improves, rather than spending the
        # same depth resolving spatial directions with little determinant variation.
        edge = max(range(6), key=lambda i: (
            float(lows[i].min()), -int(np.count_nonzero(lows[i] <= margin)),
            float(np.sum((physical[EDGES[i][0]]-physical[EDGES[i][1]])**2))))
        a,b = EDGES[edge]
        midpoint = (barycentric[a]+barycentric[b])/2
        one, two = barycentric.copy(), barycentric.copy()
        one[a], two[b] = midpoint, midpoint
        children = [(lows[edge,0], highs[edge,0], one, depth+1),
                    (lows[edge,1], highs[edge,1], two, depth+1)]
        # Inspect the less promising child first, finding witnesses promptly.
        children.sort(key=lambda entry: float(entry[0].min()), reverse=True)
        stack.extend(children)
    return dict(lower_bound=minimum, leaf_tetrahedra=leaves, maximum_depth=deepest,
                nodes_visited=nodes, algorithm='determinant_adaptive_interval_bernstein',
                arithmetic='outward-rounded float64 coefficient intervals', margin=margin)
