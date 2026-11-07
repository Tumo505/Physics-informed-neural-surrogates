"""Experimental bulk determinant path for a fixed-field benchmark."""
from types import MethodType

import numpy as np

from lv_surrogate.physics.threaded_bernstein import ThreadedBernsteinAudit


def bulk_coefficients(matrices, transform, margin):
    """Use cofactor expansion away from the margin, with legacy fallback.

    The roundoff screen uses absolute products before cancellation. It is a
    conservative routing heuristic for this float64 comparison, not an interval
    arithmetic certificate. Ambiguous elements keep the original calculation.
    """
    a, b, c = (matrices[..., 0, i] for i in range(3))
    d, e, f = (matrices[..., 1, i] for i in range(3))
    g, h, i = (matrices[..., 2, j] for j in range(3))
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        ei, fh, di, fg, dh, eg = e*i, f*h, d*i, f*g, d*h, e*g
        values = a*(ei-fh) - b*(di-fg) + c*(dh-eg)
        magnitude = abs(a)*(abs(ei)+abs(fh)) + abs(b)*(abs(di)+abs(fg)) + abs(c)*(abs(dh)+abs(eg))
        coefficients = values @ transform.T
        guard = 1e-12 * (magnitude @ abs(transform.T)) + 1e-12
        fallback = np.any(~np.isfinite(coefficients) | ~np.isfinite(guard)
                          | (coefficients <= margin + guard), axis=1)
        # Large entries can amplify an underflowed intermediate product.
        fallback |= np.any(np.abs(matrices) > 1e80, axis=(1, 2, 3))
    if fallback.any():
        coefficients[fallback] = np.linalg.det(matrices[fallback]) @ transform.T
    return coefficients


def cofactor_field(self, u):
    minimum = parent = float('inf')
    refined = []
    error = 0.
    # Preserve the original element traversal, proof report and subdivision.
    for start in range(0, len(self.maps), self.chunk):
        stop = min(start + self.chunk, len(self.maps))
        gradients = self.gl[start:stop]
        grad = np.stack([
            np.concatenate(((4*b[None, :, None]-1)*gradients,
                            np.stack([4*(b[i]*gradients[:, j]+b[j]*gradients[:, i])
                                      for i, j in self.edges], axis=1)), axis=1)
            for b in np.eye(4)
        ], axis=1)
        deformation = np.eye(3) + np.einsum('eki,evkj->evij', u[self.maps[start:stop]], grad)
        matrices = np.einsum('qv,evij->eqij', self.lattice, deformation)
        coefficients = bulk_coefficients(matrices, self.transform, self.margin)
        assert np.isfinite(coefficients).all()
        lower = coefficients.min(axis=1)
        parent = min(parent, float(lower.min()))
        positive = lower[lower > self.margin]
        if len(positive):
            minimum = min(minimum, float(positive.min()))
        for index in np.flatnonzero(lower <= self.margin):
            proof = self.resolve(deformation[index], self.X[start + index])
            minimum = min(minimum, proof['lower_bound'])
            refined.append(dict(element=int(start + index), classification='positive_bound', **proof))
        direct = np.linalg.det(np.einsum('qv,vij->qij', self.check, deformation[0]))
        reconstructed = self.basis(self.check) @ coefficients[0]
        np.testing.assert_allclose(direct, reconstructed, rtol=1e-11, atol=1e-11)
        error = max(error, float(np.max(abs(direct-reconstructed))))
    assert minimum > self.margin
    return dict(minimum_Bernstein_bound=minimum, minimum_parent_Bernstein_bound=parent,
                original_unresolved_elements=len(refined), refined_elements=refined,
                polynomial_reconstruction_max_abs_error=error, passed=True)


class CofactorBernsteinAudit(ThreadedBernsteinAudit):
    def __init__(self, arrays, **kwargs):
        super().__init__(arrays, **kwargs)
        for _, child in self.shards:
            child.field = MethodType(cofactor_field, child)
