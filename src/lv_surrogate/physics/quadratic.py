"""Serial P2 tetrahedral kinematics at the source quadrature points and vertices."""
import itertools
import numpy as np


class QuadraticKinematics:
    def __init__(self, space, qpoints, qweights):
        mesh = space.mesh()
        self.cells = mesh.cells()
        X = mesh.coordinates()[self.cells]
        D = (X[:, 1:] - X[:, :1]).transpose(0, 2, 1)
        inverse = np.linalg.inv(D)
        self.measure = abs(np.linalg.det(D))
        self.volume = self.measure.sum() / 6
        self.grad_lambda = np.concatenate((-inverse.sum(1, keepdims=True), inverse), axis=1)
        self.edges = list(itertools.combinations(range(4), 2))
        canonical = np.concatenate((np.eye(4), np.array([
            (np.eye(4)[i] + np.eye(4)[j]) / 2 for i, j in self.edges])))
        coords = space.tabulate_dof_coordinates().reshape(-1, 3)
        maps = []
        for component in range(3):
            local = np.stack([space.sub(component).dofmap().cell_dofs(i)
                              for i in range(len(self.cells))])
            delta = coords[local] - X[:, :1]
            xyz = np.einsum('ekj,eij->eki', delta, inverse)
            bary = np.concatenate((1 - xyz.sum(-1, keepdims=True), xyz), axis=-1)
            distance = abs(bary[:, :, None, :] - canonical[None, None, :, :]).sum(-1)
            order = distance.argmin(axis=1)
            assert np.take_along_axis(distance, order[:, None, :], axis=1).max() < 1e-9
            maps.append(np.take_along_axis(local, order, axis=1))
        self.maps = np.stack(maps, axis=-1)
        self.quad_bary = np.c_[1 - np.asarray(qpoints).sum(1), qpoints]
        self.weights = np.asarray(qweights)
        np.testing.assert_allclose(self.weights.sum(), 1 / 6, rtol=1e-12)
        self.sample_bary = np.concatenate((self.quad_bary, np.eye(4)))

    def determinants(self, coefficients, bary):
        values = np.asarray(coefficients)[self.maps]
        gradients = np.concatenate(((4 * bary[None, :, None] - 1) * self.grad_lambda,
                                    np.stack([4 * (bary[i] * self.grad_lambda[:, j] +
                                                   bary[j] * self.grad_lambda[:, i])
                                              for i, j in self.edges], axis=1)), axis=1)
        F = np.eye(3) + np.einsum('eki,ekj->eij', values, gradients)
        return np.linalg.det(F)

    def audit(self, coefficients):
        minimum = float('inf')
        integral = 0.
        for i, bary in enumerate(self.sample_bary):
            J = self.determinants(coefficients, bary)
            minimum = min(minimum, float(J.min()))
            if i < len(self.weights):
                integral += self.weights[i] * float(self.measure @ J)
        return minimum, integral
