"""Local first-order determinant correction for the state-555 search direction."""
import numpy as np
from scipy.optimize import minimize


def shape_gradients(barycentric, linear_gradients, edges):
    vertices = (4 * barycentric[:, None] - 1) * linear_gradients
    edge_nodes = [4 * (barycentric[i] * linear_gradients[j]
                       + barycentric[j] * linear_gradients[i]) for i, j in edges]
    return np.vstack((vertices, edge_nodes))


def cofactor(matrix):
    return np.cross(matrix[[1, 2, 0]], matrix[[2, 0, 1]])


def correct_direction(audit, element, current, direction, merit_gradient):
    """Protect a sampled endpoint determinant while preserving the merit slope."""
    dofs = audit.maps[element]
    gradients = audit.gl[element]
    displacement = current[dofs]
    vertex_matrices = np.array([
        np.eye(3) + displacement.T @ shape_gradients(vertex, gradients, audit.edges)
        for vertex in np.eye(4)
    ])

    def value_and_gradient(point):
        deformation = np.einsum('i,ijk->jk', point, vertex_matrices)
        determinant = np.linalg.det(deformation)
        gradient = np.einsum('ij,kij->k', cofactor(deformation), vertex_matrices)
        return determinant, gradient

    lattice_values = np.linalg.det(np.einsum('bi,ijk->bjk', audit.lattice, vertex_matrices))
    point = audit.lattice[np.argmin(lattice_values)]
    candidate = minimize(
        value_and_gradient, point, jac=True, method='SLSQP', bounds=[(0., 1.)] * 4,
        constraints=[dict(type='eq', fun=lambda x: x.sum() - 1,
                          jac=lambda x: np.ones_like(x))],
        options=dict(maxiter=40, ftol=1e-12),
    )
    if np.isfinite(candidate.x).all() and np.all(candidate.x >= 0) and candidate.x.sum() > 0:
        refined = candidate.x / candidate.x.sum()
        if value_and_gradient(refined)[0] < value_and_gradient(point)[0]:
            point = refined

    deformation = np.einsum('i,ijk->jk', point, vertex_matrices)
    determinant = float(np.linalg.det(deformation))
    sensitivity = (shape_gradients(point, gradients, audit.edges) @ cofactor(deformation).T).ravel()
    local_direction = direction[dofs].ravel()
    local_merit = merit_gradient[dofs].ravel().copy()
    # Preserve exactly zero components, including homogeneous constrained increments.
    frozen = local_direction == 0
    sensitivity[frozen] = 0
    local_merit[frozen] = 0
    slope = float(sensitivity @ local_direction)
    record = dict(element=element, determinant=determinant, barycentric=point.tolist(),
                  original_determinant_slope=slope, applied=False)
    if slope >= -.5 * max(0., determinant - audit.margin):
        return None, record

    normal = sensitivity.copy()
    merit_norm = float(local_merit @ local_merit)
    if merit_norm > 0:
        normal -= local_merit * ((local_merit @ normal) / merit_norm)
    denominator = float(sensitivity @ normal)
    if denominator <= np.finfo(float).eps * float(sensitivity @ sensitivity):
        record['reason'] = 'no independent determinant correction'
        return None, record
    correction = (-slope / denominator) * normal
    corrected = direction.copy()
    corrected[dofs] = (local_direction + correction).reshape(dofs.shape)
    record.update(applied=True, corrected_determinant_slope=float(sensitivity @ corrected[dofs].ravel()),
                  correction_norm=float(np.linalg.norm(correction)))
    return corrected, record
