"""Joint linearized determinant constraints and global force-merit descent.

This is a globalization correction after the qualified Newton linear solve.
Every resulting trial still requires actual Armijo decrease and whole-cell proof.
The incoming merit vector is J.T @ (-R), the negative merit gradient.
"""
import numpy as np
from scipy.optimize import minimize

from cuda_determinant_direction import cofactor, shape_gradients


def point_constraint(displacement, factor, point, gradients, edges):
    shape = shape_gradients(point, gradients, edges)
    f = np.eye(3)+displacement.T @ shape
    return float(np.linalg.det(f)), factor*(shape @ cofactor(f).T).ravel()


def minimum_points(audit, matrices):
    values = np.linalg.det(np.einsum('pv,vij->pij', audit.lattice, matrices))
    points = [point.copy() for point in audit.lattice]

    def objective(point):
        f = np.einsum('v,vij->ij', point, matrices)
        return float(np.linalg.det(f)), np.einsum('ij,vij->v', cofactor(f), matrices)

    for start in audit.lattice[np.argsort(values)[:4]]:
        result = minimize(objective, start.copy(), jac=True, method='SLSQP', bounds=[(0.,1.)]*4,
                          constraints=[dict(type='eq', fun=lambda p:p.sum()-1,
                                            jac=lambda p:np.ones(4))],
                          options=dict(maxiter=80, ftol=1e-14))
        if (np.isfinite(result.x).all() and result.x.min() >= -1e-10
                and abs(result.x.sum()-1) <= 1e-8):
            point = np.maximum(result.x, 0)
            point /= point.sum()
            if all(np.linalg.norm(point-other) > 1e-8 for other in points):
                points.append(point)
    return points


def determinant_constraints(audit, element, current, old, direction):
    dofs, gl = audit.maps[element], audit.gl[element]
    vertex_gradients = np.array([shape_gradients(p, gl, audit.edges) for p in np.eye(4)])
    normals, values, stages = [], [], []
    for stage, displacement, factor in (
            ('endpoint', current[dofs], 1.), ('stress', .4*old[dofs]+.6*current[dofs], .6)):
        matrices = np.eye(3)+np.einsum('ki,vkj->vij', displacement, vertex_gradients)
        trial = matrices+factor*np.einsum('ki,vkj->vij', direction[dofs], vertex_gradients)
        points = minimum_points(audit, matrices)
        for point in minimum_points(audit, trial):
            if all(np.linalg.norm(point-other)>1e-8 for other in points):
                points.append(point)
        for point in points:
            value, normal = point_constraint(displacement, factor, point, gl, audit.edges)
            normals.append(normal)
            values.append(value)
            stages.append(stage)
    return np.array(normals), np.array(values), stages


def project_joint_direction(direction, negative_merit, dofs, normals, determinants, *, margin=1e-9):
    """Small convex QP with local increments and one global descent component."""
    assert margin == 1e-9
    direction, negative_merit = np.asarray(direction), np.asarray(negative_merit)
    assert direction.shape == negative_merit.shape
    assert np.isfinite(direction).all() and np.isfinite(negative_merit).all()
    ids = np.asarray(dofs).ravel()
    assert len(np.unique(ids)) == len(ids)
    frozen = direction == 0
    local = direction[ids]
    normals = np.asarray(normals).copy()
    assert np.isfinite(normals).all() and np.isfinite(determinants).all()
    normals[:, frozen[ids]] = 0
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 0
    normals, lengths = normals[keep], lengths[keep]
    determinants = np.asarray(determinants)[keep]
    bounds = -.5*np.maximum(determinants-margin, 0.)
    slopes = normals @ local
    detail = dict(kind='joint_determinant_merit_qp', applied=False, constraints=len(bounds),
                  minimum_point_determinant=float(np.min(determinants)) if len(determinants) else None,
                  violated_linear_constraints=int(np.count_nonzero(slopes < bounds)),
                  original_merit_slope=-float(negative_merit @ direction))
    if not np.any(slopes < bounds):
        return None, detail
    h = np.array(negative_merit, copy=True)
    h[frozen] = 0
    hlocal = h[ids].copy()
    h[ids] = 0
    outside_norm = float(np.linalg.norm(h))
    hnorm = float(np.hypot(np.linalg.norm(hlocal), outside_norm))
    scale = float(np.linalg.norm(direction))
    assert np.isfinite(hnorm) and np.isfinite(scale) and hnorm > 0 and scale > 0
    original_descent = float(negative_merit @ direction)
    outside_descent = original_descent-float(hlocal @ local)
    matrix = np.column_stack((normals/lengths[:,None], np.zeros(len(bounds))))
    force_row = np.r_[hlocal, outside_norm]/hnorm
    matrix = np.vstack((matrix, force_row))
    x0 = np.r_[local/scale, 0.]
    variable_bounds = [(0.,0.) if fixed else (None,None) for fixed in frozen[ids]]+[
        (None,None) if outside_norm > 0 else (0.,0.)]
    base_descent = max(original_descent, 1e-4*hnorm*scale)
    for fraction in (.5, .1, .01, 1e-4):
        target = fraction*base_descent
        rhs = np.r_[bounds/(lengths*scale), (target-outside_descent)/(hnorm*scale)]
        result = minimize(lambda x:.5*float((x-x0) @ (x-x0)), x0,
                          jac=lambda x:x-x0, method='SLSQP', bounds=variable_bounds,
                          constraints=[dict(type='ineq', fun=lambda x:matrix @ x-rhs,
                                            jac=lambda x:matrix)], options=dict(maxiter=200, ftol=1e-13))
        if not result.success or not np.isfinite(result.x).all():
            continue
        corrected = direction.copy()
        if outside_norm > 0:
            corrected += (scale*result.x[-1]/outside_norm)*h
        corrected[ids] = scale*result.x[:-1]
        corrected[frozen] = 0
        new_slopes = normals @ corrected[ids]
        tolerance = 1e-10*np.maximum.reduce((abs(bounds), abs(slopes), np.full(len(bounds), 1e-14)))
        descent = float(negative_merit @ corrected)
        if np.any(new_slopes < bounds-tolerance) or descent < target*(1-1e-10):
            continue
        if np.linalg.norm(corrected) > 4*scale:
            continue
        assert descent > 0 and np.isfinite(corrected).all()
        detail.update(applied=True, corrected_merit_slope=-descent, retained_descent_fraction=fraction,
                      minimum_linear_constraint_slack=float(np.min(new_slopes-bounds)),
                      correction_norm=float(np.linalg.norm(corrected-direction)),
                      outside_correction_norm=float(abs(scale*result.x[-1])), qp_iterations=int(result.nit))
        return corrected, detail
    raise RuntimeError('No joint determinant-safe force-descent direction found')


def correct_direction(audit, element, current, old, direction, negative_merit):
    assert audit.margin == 1e-9
    normals, values, stages = determinant_constraints(audit, element, current, old, direction)
    corrected, detail = project_joint_direction(direction, negative_merit, audit.maps[element], normals, values,
                                               margin=audit.margin)
    detail.update(element=element, endpoint_constraints=stages.count('endpoint'), stress_constraints=stages.count('stress'))
    return corrected, detail
