"""Active-set constrained Newton equations using distributed Jacobian responses.

For A=J+mu*D, solve A*d-N.T*lambda=b, N*d>=lower,
lambda>=0 with complementarity. A response solve A*z=N_i.T uses the
same assembled operator and preconditioner as the original Newton solve.
"""
import numpy as np
import time
import re


class ActiveSet:
    """Small Schur controller; response columns contain only the constrained dofs."""
    def __init__(self, normals, lower, raw_local):
        self.n = np.asarray(normals)
        self.lower = np.asarray(lower)
        self.raw = np.asarray(raw_local)
        assert self.n.ndim == 2 and self.n.shape == (len(self.lower), len(self.raw))
        assert np.isfinite(self.n).all() and np.isfinite(self.lower).all() and np.isfinite(self.raw).all()
        self.tolerance = 1e-10*max(float(np.max(abs(self.raw), initial=0)),
                                  float(np.max(abs(self.lower), initial=0)), 1e-14)
        assert np.all(self.lower <= self.tolerance), 'Zero direction must be a feasible start'
        self.current = np.zeros_like(self.raw)
        self.active = []
        self.responses = {}
        self.iterations = 0
        self.maximum_iterations = max(100,4*(len(self.lower)+len(self.raw)))

    def add_response(self, index, local_response):
        value = np.asarray(local_response)
        assert value.shape == self.raw.shape and np.isfinite(value).all()
        self.responses[index] = value.copy()

    def action(self):
        for _ in range(self.maximum_iterations):
            self.iterations += 1
            if self.iterations > self.maximum_iterations:
                return 'failed', 'Coupled constraint active-set iteration ceiling'
            for index in self.active:
                if index not in self.responses:
                    return 'response', index
            multipliers = np.empty(0)
            target = self.raw.copy()
            if self.active:
                columns = np.column_stack([self.responses[index] for index in self.active])
                schur = self.n[self.active] @ columns
                rhs = self.lower[self.active]-self.n[self.active] @ self.raw
                scale = float(np.max(abs(schur)))
                if not np.isfinite(scale) or scale == 0:
                    return 'failed', 'Degenerate Jacobian response in coupled constraint solve'
                multipliers, _, _, _ = np.linalg.lstsq(schur/scale, rhs/scale, rcond=1e-12)
                if not np.isfinite(multipliers).all():
                    return 'failed', 'Nonfinite coupled constraint multipliers'
                target += columns @ multipliers
                if np.max(abs(self.n[self.active] @ target-self.lower[self.active]), initial=0) > 10*self.tolerance:
                    return 'failed', 'Coupled active equalities failed residual qualification'
            values = self.n @ target-self.lower
            blockers = [i for i, value in enumerate(values) if value < -self.tolerance and i not in self.active]
            if blockers:
                old_values = self.n @ self.current-self.lower
                fractions = [(max(0., old_values[i])/(old_values[i]-values[i]), i) for i in blockers]
                fraction, index = min(fractions)
                self.current += min(1., max(0., fraction))*(target-self.current)
                self.active.append(index)
                continue
            dual_tolerance = 1e-10*max(float(np.max(abs(multipliers), initial=0)), 1e-14)
            if len(multipliers) and multipliers.min() < -dual_tolerance:
                self.current = target
                self.active.pop(int(multipliers.argmin()))
                continue
            if np.any(values < -10*self.tolerance):
                return 'failed', 'Coupled determinant constraints failed residual qualification'
            return 'done', dict(active=list(self.active), multipliers=multipliers.tolist(),
                                active_set_iterations=self.iterations,
                                minimum_constraint_slack=float(values.min()) if len(values) else 0.,
                                constraint_tolerance=self.tolerance)
        return 'failed', 'Coupled active-set action ceiling'


def build_constraints(audit, element, current, old, raw):
    return build_multi_constraints(audit, [element], current, old, raw)


def build_multi_constraints(audit, elements, current, old, raw):
    from refined_constrained_direction import determinant_constraints
    elements = sorted(set(elements))
    assert elements and audit.margin == 1e-9
    ids = np.unique(np.concatenate([audit.maps[element].ravel() for element in elements]))
    rows, bounds = [], []
    endpoint = stress = 0
    minimum = float('inf')
    for element in elements:
        normals, determinants, stages = determinant_constraints(audit, element, current, old, raw)
        lengths = np.linalg.norm(normals, axis=1)
        keep = lengths > 0
        assert np.isfinite(normals).all() and np.isfinite(determinants).all()
        embedded = np.zeros((int(keep.sum()),len(ids)))
        embedded[:,np.searchsorted(ids,audit.maps[element].ravel())] = normals[keep]/lengths[keep,None]
        rows.append(embedded)
        bounds.extend(-.5*np.maximum(determinants[keep]-audit.margin,0.)/lengths[keep])
        minimum = min(minimum,float(determinants.min()))
        endpoint += stages.count('endpoint')
        stress += stages.count('stress')
    return dict(ids=ids.tolist(),normals=np.vstack(rows).tolist(),lower=np.asarray(bounds).tolist(),
                minimum_point_determinant=minimum,endpoint_constraints=endpoint,stress_constraints=stress,
                elements=elements)


class GrowingConstraints:
    """Keep every discovered cell; each member contributes both stage constraints."""
    def __init__(self,audit,elements,old,max_elements=32):
        self.audit,self.old = audit,old
        self.elements = set(int(element) for element in elements)
        self.max_elements = max_elements
        assert self.elements and len(self.elements) <= max_elements

    def __call__(self,current,raw):
        return build_multi_constraints(self.audit,self.elements,current,self.old(),raw)

    def observe_failure(self,reason):
        match = re.search(r'\(stage=(endpoint|stress), element=(\d+)\)$',reason)
        if not match or not reason.startswith(('Whole-element positivity remains unresolved',
                'Direct nonpositive determinant witness','Direct determinant margin violation witness')):
            return None
        stage,element = match.group(1),int(match.group(2))
        assert 0 <= element < len(self.audit.maps)
        if element in self.elements:
            return None
        if len(self.elements) >= self.max_elements:
            raise RuntimeError('Coupled constraint element budget exhausted; no constraints dropped')
        self.elements.add(element)
        return dict(element=element,stage=stage,elements=sorted(self.elements),reason=reason)


def solve_coupled(A, ksp, b, delta, adapter, builder, current, raw_direction, *,
                  cuda_linear, budget, eta):
    """Replace delta in place only after qualifying the assembled coupled equations."""
    from mpi4py import MPI
    started = time.perf_counter()
    comm, rank = adapter.comm, adapter.rank
    spec = error = None
    if rank == 0:
        try:
            spec = builder(current, raw_direction)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)
    spec = comm.bcast(spec, root=0)
    ids, normals, lower = np.array(spec['ids']), np.array(spec['normals']), np.array(spec['lower'])
    assert len(np.unique(ids)) == len(ids) and normals.shape == (len(lower),len(ids))
    _, positions, columns = np.intersect1d(adapter.mapping, ids, return_indices=True)
    counts = np.zeros(len(ids), dtype=int)
    counts[columns] = 1
    assert np.all(comm.allreduce(counts, op=MPI.SUM) == 1)

    def local_values(vector):
        values = np.zeros(len(ids))
        values[columns] = vector.getArray(readonly=True)[positions]
        return comm.allreduce(values, op=MPI.SUM)

    def fill(vector, row):
        vector.set(0.)
        vector.getArray()[positions] = normals[row, columns]

    controller = ActiveSet(normals, lower, raw_direction[ids]) if rank == 0 else None
    rhs, residual, effective = b.duplicate(), b.duplicate(), b.copy()
    responses = {}
    attempts = []
    gpu_solution = gpu_rhs = None
    if cuda_linear:
        operator, _ = ksp.getOperators()
        gpu_solution, gpu_rhs = operator.createVecs()
    try:
        while True:
            budget()
            action = None
            if rank == 0:
                try:
                    action = controller.action()
                except Exception as exc:
                    action = ('failed', f'{type(exc).__name__}: {exc}')
            kind, payload = comm.bcast(action, root=0)
            if kind == 'failed':
                raise RuntimeError(payload)
            if kind == 'done':
                detail = payload
                break
            index = payload
            fill(rhs, index)
            solution = b.duplicate()
            responses[index] = solution
            rhs_norm = float(rhs.norm())
            response_rtol = min(1e-7, eta*.1)
            limit = max(1.1e-12, 1.1*response_rtol*rhs_norm)
            qualified = False
            ksp.getPC().setReusePreconditioner(True)
            for reduction in (1., .1, .01):
                budget()
                if rank == 0:
                    print('COUPLED RESPONSE constraint', index, 'rtol', response_rtol*reduction, flush=True)
                ksp.setTolerances(rtol=response_rtol*reduction, atol=min(1e-12,limit*.05), max_it=3000)
                solution.set(0.)
                if cuda_linear:
                    rhs.copy(gpu_rhs)
                    gpu_solution.set(0.)
                    ksp.solve(gpu_rhs, gpu_solution)
                    gpu_solution.copy(solution)
                else:
                    ksp.solve(rhs, solution)
                A.mult(solution, residual)
                residual.axpy(-1., rhs)
                true = float(residual.norm())
                reason = int(ksp.getConvergedReason())
                qualified = bool(reason > 0 and np.isfinite(true) and true <= limit)
                attempts.append(dict(constraint=index, KSP_reason=reason,
                    KSP_iterations=int(ksp.getIterationNumber()), rtol=response_rtol*reduction,
                    true_residual_N=true if np.isfinite(true) else None, limit_N=limit, qualified=qualified))
                if qualified or reason <= 0:
                    break
            if not qualified:
                raise RuntimeError('Jacobian constraint response failed true residual qualification')
            values = local_values(solution)
            if rank == 0:
                controller.add_response(index, values)
        for index, multiplier in zip(detail['active'], detail['multipliers'], strict=True):
            delta.axpy(multiplier, responses[index])
            fill(rhs, index)
            effective.axpy(multiplier, rhs)
        A.mult(delta, residual)
        residual.axpy(-1., effective)
        true = float(residual.norm())
        # Keep the original force-based accuracy requirement, even when the
        # constraint forces make the augmented right-hand side larger.
        limit = max(1.1e-12, 1.1*eta*float(b.norm()))
        values = local_values(delta)
        assert np.isfinite(true) and true <= limit, 'Coupled stationarity residual failed qualification'
        assert np.all(normals @ values >= lower-10*detail['constraint_tolerance']), 'Coupled constraint residual failed qualification'
        detail.update(kind='assembled_jacobian_active_set', response_solves=len(responses),
                      response_attempts=attempts, stationarity_residual_N=true, stationarity_limit_N=limit,
                      augmented_rhs_norm_N=float(effective.norm()),
                      minimum_point_determinant=spec['minimum_point_determinant'],
                      endpoint_constraints=spec['endpoint_constraints'], stress_constraints=spec['stress_constraints'],
                      elements=spec.get('elements',[]),
                      wall_seconds=time.perf_counter()-started)
        return detail
    finally:
        ksp.getPC().setReusePreconditioner(False)
        for vector in (rhs, residual, effective, *responses.values(), gpu_solution, gpu_rhs):
            if vector is not None:
                vector.destroy()
