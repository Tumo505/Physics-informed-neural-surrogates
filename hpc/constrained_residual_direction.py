"""Matrix-free constrained Gauss-Newton direction in diagonally scaled coordinates.

Minimize .5*||R+J*d||**2 + .5*mu*||S*d||**2 subject to N*d>=lower.
H=(J/S).T*(J/S)+mu*I is applied without assembling the normal matrix.
All candidate steps still require the original nonlinear acceptance checks.
"""
import time
import numpy as np
from coupled_newton_direction import solve_coupled


class NormalOperator:
    def __init__(self, jacobian, scale, mu, template, cuda):
        self.jacobian, self.scale, self.mu, self.cuda = jacobian, scale, mu, cuda
        self.scaled = template.duplicate()
        self.input, self.product = jacobian.createVecs()
        self.transpose = self.input.duplicate()

    def mult(self, matrix, x, y):
        self.scaled.getArray()[:] = x.getArray(readonly=True)/self.scale
        self.scaled.copy(self.input)
        self.jacobian.mult(self.input,self.product)
        self.jacobian.multTranspose(self.product,self.transpose)
        self.transpose.copy(y)
        y.getArray()[:] /= self.scale
        y.axpy(self.mu,x)

    def gradient_rhs(self, b, output):
        b.copy(self.product)
        self.jacobian.multTranspose(self.product,self.transpose)
        self.transpose.copy(output)
        output.getArray()[:] /= self.scale

    def close(self):
        for vector in (self.scaled,self.input,self.product,self.transpose):vector.destroy()


def residual_direction(J, b, delta, adapter, builder, current, scale, mu, *, cuda, budget, eta):
    from petsc4py import PETSc
    from mpi4py import MPI
    started=time.perf_counter()
    assert np.isfinite(mu) and mu > 0
    scaling=np.array(scale.getArray(readonly=True),copy=True)
    assert np.isfinite(scaling).all() and np.all(scaling>0)
    jacobian=J.copy()
    if cuda:jacobian.convert('aijcusparse')
    context=NormalOperator(jacobian,scaling,mu,b,cuda)
    H=solver=rhs=raw=residual=None
    try:
        H=PETSc.Mat().createPython((b.getSizes(),b.getSizes()),context=context,comm=J.comm)
        H.setUp()
        solver=PETSc.KSP().create(J.comm)
        solver.setType('cg');solver.getPC().setType('none')
        solver.setOperators(H)
        solver.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        def monitor(k,it,r):
            budget()
            if adapter.rank==0 and it % 100 == 0:
                print('MERIT LINEAR iteration',it,'normal_residual',r,flush=True)
        solver.setMonitor(monitor)
        solver.setTolerances(rtol=min(eta,1e-7),atol=1e-14,max_it=3000)
        rhs=b.duplicate();raw=b.duplicate();residual=b.duplicate()
        context.gradient_rhs(b,rhs)
        if adapter.rank==0:
            print('MERIT NORMAL SOLVE shift',mu,'maximum_iterations',3000,flush=True)
        raw.set(0.);budget();solver.solve(rhs,raw)
        H.mult(raw,residual);residual.axpy(-1.,rhs)
        raw_error=float(residual.norm())
        limit=max(1.1e-12,1.1*eta*float(rhs.norm()))
        if solver.getConvergedReason()<=0 or not np.isfinite(raw_error) or raw_error>limit:
            raise RuntimeError('Residual-minimizing direction failed true normal-equation qualification')
        raw_iterations=int(solver.getIterationNumber())
        # PETSc ownership and the canonical P2 map have different orderings.
        physical=raw.getArray(readonly=True)/scaling
        arrays=adapter.comm.gather(physical.copy(),root=0)
        canonical=None
        if adapter.rank==0:
            canonical=np.empty(adapter.ndof)
            for ids,values in zip(adapter.parts,arrays,strict=True):canonical[ids]=values
        canonical=adapter.comm.bcast(canonical,root=0)
        canonical_scale=np.zeros(adapter.ndof)
        canonical_scale[adapter.mapping]=scaling
        canonical_scale=adapter.comm.allreduce(canonical_scale,op=MPI.SUM)
        def scaled_constraints(current, ignored):
            spec=dict(builder(current,canonical))
            normals=np.array(spec['normals'])/canonical_scale[np.array(spec['ids'])]
            lengths=np.linalg.norm(normals,axis=1)
            assert np.isfinite(lengths).all() and np.all(lengths>0)
            spec['normals']=(normals/lengths[:,None]).tolist()
            spec['lower']=(np.array(spec['lower'])/lengths).tolist()
            return spec
        # solve_coupled's controller expects the raw direction in H coordinates.
        raw.copy(delta)
        detail=solve_coupled(H,solver,rhs,delta,adapter,scaled_constraints,current,
                             canonical*canonical_scale,cuda_linear=False,budget=budget,eta=eta)
        delta.getArray()[:] /= scaling
        J.mult(delta,residual)
        slope=-float(b.dot(residual))
        # Use global vector norms for distributed predictions.
        residual.axpy(-1.,b)
        predicted=.5*(float(b.norm())**2-float(residual.norm())**2)
        detail.update(kind='constrained_residual_minimization',raw_normal_iterations=raw_iterations,
                      raw_normal_residual=raw_error,normal_residual_limit=limit,
                      merit_slope=slope,predicted_force_merit_reduction=predicted,
                      wall_seconds=time.perf_counter()-started)
        if not np.isfinite(slope) or slope>=0:
            raise RuntimeError('Constrained residual model produced no negative merit slope')
        if adapter.rank==0:
            print('MERIT DIRECTION elements',detail['elements'],'slope',slope,
                  'normal_iterations',raw_iterations,'seconds',detail['wall_seconds'],flush=True)
        return detail
    finally:
        if solver is not None:solver.destroy()
        if H is not None:H.destroy()
        for vector in (rhs,raw,residual):
            if vector is not None:vector.destroy()
        context.close();jacobian.destroy()
