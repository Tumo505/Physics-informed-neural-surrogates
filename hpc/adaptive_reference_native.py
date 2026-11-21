"""Native adapter: unchanged mechanics, compiled quadrature, variable dt and exact histories."""
from pathlib import Path
import time
import numpy as np
from adaptive_reference_control import Snapshot,TrialRejected,BudgetExpired,quality_error
from registry import atomic


class Native:
    def __init__(self,root,run,plan,budget):
        from lv_surrogate.physics.parallel_fem import ParallelP2
        from lv_surrogate.physics.reference_fem import df
        from cofactor_proof import CofactorBernsteinAudit
        from production_cuda_native import use_positivity_depth,add_positivity_diagnostics
        from refined_positivity_policy import extend_unresolved_depth,install_adaptive_proof
        from itertools import combinations
        self.root,self.run,self.plan,self.deadline=root,run,plan,budget
        self.models={}
        for degree in plan['policy']['quadrature_ladder']:
            cfg=dict(plan['config'],quadrature_degree=degree,dt_s=plan['policy']['maximum_dt'])
            model=ParallelP2(root,plan['source_reference'],cfg)
            problem=model.problem
            assert float(problem.parameters['alpha_f'])==.4 and float(problem.parameters['alpha_m'])==.2
            # A mutable coefficient is essential: replacing a Python float after
            # form construction would leave a stale dt embedded in the UFL form.
            problem.parameters['dt']=df.Constant(cfg['dt_s']);problem._init_forms()
            parameters=dict(quadrature_degree=degree,cpp_optimize=True)
            problem._problem._F=df.Form(problem._problem._F,form_compiler_parameters=parameters)
            problem._problem._J=df.Form(problem._problem._J,form_compiler_parameters=parameters)
            original=problem._problem.J
            nullspace=problem._quadratic_solver_objects[-1]
            def near(A,x,original=original,nullspace=nullspace):
                original(A,x);df.as_backend_type(A).mat().setNearNullSpace(nullspace)
            problem._problem.J=near
            problem.solver._snes.setUseEW(False)
            problem.solver._solver.parameters.update(dict(method='newtonls',maximum_iterations=0,
                absolute_tolerance=1e-8,relative_tolerance=0.))
            self.models[degree]=model
        self.model=self.models[max(plan['policy']['quadrature_ladder'])]
        self.comm,self.rank=self.model.comm,self.model.rank
        assert self.comm.size==12
        from launch_refined_production import GPU
        loaded={Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines() if '/libpetsc.so' in line}
        assert loaded=={(GPU/'lib/libpetsc.so').resolve()}
        self.audit=None
        if self.rank==0:
            with np.load(root/plan['inputs'],allow_pickle=False) as inputs:
                self.audit=CofactorBernsteinAudit(inputs,workers=8)
            use_positivity_depth(self.audit,20,24);extend_unresolved_depth(self.audit,32)
            install_adaptive_proof(self.audit);add_positivity_diagnostics(self.audit)
        self.patch=np.load(root/plan['patch'],allow_pickle=False)
        from audit_reference_cell_555 import cartesian_deformation
        self.cartesian=cartesian_deformation
        xyz=self.patch['quadrature']
        points=[*np.column_stack((1-xyz.sum(axis=1),xyz)),*np.eye(4)]
        for i,j in combinations(range(4),2):
            for weight in np.linspace(0,1,17):
                p=np.zeros(4);p[i]=weight;p[j]=1-weight;points.append(p)
        self.points=np.array(points)
        self.derivatives_checked=set()
        self.trials=self.root_call(lambda:max([int(p.name.split('-')[0]) for p in (run/'trials').glob('*') if p.is_dir()],default=0))

    def root_call(self,callback):
        value=error=None
        if self.rank==0:
            try:value=callback()
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        error=self.comm.bcast(error,root=0)
        if error:raise TrialRejected(error)
        return self.comm.bcast(value,root=0)

    def pressure(self,t):
        if not hasattr(self,'forcing'):
            self.forcing=np.load(self.run/'forcing.npz',allow_pickle=False)
        quantum=float(self.forcing['quantum'])
        index=int(round(t/quantum))
        if not 0<=index<len(self.forcing['pressure']) or abs(t-index*quantum)>1e-12:
            raise TrialRejected('Pressure evaluation outside frozen dyadic forcing lattice')
        return float(self.forcing['pressure'][index])

    def assign(self,model,state):
        for name,value in (('u',state.u),('u_old',state.u),('v_old',state.v),('a_old',state.a)):
            model.assign(name,value)

    def local_fields(self,state):
        def evaluate():
            rows=[]
            for vertices,mapping in zip(self.patch['vertices'],self.patch['maps'],strict=True):
                rows.append(self.cartesian(vertices,state.u[mapping],self.points@vertices))
            return np.array(rows)
        return self.root_call(evaluate)

    def stage_quality(self,old,new):
        endpoint=self.local_fields(new)
        stage=Snapshot(new.time,.4*old.u+.6*new.u,new.v,new.a)
        stress=self.local_fields(stage)
        return quality_error(endpoint,stress,float(self.model.problem.material.parameters['b']),self.plan['policy'])

    def initial(self):
        """Consistent a0 from the continuous mass balance; no late-history reset."""
        from lv_surrogate.physics.reference_fem import df,residual,ufl
        from petsc4py import PETSc
        problem=self.model.problem
        zero=np.zeros(self.model.ndof)
        state=Snapshot(0.,zero.copy(),zero.copy(),zero.copy());self.assign(self.model,state)
        problem.parameters['p'].assign(self.pressure(0.))
        R=residual(problem);rhs=df.as_backend_type(R).vec().copy();rhs.scale(-1.)
        mass=df.assemble(problem.parameters['rho']*ufl.inner(ufl.TrialFunction(problem.u_space),
            ufl.TestFunction(problem.u_space))*ufl.dx,
            form_compiler_parameters=dict(quadrature_degree=max(self.plan['policy']['quadrature_ladder'])))
        M=df.as_backend_type(mass).mat();a=rhs.duplicate();check=rhs.duplicate()
        ksp=PETSc.KSP().create(M.comm);ksp.setType('cg');ksp.getPC().setType('jacobi')
        ksp.setOperators(M);ksp.setTolerances(rtol=1e-11,atol=1e-13,max_it=1000)
        try:
            ksp.solve(rhs,a);M.mult(a,check);check.axpy(-1.,rhs)
            error=float(check.norm());limit=max(1e-12,1e-10*float(rhs.norm()))
            assert ksp.getConvergedReason()>0 and error<=limit,(error,limit)
            parts=self.comm.gather(a.getArray(readonly=True).copy(),root=0)
            acceleration=None
            if self.rank==0:
                acceleration=np.empty(self.model.ndof)
                for ids,values in zip(self.model.parts,parts,strict=True):acceleration[ids]=values
            acceleration=self.comm.bcast(acceleration,root=0)
            return Snapshot(0.,zero.copy(),zero.copy(),acceleration)
        finally:
            ksp.destroy()
            for vector in (rhs,a,check):vector.destroy()

    def advance(self,old,dt,degree,label):
        from lv_surrogate.physics.reference_fem import residual
        from lv_surrogate.physics.admissible_newton_gentle_shrink import solve
        self.deadline();model=self.models[degree];problem=model.problem;self.assign(model,old)
        problem.parameters['dt'].assign(dt)
        pressure=self.pressure(old.time+(1-.4)*dt);problem.parameters['p'].assign(pressure)
        self.trials+=1;trace={}
        if self.rank==0:print('ADAPTIVE TRIAL quadrature',degree,'time',old.time,'dt',dt,label,flush=True)
        trial=self.run/'trials'/f'{self.trials:06d}-{label}'
        self.root_call(lambda:trial.mkdir(parents=True))
        # Predictors can be discarded. Every solve and update uses the exact same
        # accepted input histories; no failed trial calls _update_fields().
        predictor=old.u+dt*old.v+.5*dt*dt*old.a
        error=None
        if self.rank==0:
            try:self.audit.pair(predictor,old.u)
            except Exception:error=True
        if self.comm.bcast(error,root=0):predictor=old.u.copy()
        model.assign('u',predictor)
        options=dict(self.plan['newton_options'],initial_regularization=0.,minimum_regularization=0.,
                     wall_budget_seconds=float('inf'),maximum_custom_newton_iterations=30)
        if degree not in self.derivatives_checked:
            from solver_qualification import derivative_gate
            elements=[int(value) for value in self.patch['cells'] if int(value) in (813578,813580)]
            assert elements
            derivative_gate(self.run,trial,model,dict(u_old=old.u,v_old=old.v,a_old=old.a),self.audit,
                elements,predictor,self.base_digest,self.deadline)
            self.derivatives_checked.add(degree)
        started=time.perf_counter()
        def persist():
            if self.rank==0:atomic(trial/'trace.json',trace)
        try:
            solve(problem,model,dict(u_old=old.u,v_old=old.v,a_old=old.a),self.audit,options,trace,persist,
                policy='fixed_shift',cuda_linear=True,host_preconditioner=True,directional_armijo=True,
                late_forcing_floor=1e-5,check_deadline=self.deadline)
            assert trace['custom_force_N']<1e-9
            before=self.comm.bcast(model.gather('u'),root=0)
            problem.solver.solve()
            after=self.comm.bcast(model.gather('u'),root=0)
            assert problem.solver._snes.getConvergedReason()>0 and problem.solver._snes.getIterationNumber()==0
            np.testing.assert_array_equal(before,after)
            force=float(residual(problem).norm('l2'));assert force<1e-8
            proof=self.root_call(lambda:self.audit.pair(after,old.u))
            problem._update_fields()
            u,v,a=[self.comm.bcast(model.gather(name),root=0) for name in ('u_old','v_old','a_old')]
            beta,gamma=float(problem._beta),float(problem._gamma)
            expected_a=(u-old.u-dt*old.v-(.5-beta)*dt*dt*old.a)/(beta*dt*dt)
            expected_v=old.v+(1-gamma)*dt*old.a+gamma*dt*expected_a
            np.testing.assert_allclose(a,expected_a,rtol=1e-7,atol=1e-7)
            np.testing.assert_allclose(v,expected_v,rtol=1e-7,atol=1e-10)
            result=Snapshot(old.time+dt,u,v,a)
            trace.update(dt_s=dt,pressure_Pa=pressure,quadrature_degree=degree,force_N=force,
                input_time_s=old.time,time_s=result.time,solve_seconds=time.perf_counter()-started,proof=proof)
            persist();self.last_trace=trace;result.details=dict(trace,trace_path=str((trial/'trace.json').relative_to(self.run)))
            return result
        except Exception as exc:
            # A deadline raised by a PETSc callback may emerge as PETSc.Error(-1).
            # The budget object is marked BEFORE raising, so this classification
            # does not guess from elapsed time or from an error number.
            if self.deadline.expired:raise BudgetExpired('Adaptive budget pause') from exc
            trace['failure']=f'{type(exc).__name__}: {exc}';persist()
            raise TrialRejected(trace['failure']) from exc
        finally:self.assign(model,old)

    def close(self):
        if self.audit is not None:self.audit.close()
        self.patch.close()
        if hasattr(self,'forcing'):self.forcing.close()
