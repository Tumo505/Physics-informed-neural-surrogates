"""Native P2 solver construction; identical weak forms for direct/iterative solves."""
from .reference_fem import LVProblem, df


def make_quadratic_problem(geo, material, parameters, method='mumps',
                           linear_maximum_iterations=1000, gmres_restart=100):
    if method == 'mumps':
        return LVProblem(geo, material, parameters=parameters,
            solver_parameters={'linear_solver': 'mumps',
                'petsc': {'ksp_type': 'preonly', 'pc_type': 'none'}})
    if method != 'gamg':
        raise ValueError(f'Unsupported native quadratic solver: {method}')
    from petsc4py import PETSc
    problem = LVProblem(geo, material, parameters=parameters, solver_parameters={
        'linear_solver': 'gmres', 'preconditioner': 'petsc_amg',
        'petsc': {'ksp_type': 'gmres', 'pc_type': 'gamg', 'ksp_gmres_restart': int(gmres_restart),
                  'ksp_norm_type': 'unpreconditioned'},
        'krylov_solver': {'absolute_tolerance': 1e-12, 'relative_tolerance': 1e-10,
            'maximum_iterations': int(linear_maximum_iterations), 'monitor_convergence': False}})
    expressions = [('1','0','0'), ('0','1','0'), ('0','0','1'),
        ('0','-x[2]','x[1]'), ('x[2]','0','-x[0]'), ('-x[1]','x[0]','0')]
    modes = [df.interpolate(df.Expression(e, degree=1), problem.u_space).vector().copy()
             for e in expressions]
    basis = df.VectorSpaceBasis(modes); basis.orthonormalize()
    nullspace = PETSc.NullSpace().create(vectors=[df.as_backend_type(v).vec() for v in modes])
    original_J = problem._problem.J
    def with_nullspace(A, x):
        original_J(A, x)
        df.as_backend_type(A).mat().setNearNullSpace(nullspace)
    problem._problem.J = with_nullspace
    # Hold all native vectors alive for PETSc's near-nullspace throughout the solve.
    problem._quadratic_solver_objects = (modes, basis, nullspace)
    return problem
