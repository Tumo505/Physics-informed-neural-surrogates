"""Native integration check: restore iterative/direct backends between solves."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,residual,df
from lv_surrogate.physics.quadratic_solver import make_quadratic_problem

root = P/'data/processed/cardiac_quadratic_reference_halfstep_gamg_v1'
m = json.loads((root/'manifest.json').read_text()); assert m['complete']
data = np.load(root/'inputs.npz'); u = np.load(root/'displacement_coefficients.npy',mmap_mode='r')
geo,material,base = setup(dt=.000625,geometry_scales=[1,1])
parameters = dict(base.parameters); parameters['function_space']='P_2'
problems = {method:make_quadratic_problem(geo,material,parameters,method) for method in ['gamg','mumps']}
for p in problems.values():
    p.solver._solver.parameters['absolute_tolerance']=1e-8
    p.solver._solver.parameters['relative_tolerance']=0.
    p.solver._solver.parameters['maximum_iterations']=100
    p.solver._solver.parameters['line_search']='bt'
    np.testing.assert_allclose(p.u_space.tabulate_dof_coordinates(),data['dof_coordinates'],atol=1e-14,rtol=0)
records=[]
for step in range(2):
    for method,p in problems.items():
        df.PETScOptions.clear()
        options = dict(ksp_type='gmres',pc_type='gamg',ksp_gmres_restart=100,ksp_norm_type='unpreconditioned') if method=='gamg' else dict(mat_mumps_icntl_23=20000)
        for key,value in options.items():df.PETScOptions.set(key,value)
        p.parameters['p'].assign(float(data['pressure'][step]))
        best_R=float('inf'); old=p.u_old.vector().get_local(); v=p.v_old.vector().get_local(); a=p.a_old.vector().get_local();dt=.000625
        for guess in [old,old+dt*v,old+dt*v+.5*dt**2*a]:
            p.u.vector().set_local(guess);p.u.vector().apply('insert')
            R=float(residual(p).norm('l2'))
            if np.isfinite(R) and R<best_R:best=guess;best_R=R
        p.u.vector().set_local(best);p.u.vector().apply('insert')
        print('INTERLEAVED SOLVE',step,method,flush=True)
        p.solver.solve()
        snes=p.solver._snes;ksp=snes.getKSP();R=float(residual(p).norm('l2'));values=p.u.vector().get_local()
        relative=float(np.linalg.norm(values-u[step])/max(np.linalg.norm(u[step]),1e-30))
        absolute=float(abs(values-u[step]).max())
        backend_ok=(ksp.getType()=='gmres' and ksp.getPC().getType()=='gamg') if method=='gamg' else (ksp.getPC().getType()=='lu' and ksp.getPC().getFactorSolverType()=='mumps')
        accepted=bool(snes.getConvergedReason()>0 and np.isfinite(R) and R<1e-6 and relative<1e-6 and absolute<1e-9 and backend_ok)
        records.append(dict(step=step,method=method,accepted=accepted,residual_N=R,relative_coefficient_difference=relative,maximum_absolute_difference_m=absolute,KSP=ksp.getType(),PC=ksp.getPC().getType()))
        assert accepted,records[-1]
        p._update_fields()
result=dict(complete=True,passed=True,engineering_only=True,records=records,
    executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    reference_output_sha256=m['output_sha256'],
    scope='Two coarse startup timesteps with separate recursively updated histories, alternating native iterative/direct solves in one process. Verify restored backend options and agreement with the accepted reference. Integration test of backend interleaving only, not loaded-state robustness or full-cycle fallback equivalence.')
(P/'experiments/metrics/quadratic_fallback_interleaving.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result,indent=2),flush=True)
