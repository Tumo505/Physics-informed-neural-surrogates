"""One actual benchmark-formulation dynamic LV timestep; not a full cardiac cohort."""
from pathlib import Path
import os,sys,types,time,json
P=Path(__file__).resolve().parents[1]
# Keep generated compiler artifacts inside the project.
os.environ.setdefault('DIJITSO_CACHE_DIR',str(P/'.fenics-cache'));os.environ.setdefault('XDG_CACHE_HOME',str(P/'.fenics-cache'))
import numpy as np
import dolfin as df
try:import ufl_legacy as ufl
except ImportError:import ufl
# Import pinned modules without package CLI imports/metadata; source files unedited.
pkg=types.ModuleType('cardiac_benchmark');pkg.__path__=[str(P/'external/cardiac_benchmark/src/cardiac_benchmark')];sys.modules['cardiac_benchmark']=pkg
from cardiac_benchmark.geometry import LVGeometry
from cardiac_benchmark.microstructure import create_microstructure
from cardiac_benchmark.material import HolzapfelOgden
from cardiac_benchmark.problem import LVProblem
from cardiac_benchmark.pressure_model import pressure_function,default_parameters_benchmark1
meshfile=next((P/'data/raw/benchmark').rglob('*.xdmf'))
mesh,ffun,markers=LVGeometry.load_mesh(meshfile)
print('mesh',mesh.num_vertices(),mesh.num_cells(),'facet marker values',np.unique(ffun.array()),flush=True)
df.parameters['form_compiler']['quadrature_degree']=6;df.parameters['form_compiler']['cpp_optimize']=True
fiber_params=LVGeometry.default_fiber_parameters();fiber_params['function_space']='P_1'
start=time.perf_counter();micro=create_microstructure(mesh,ffun,markers,LVGeometry.default_mesh_parameters(),fiber_params);fiber_seconds=time.perf_counter()-start
geo=LVGeometry(mesh=mesh,ffun=ffun,markers=markers,f0=micro.f0,s0=micro.s0,n0=micro.n0)
material=HolzapfelOgden(geo.f0,geo.s0,tau=df.Constant(0.))
params=LVProblem.default_parameters();params['function_space']='P_1'
pval=float(np.asarray(pressure_function((0,1),parameters=default_parameters_benchmark1(),t_eval=np.array([.0006]))).reshape(-1)[0]);params['p'].assign(pval)
start=time.perf_counter();problem=LVProblem(geo,material,parameters=params,solver_parameters={'linear_solver':'mumps','petsc':{'ksp_type':'preonly','pc_type':'none'}});setup_seconds=time.perf_counter()-start
form=problem._problem._F
start=time.perf_counter();r0=df.assemble(form).norm('l2');assembly0=time.perf_counter()-start
start=time.perf_counter();iterations,converged=problem.solver.solve();solve_seconds=time.perf_counter()-start
res=df.assemble(form).get_local();rnorm=float(np.linalg.norm(res));unorm=float(problem.u.vector().norm('l2'))
assert converged,'trusted nonlinear solver failed'
orig=problem.u.vector().get_local();rng=np.random.default_rng(11);problem.u.vector().set_local(orig+1e-7*rng.normal(size=len(orig)));problem.u.vector().apply('insert');pert=df.assemble(form).norm('l2');problem.u.vector().set_local(orig);problem.u.vector().apply('insert')
F=ufl.Identity(3)+ufl.grad(problem.u);Q=df.FunctionSpace(mesh,'DG',0);J=df.project(ufl.det(F),Q).vector().get_local()
u=problem.u.compute_vertex_values(mesh).reshape(3,-1).T
out={'label':'trusted cardiac benchmark first dynamic step, P1 displacement/fibers; not full published trajectory','source_commit':'c7d2d1ba6d680627dc465598dbf6b0363a75d604','dolfin':df.__version__,'n_nodes':mesh.num_vertices(),'n_tetrahedra':mesh.num_cells(),'n_dofs':problem.u_space.dim(),'quadrature_degree':6,'pressure_Pa':pval,'pressure_evaluation_time_s':.0006,'dt_s':.001,'alpha_f':.4,'alpha_m':.2,'active_tau_Pa':0.,'material_parameters':{k:float(v) for k,v in material.parameters.items()},'boundary_parameters':{k:float(v) for k,v in params.items() if k!='function_space'},'fiber_function_space':'P1','displacement_function_space':'P1','reference_mesh_m':str(meshfile),'all_dofs_free':True,'BC':'Robin epi/base; follower endocardial pressure, no imposed Dirichlet','converged':bool(converged),'solver_iterations_reported':int(iterations),'initial_force_l2_N':r0,'converged_force_l2_N':rnorm,'normalized_residual':rnorm/max(r0,1e-12),'perturbed_force_l2_N':pert,'displacement_dof_l2_m':unorm,'max_nodal_displacement_m':float(np.linalg.norm(u,axis=-1).max()),'min_detF':float(J.min()),'inverted_element_fraction':float((J<=0).mean()),'fiber_setup_seconds':fiber_seconds,'problem_setup_seconds':setup_seconds,'initial_assembly_seconds':assembly0,'solve_seconds_including_first_JIT':solve_seconds,'limitations':['one timestep only','P1 choice documented; default P2 not reproduced','no independently implemented residual comparison yet','no physics-guided surrogate trained','timing includes first JIT and is not a matched Dalton FEM timing']}
np.savez_compressed(P/'experiments/metrics/cardiac_probe.npz',coords=mesh.coordinates(),cells=mesh.cells(),displacement=u,residual=res,detF=J,ffun=ffun.array())
(P/'experiments/metrics/cardiac_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
