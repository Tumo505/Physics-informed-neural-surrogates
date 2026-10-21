"""Pinned benchmark mechanics adapter for isolated legacy FEniCS Python.
Only solver options/discretization differ; original constitutive/forms unchanged.
"""
from pathlib import Path
import os,sys,types,time
import numpy as np
P=Path(__file__).resolve().parents[3]
os.environ.setdefault('DIJITSO_CACHE_DIR',str(P/'.fenics-cache-activated'))
import dolfin as df
try:import ufl_legacy as ufl
except ImportError:import ufl
pkg=types.ModuleType('cardiac_benchmark');pkg.__path__=[str(P/'external/cardiac_benchmark/src/cardiac_benchmark')];sys.modules['cardiac_benchmark']=pkg
from cardiac_benchmark.geometry import LVGeometry
from cardiac_benchmark.microstructure import create_microstructure
from cardiac_benchmark.material import HolzapfelOgden
from cardiac_benchmark.problem import LVProblem
from cardiac_benchmark.pressure_model import pressure_function,default_parameters_benchmark1

def setup(refinement_level=0,dt=None,geometry_scales=None):
 df.set_log_level(30);df.parameters['form_compiler']['quadrature_degree']=6;df.parameters['form_compiler']['cpp_optimize']=True
 meshfile=next((P/'data/raw/benchmark').rglob('*.xdmf'));mesh,ffun,markers=LVGeometry.load_mesh(meshfile)
 mesh_parameters=LVGeometry.default_mesh_parameters()
 if geometry_scales is not None:
  radial,longitudinal=map(float,geometry_scales)
  if min(radial,longitudinal)<=0:raise ValueError('Nonpositive geometry scaling')
  mesh.coordinates()[:]*=np.array([longitudinal,radial,radial])
  mesh.bounding_box_tree().build(mesh)
  for key in ['r_short_endo','r_short_epi']:mesh_parameters[key]*=radial
  for key in ['r_long_endo','r_long_epi']:mesh_parameters[key]*=longitudinal
 fp=LVGeometry.default_fiber_parameters();fp['function_space']='P_1';micro=create_microstructure(mesh,ffun,markers,mesh_parameters,fp)
 fibers=[micro.f0,micro.s0,micro.n0]
 for _ in range(refinement_level):
  # Keep the same polyhedral domain and continuous P1 fiber fields; isolate
  # displacement discretization without changing geometry or constitutive data.
  df.parameters['refinement_algorithm']='plaza_with_parent_facets'
  refined=df.refine(mesh);ffun=df.adapt(ffun,refined)
  space=df.VectorFunctionSpace(refined,'CG',1)
  fibers=[df.interpolate(f,space) for f in fibers];mesh=refined
 geo=LVGeometry(mesh=mesh,ffun=ffun,markers=markers,f0=fibers[0],s0=fibers[1],n0=fibers[2])
 mat=HolzapfelOgden(geo.f0,geo.s0,tau=df.Constant(0.));params=LVProblem.default_parameters();params['function_space']='P_1'
 if dt is not None:params['dt']=float(dt)
 problem=LVProblem(geo,mat,parameters=params,solver_parameters={'linear_solver':'mumps','petsc':{'ksp_type':'preonly','pc_type':'none'}})
 return geo,mat,problem

def vertex_map(problem):return df.vertex_to_dof_map(problem.u_space)

def assign_nodes(problem,u):
 a=np.empty(problem.u_space.dim());a[vertex_map(problem)]=np.asarray(u).reshape(-1);problem.u.vector().set_local(a);problem.u.vector().apply('insert')

def assign_history(problem,u_old=None,v_old=None,a_old=None):
 for name,value in [('u_old',u_old),('v_old',v_old),('a_old',a_old)]:
  f=getattr(problem,name)
  if value is None:f.vector().zero()
  else:
   a=np.empty(problem.u_space.dim());a[vertex_map(problem)]=np.asarray(value).reshape(-1);f.vector().set_local(a);f.vector().apply('insert')

def residual(problem):return df.assemble(problem._problem._F)

def solve_at_pressure(problem,p):
 for f in [problem.u,problem.u_old,problem.v_old,problem.a_old]:f.vector().zero()
 problem.parameters['p'].assign(float(p));r0=residual(problem).norm('l2');t=time.perf_counter();iterations,ok=problem.solver.solve();elapsed=time.perf_counter()-t
 reason=int(problem.solver._snes.getConvergedReason())
 if reason<=0:raise RuntimeError(f'Failed solve pressure={p}, SNES reason={reason}')
 r=residual(problem)
 if not np.isfinite(r.norm('l2')):raise RuntimeError('Nonfinite reference residual')
 u=problem.u.compute_vertex_values(problem.geometry.mesh).reshape(3,-1).T
 np.testing.assert_allclose(u,problem.u.vector().get_local()[vertex_map(problem)].reshape(-1,3),atol=1e-14)
 return u,{'pressure_Pa':float(p),'converged':True,'residual_l2_N':r.norm('l2'),'initial_residual_l2_N':r0,'seconds':elapsed,'iterations':int(iterations)}

def export_geometry(geo,mat,problem):
 from FIAT.reference_element import UFCTetrahedron
 from FIAT.quadrature_schemes import create_quadrature
 mesh=geo.mesh;mesh.init(2,0);mesh.init(2,3);faces=[];normals=[];markers=[];owners=[]
 for facet in df.facets(mesh):
  if facet.exterior():
   faces.append(facet.entities(0));normals.append(facet.normal().array());markers.append(int(geo.ffun[facet]));owners.append(int(facet.entities(3)[0]))
 quad=create_quadrature(UFCTetrahedron(),6)
 return {'coords':mesh.coordinates(),'cells':mesh.cells().astype('int64'),'fiber':geo.f0.compute_vertex_values(mesh).reshape(3,-1).T,'sheet':geo.s0.compute_vertex_values(mesh).reshape(3,-1).T,'facets':np.asarray(faces,dtype='int64'),'facet_normals':np.asarray(normals),'facet_markers':np.asarray(markers),'facet_cells':np.asarray(owners),'qpoints':quad.get_points(),'qweights':quad.get_weights(),'v2d':vertex_map(problem),'material_keys':np.array(list(mat.parameters)),'material_values':np.array([float(v) for v in mat.parameters.values()])}
