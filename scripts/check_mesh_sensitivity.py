"""Two-level P1 mesh sensitivity on the same authentic polyhedral LV domain.
This does not establish asymptotic convergence or anatomical generalization.
"""
from pathlib import Path
import json,sys,time,hashlib,os
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,solve_at_pressure,df,ufl
manifest=json.loads((P/'experiments/manifests/cardiac_cohort.json').read_text())
pressures=np.array([.25,.5,1.])*manifest['pressure_range_Pa'][1]
start=time.perf_counter();coarse_geo,_,coarse=setup();fine_geo,_,fine=setup(refinement_level=1)
volumes=[float(df.assemble(df.Constant(1.)*df.dx(domain=g.mesh))) for g in [coarse_geo,fine_geo]]
np.testing.assert_allclose(volumes[0],volumes[1],rtol=1e-12)
markers={}
for name,problem in [('coarse',coarse),('refined',fine)]:
 markers[name]={str(k):float(df.assemble(df.Constant(1.)*problem.ds(k))) for k in [1,2,3]}
for k in markers['coarse']:np.testing.assert_allclose(markers['coarse'][k],markers['refined'][k],rtol=1e-11,atol=1e-14)
records=[]
for p in pressures:
 uc,rc=solve_at_pressure(coarse,float(p));uf,rf=solve_at_pressure(fine,float(p))
 coarse_on_fine=df.interpolate(coarse.u,fine.u_space)
 difference=df.Function(fine.u_space);difference.vector()[:]=coarse_on_fine.vector()-fine.u.vector()
 absolute=float(df.norm(difference,'L2'));reference=float(df.norm(fine.u,'L2'))
 V=df.FunctionSpace(fine_geo.mesh,'DG',0)
 det=df.project(ufl.det(ufl.Identity(3)+ufl.grad(fine.u)),V).vector().get_local()
 record={'pressure_Pa':float(p),'coarse':rc,'refined':rf,'displacement_L2_difference_m_power_2_5':absolute,'relative_displacement_volume_L2_difference':absolute/reference,'refined_displacement_L2_m_power_2_5':reference,'minimum_refined_endpoint_detF':float(det.min()),'refined_inverted_element_fraction':float((det<=0).mean())}
 records.append(record);print(json.dumps(record),flush=True)
 np.savez_compressed(P/f'data/processed/cardiac/mesh_sensitivity_{len(records)}.npz',coarse_coords=coarse_geo.mesh.coordinates(),coarse_displacements=uc,refined_coords=fine_geo.mesh.coordinates(),refined_displacements=uf)
 result={'task':'two-level displacement mesh sensitivity, identical polyhedral domain and interpolated original P1 fiber fields','complete':len(records)==len(pressures),'runtime':{'dolfin':df.__version__,'environment':{k:os.environ.get(k) for k in ['FI_PROVIDER','FI_TCP_IFACE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},'note':'pin TCP to physical interface en0 on this macOS host; VPN utun6 caused MPI shutdown failure'},'source_commit':manifest['source_commit'],'scope':'passive one-step dynamics from rest; same dt/material/Robin/follower loads; no mesh-independent solution or asymptotic rate claimed','pressure_source':'25%, 50%, 100% of documented benchmark waveform maximum; numerical sensitivity cases, no surrogate selection','mesh_volume_m3':volumes,'boundary_area_m2':markers,'meshes':{name:{'vertices':g.mesh.num_vertices(),'tetrahedra':g.mesh.num_cells(),'displacement_dofs':pr.u_space.dim()} for name,g,pr in [('coarse',coarse_geo,coarse),('refined',fine_geo,fine)]},'cases':records,'seconds':time.perf_counter()-start,'source_sha256':{str(q.relative_to(P)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),P/'src/lv_surrogate/physics/reference_fem.py']}}
 (P/'experiments/metrics/mesh_sensitivity.json').write_text(json.dumps(result,indent=2))
print('MESH SENSITIVITY COMPLETE',flush=True)
