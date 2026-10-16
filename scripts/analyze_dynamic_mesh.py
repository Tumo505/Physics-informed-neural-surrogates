"""Exact FE volume-L2 comparison on every full-cycle refined-mesh endpoint."""
from pathlib import Path
import sys,json,numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,assign_nodes,df
root=P/'data/processed/cardiac_dynamic_v1';a=np.load(root/'convergence_dt0.005/trajectory.npz');b=np.load(root/'convergence_refined_dt0005/trajectory.npz');gc,_,coarse=setup(dt=.005,geometry_scales=[1,1]);gf,_,fine=setup(dt=.005,geometry_scales=[1,1],refinement_level=1)
for name,g in [('convergence_dt0.005',gc),('convergence_refined_dt0005',gf)]:
 saved=np.load(root/name/'geometry.npz');np.testing.assert_array_equal(g.mesh.cells(),saved['cells']);np.testing.assert_allclose(g.mesh.coordinates(),saved['coords'],rtol=0,atol=1e-14)
areas={}
for label,problem in [('coarse',coarse),('refined',fine)]:areas[label]=[float(df.assemble(df.Constant(1.)*problem.ds(k))) for k in [1,2,3]]
np.testing.assert_allclose(areas['coarse'],areas['refined'],rtol=1e-11);volume=float(df.assemble(df.Constant(1.)*df.dx(domain=gf.mesh)));records=[]
for i,(uc,uf) in enumerate(zip(a['u'],b['u'])):
 assign_nodes(coarse,uc);assign_nodes(fine,uf);interp=df.interpolate(coarse.u,fine.u_space);diff=df.Function(fine.u_space);diff.vector()[:]=interp.vector()-fine.u.vector();d=float(df.norm(diff,'L2'));ref=float(df.norm(fine.u,'L2'));records.append({'time_s':float(a['time'][i]),'absolute_volume_L2_difference_m_power2_5':d,'relative_volume_L2_difference':d/max(ref,1e-30),'refined_volume_L2_m_power2_5':ref})
 if i%20==0:print('MESH COMPARED',i,flush=True)
relative=float(np.sqrt(sum(r['absolute_volume_L2_difference_m_power2_5']**2 for r in records)/sum(r['refined_volume_L2_m_power2_5']**2 for r in records)));result={'complete':True,'n_endpoints':len(records),'dt_s':.005,'mesh_tetrahedra':[gc.mesh.num_cells(),gf.mesh.num_cells()],'boundary_areas_m2':areas,'mesh_volume_m3':volume,'relative_spacetime_volume_L2_difference':relative,'maximum_endpoint_relative_volume_L2_difference':max(r['relative_volume_L2_difference'] for r in records),'volume_time_component_rmse_m':float(np.sqrt(np.mean([r['absolute_volume_L2_difference_m_power2_5']**2 for r in records])/(3*volume))),'within_1percent_spacetime_tolerance':relative<=.01,'cases':records,'scope':'same polyhedral geometry, source material/BC, interpolated same continuous fibers; isolate displacement discretization over complete passive cycle','limitations':['two levels quantify sensitivity; do not establish asymptotic mesh convergence','one geometry/waveform; fine-grid result retains temporal and spatial discretization error']};(P/'experiments/metrics/dynamic_mesh_convergence.json').write_text(json.dumps(result,indent=2));print('FULL-CYCLE MESH COMPLETE',relative,flush=True)
