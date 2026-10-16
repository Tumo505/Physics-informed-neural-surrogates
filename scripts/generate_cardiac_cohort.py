"""Thirty-two passive startup load cases on one authentic LV geometry.
These are load variants, NOT independent geometries or full trajectories.
"""
from pathlib import Path
import sys,json,time
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
import numpy as np
from lv_surrogate.physics.reference_fem import setup,solve_at_pressure,export_geometry,pressure_function,default_parameters_benchmark1,df
out=P/'data/processed/cardiac';out.mkdir(exist_ok=True);geo,mat,problem=setup()
wave=np.asarray(pressure_function((0,1),parameters=default_parameters_benchmark1(),t_eval=np.linspace(0,1,1001)))
pressures=np.linspace(float(wave.min()),float(wave.max()),32);fields=[];records=[];geometry=export_geometry(geo,mat,problem)
for i,p in enumerate(pressures):
 u,record=solve_at_pressure(problem,p);fields.append(u);record['case']=i;records.append(record);print('CASE',record,flush=True)
faces=geometry['facets'];X=geometry['coords'];area=np.linalg.norm(np.cross(X[faces[:,1]]-X[faces[:,0]],X[faces[:,2]]-X[faces[:,0]]),axis=-1)/2;endo_area=float(area[geometry['facet_markers']==1].sum());force_scale=max(float(pressures.max())*endo_area/np.sqrt(problem.u_space.dim()),1e-10)
np.savez_compressed(out/'cohort.npz',**geometry,displacements=np.asarray(fields),pressures=pressures,force_scale_N=force_scale)
ids=np.random.default_rng(2027).permutation(32);splits={'train':ids[:20].tolist(),'validation':ids[20:26].tolist(),'test':ids[26:].tolist()}
manifest={'task':'independent startup pressure sweeps from rest; one fixed LV geometry, P1 displacement/fiber, zero active stress','case_count':32,'independent_geometries':1,'pressure_range_Pa':[float(pressures.min()),float(pressures.max())],'pressure_range_source':'range of pinned benchmark1 pressure waveform evaluated at1001 times over0:1s','dt_s':.001,'alpha_m':.2,'alpha_f':.4,'constitutive':'exact pinned HolzapfelOgden source with viscosity','boundary':'reference epi/base Robin and follower endocardial pressure; all DOFs free','quadrature_degree':6,'force_scale_N':force_scale,'force_scaling':'p_max*reference_endo_area/sqrt(n_dofs), nonzero shared scale','n_quadrature_points':len(geometry['qweights']),'split_seed':2027,'splits':splits,'cases':records,'source_commit':'c7d2d1ba6d680627dc465598dbf6b0363a75d604'}
(P/'experiments/manifests/cardiac_cohort.json').write_text(json.dumps(manifest,indent=2));print('COHORT DONE',manifest['pressure_range_Pa'],flush=True)
