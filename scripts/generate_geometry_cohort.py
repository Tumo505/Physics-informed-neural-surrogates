"""Verified numerical geometry sensitivity cohort; not an anatomical population."""
from pathlib import Path
import json,sys,hashlib,time
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,export_geometry,solve_at_pressure,df
cfg=json.loads((P/'configs/physics/geometry_cohort.json').read_text());out=P/'data/processed'/cfg['name'];out.mkdir(exist_ok=True)
cohort=json.loads((P/'experiments/manifests/cardiac_cohort.json').read_text());pressures=np.asarray(cfg['pressure_fractions_of_documented_waveform_max'])*cohort['pressure_range_Pa'][1]
scales=np.r_[np.random.default_rng(cfg['seed_geometry']).uniform(*cfg['scale_bounds'],size=(cfg['iid_geometry_count'],2)),np.array(cfg['shift_scales'])]
base=np.load(P/'data/processed/cardiac/cohort.npz');records=[];start=time.perf_counter()
for i,scale in enumerate(scales):
 path=out/f'geometry_{i:03d}.npz';record_path=out/f'geometry_{i:03d}.json'
 if path.exists() and record_path.exists():records.append(json.loads(record_path.read_text()));continue
 geo,mat,problem=setup(geometry_scales=scale);a=export_geometry(geo,mat,problem)
 np.testing.assert_array_equal(a['cells'],base['cells']);np.testing.assert_allclose(a['coords'],base['coords']*np.array([scale[1],scale[0],scale[0]]),rtol=0,atol=1e-14)
 volume=float(df.assemble(df.Constant(1.)*df.dx(domain=geo.mesh)));baseline_volume=abs(np.linalg.det(base['coords'][base['cells'][:,1:]]-base['coords'][base['cells'][:,:1]])).sum()/6
 np.testing.assert_allclose(volume,baseline_volume*scale[0]**2*scale[1],rtol=1e-11)
 fields=[];solves=[]
 for p in pressures:
  u,r=solve_at_pressure(problem,float(p));fields.append(u);solves.append(r)
 area=float(df.assemble(df.Constant(1.)*problem.ds(1)));force_scale=float(pressures.max())*area/np.sqrt(problem.u_space.dim())
 np.savez_compressed(path,**a,displacements=fields,pressures=pressures,geometry_scales=scale,force_scale_N=force_scale)
 rec={'geometry_id':i,'scales':scale.tolist(),'reference_volume_m3':volume,'endo_area_m2':area,'force_scale_N':force_scale,'solves':solves,'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 record_path.write_text(json.dumps(rec,indent=2));records.append(rec);print('GEOMETRY',i,'DONE',scale,flush=True)
 manifest={'config':cfg,'complete':len(records)==len(scales),'geometry_count':len(records),'load_cases_per_geometry':len(pressures),'split_unit':'geometry; all loads share same split','pressure_values_Pa':pressures.tolist(),'source_commit':cohort['source_commit'],'nodal_correspondence':'verified exact affine coordinate mapping and identical tetrahedral indices for every case','normalizers':'none fitted here; training script must use only training geometry IDs','records':records,'seconds_this_invocation':time.perf_counter()-start,'source_sha256':{str(q.relative_to(P)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),P/'src/lv_surrogate/physics/reference_fem.py',P/'configs/physics/geometry_cohort.json']}}
 (P/f'experiments/manifests/{cfg["name"]}.json').write_text(json.dumps(manifest,indent=2))
print('GEOMETRY COHORT COMPLETE',flush=True)
