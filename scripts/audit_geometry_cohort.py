"""Audit all generated FE geometry/load cases; no target-based filtering."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parents[1];m=json.loads((P/'experiments/manifests/cardiac_geometry_v1.json').read_text());records=[];scales=[]
if not m['complete']:raise RuntimeError('Incomplete cohort')
for rec in m['records']:
 g=rec['geometry_id'];a=np.load(P/f'data/processed/cardiac_geometry_v1/geometry_{g:03d}.npz');X=a['coords'];C=a['cells'];U=a['displacements'];finite=all(np.isfinite(a[k]).all() for k in ['coords','displacements','fiber','sheet','material_values']);assert finite
 dm=(X[C[:,1:]]-X[C[:,:1]]).transpose(0,2,1);inv=np.linalg.inv(dm);du=(U[:,C[:,1:]]-U[:,C[:,:1]]).transpose(0,1,3,2);F=np.eye(3)+np.einsum('beij,ejk->beik',du,inv);J=np.linalg.det(F);minimum=float(J.min());inverted=float((J<=0).mean());assert minimum>0
 residuals=[r['residual_l2_N']/(np.sqrt(3*len(X))*float(a['force_scale_N'])) for r in rec['solves']];assert max(residuals)<1e-8
 scales.append(a['geometry_scales']);records.append({'geometry':g,'n_nodes':len(X),'n_tetrahedra':len(C),'n_loads':len(U),'finite':finite,'minimum_endpoint_detF_all_loads':minimum,'inverted_element_fraction':inverted,'maximum_reference_normalized_residual_rms':max(residuals)})
scales=np.array(scales);dist=np.linalg.norm(scales[:,None]-scales[None],axis=-1);np.fill_diagonal(dist,np.inf)
result={'complete':True,'n_geometries':len(records),'n_iid_parameter_geometries':64,'n_shift_corners':4,'n_simulations':sum(r['n_loads'] for r in records),'minimum_endpoint_detF':min(r['minimum_endpoint_detF_all_loads'] for r in records),'maximum_reference_normalized_residual_rms':max(r['maximum_reference_normalized_residual_rms'] for r in records),'exact_duplicate_parameter_rows':len(scales)-len(np.unique(scales,axis=0)),'minimum_parameter_scale_distance':float(dist.min()),'near_duplicate_caveat':'All cases intentionally derive from one ellipsoid family with small parameter perturbations; geometry grouping prevents load leakage but does not establish anatomical-source independence','units':{'coords':'m','displacement':'m','pressure':'Pa','residual':'N'},'per_geometry':records}
(P/'experiments/metrics/geometry_cohort_audit.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='per_geometry'},flush=True)
