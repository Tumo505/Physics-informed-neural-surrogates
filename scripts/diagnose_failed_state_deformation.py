"""Read-only whole-element positivity and sampled deformation diagnostics."""
import json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.bernstein import BernsteinAudit
p=json.loads((P/'configs/physics/dynamic_quadratic_mpi_history_failed_step_v1.json').read_text());parent=json.loads((P/p['resume_probe_protocol']).read_text());prior=json.loads((P/p['previous_history_protocol']).read_text())
cp=np.load(P/parent['checkpoint']);old=np.load(P/prior['checkpoint']);engineering=np.load(P/p['engineering_checkpoint']);trial=np.load(P/p['iterate_checkpoint']);audit=BernsteinAudit(np.load(P/'data/processed'/p['reference']/'inputs.npz'))
states=[('accepted_physical_anchor',cp['u'],old['u_old']),('converged_engineering_root',engineering['u'],engineering['u_old']),('stalled_engineering_iterate',trial['u'],trial['u_old']),('exact_failed_equation_at_stalled_initialization',trial['u'],cp['u_old'])];records=[];started=time.perf_counter()
for name,u,history in states:
 record=dict(name=name,fields={})
 for label,field in [('endpoint',u),('stress',.4*history+.6*u)]:
  metrics=dict(minimum_sampled_determinant=float('inf'),minimum_sampled_singular_value=float('inf'),maximum_sampled_singular_value=0.,maximum_sampled_F_condition=0.)
  for start in range(0,len(audit.maps),audit.chunk):
   stop=min(start+audit.chunk,len(audit.maps));gl=audit.gl[start:stop]
   grad=np.stack([np.concatenate(((4*b[None,:,None]-1)*gl,np.stack([4*(b[i]*gl[:,j]+b[j]*gl[:,i]) for i,j in audit.edges],axis=1)),axis=1) for b in np.eye(4)],axis=1)
   F=np.eye(3)+np.einsum('eki,evkj->evij',field[audit.maps[start:stop]],grad);F=np.concatenate([F,F.mean(1,keepdims=True)],axis=1);sv=np.linalg.svd(F,compute_uv=False)
   metrics['minimum_sampled_determinant']=min(metrics['minimum_sampled_determinant'],float(np.linalg.det(F).min()))
   metrics['minimum_sampled_singular_value']=min(metrics['minimum_sampled_singular_value'],float(sv[:,:,-1].min()))
   metrics['maximum_sampled_singular_value']=max(metrics['maximum_sampled_singular_value'],float(sv[:,:,0].max()))
   metrics['maximum_sampled_F_condition']=max(metrics['maximum_sampled_F_condition'],float((sv[:,:,0]/np.maximum(sv[:,:,-1],1e-300)).max()))
  try:metrics['whole_element_proof']=audit.field(field)
  except Exception as exc:metrics['whole_element_proof_passed']=False;metrics['whole_element_failure']=str(exc)
  record['fields'][label]=metrics
 records.append(record);print('DEFORMATION CHECK',name,flush=True)
result=dict(complete=True,records=records,wall_seconds=time.perf_counter()-started,scope='Exact saved native canonical fields: whole-tetrahedron sufficient float64 Bernstein/subdivision proof and sampled F diagnostics at four vertices plus centroid of every element. Sampled singular values are not whole-element spectral bounds or a condition number for the assembled tangent. Whole-element positivity does not prove nonlinear root existence or uniqueness. No FE solve or physical sample created.')
(P/'experiments/metrics/dynamic_quadratic_failed_state_deformation_v1.json').write_text(json.dumps(result,indent=2)+'\n')
