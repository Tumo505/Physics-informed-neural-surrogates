"""All-cell positivity at endpoints AND generalized-alpha stress states."""
from pathlib import Path
import json,sys,hashlib,numpy as np,torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.models.dynamic_graph import DynamicGraph
torch.set_num_threads(2);cfg=json.loads((P/'configs/physics/dynamic_training.json').read_text());primary=json.loads((P/'configs/physics/dynamic_study.json').read_text());confirmation=json.loads((P/'configs/physics/dynamic_confirmation.json').read_text());cases=[c for c in primary['cases'] if c['split']=='test']+confirmation['cases'];evaluation=json.loads((P/'experiments/metrics/dynamic_final_evaluation.json').read_text());freeze=json.loads((P/'experiments/manifests/dynamic_final_freeze.json').read_text());top=torch.tensor(np.load(P/'data/processed/cardiac_geometry_ml/topology.npy'));labels=np.load(P/'data/processed/cardiac_geometry_ml/cluster_labels.npy');counts=np.bincount(labels,minlength=64);models={}
for key in evaluation['models']:
 regime,label=key.split('/');stats=dict(np.load(P/f'data/processed/{cfg["data_tag"]}_{regime}/normalizers.npz'));models[key]=[]
 for seed in cfg['seeds']:
  path=P/f'experiments/checkpoints/{cfg["name"]}_{regime}/{label}_seed{seed}.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==freeze['checkpoint_sha256'][str(path.relative_to(P))];net=DynamicGraph(stats).double();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);net.eval();models[key].append(net)
records=[]
for c in cases:
 root=P/f'data/processed/cardiac_dynamic_v1/{c["id"]}';g=np.load(root/'geometry.npz');a=np.load(root/'trajectory.npz');X=g['coords'];C=g['cells'];inv=np.linalg.inv((X[C[:,1:]]-X[C[:,:1]]).transpose(0,2,1));node=np.c_[X/.1,g['fiber'],g['sheet']];coarse=np.zeros((64,9));np.add.at(coarse,labels,node);coarse/=counts[:,None];delta=coarse[top[:,1],:3]-coarse[top[:,0],:3];b=[torch.tensor(v[None],dtype=torch.float64) for v in [coarse,np.r_[(np.array(c['scales'])-1)/.05,0.],np.c_[delta,np.linalg.norm(delta,axis=-1)],a['pressure']/16070.954]]
 for key,nets in models.items():
  with torch.inference_mode():q=torch.stack([n(*b[:3],top,b[3])[0] for n in nets]).mean(0);old=torch.cat((torch.zeros_like(q[:1]),q[:-1]),0);uaf=nets[0].decode(.4*old+.6*q).reshape(200,-1,3).numpy()
  minimum=1.;inverted=0
  for u in np.array_split(uaf,20):
   F=np.eye(3)+np.einsum('beij,ejk->beik',(u[:,C[:,1:]]-u[:,C[:,:1]]).transpose(0,1,3,2),inv);J=np.linalg.det(F);minimum=min(minimum,float(J.min()));inverted+=int((J<=0).sum())
  records.append({'model':key,'case':c['id'],'minimum_stress_state_detF_all200':minimum,'inverted_element_stress_states':inverted})
 print('ALPHA-STATE AUDITED',c['id'],flush=True)
result={'passed':all(r['inverted_element_stress_states']==0 and r['minimum_stress_state_detF_all200']>0 for r in records),'alpha_f':.4,'n_ensemble_trajectories':len(records),'n_independent_test_geometries':len(cases),'minimum_stress_state_detF':min(r['minimum_stress_state_detF_all200'] for r in records),'scope':'all200 generalized-alpha stress evaluation states and every P1 tetrahedron, four ensembles; endpoint checks reported separately; no new model choice','records':records};(P/'experiments/metrics/dynamic_alpha_jacobians.json').write_text(json.dumps(result,indent=2));print('FULL STRESS-STATE POSITIVITY',result['passed'],result['minimum_stress_state_detF'],flush=True)
