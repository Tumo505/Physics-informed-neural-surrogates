"""Seed a new reference from an immutable verified physical chain; preserve every original."""
import argparse,json,sys,shutil,hashlib,mmap
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.durable_reference import atomic_json,sha
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args();pp=P/args.protocol;p=json.loads(pp.read_text());cfg=json.loads((P/p['config']).read_text());root=P/'data/processed'/cfg['name'];assert not root.exists(),'Never overwrite or retry a seed'
for f,h in p['frozen_source_sha256'].items():assert sha(P/f)==h
for f,h in p['required_audit_sha256'].items():
 a=json.loads((P/f).read_text());assert a['passed'] and sha(P/f)==h
for f,h in p.get('input_sha256',{}).items():assert sha(P/f)==h
for f in p.get('required_audits',[]):assert json.loads((P/f).read_text())['passed']
source=P/'data/processed'/p['seed_reference'];sm=json.loads((source/'manifest.json').read_text());inputs=np.load(source/'inputs.npz');n=len(inputs['time']);ndof=len(inputs['dof_coordinates']);base=p['base_seed_steps']
assert shutil.disk_usage(P).free>100*2**30
for k in ['dt_s','duration_s','displacement_degree','refinement_level','geometry_scales','amplitude','timing_scale','quadrature_degree']:assert cfg[k]==sm['config'][k]
root.mkdir();c=np.lib.format.open_memmap(root/'displacement_coefficients.npy',mode='w+',dtype='float64',shape=(n,ndof));node_map=np.full((len(inputs['coords']),3),-1,dtype=int)
for component in range(3):
 for j in range(4):node_map[inputs['cells'][:,j],component]=inputs['canonical_cell_dofs'][:,j,component]
assert (node_map>=0).all();nodes=np.lib.format.open_memmap(root/'displacements.npy',mode='w+',dtype='float64',shape=(n,len(node_map),3));original=np.load(source/'displacement_coefficients.npy',mmap_mode='r');steps=list(sm['steps'][:base]);assert len(steps)==base
if (source/'positivity.json').exists():positivity=json.loads((source/'positivity.json').read_text())[:base]
else:
 b=json.loads((P/p['seed_positivity']).read_text());assert sm['complete'] and b['whole_element_positivity_sufficient_condition_passed'] and b['output_sha256']==sm['output_sha256']
 positivity=[dict(step=i,time_s=inputs['time'][i],**{s:dict(passed=True,minimum_Bernstein_bound=x[s+'_minimum_Bernstein_lower_bound'],inherited_audit=p['seed_positivity'],inherited_audit_sha256=sha(P/p['seed_positivity'])) for s in ['endpoint','stress']}) for i,x in enumerate(b['records'][:base])]
assert len(positivity)==base;digest=hashlib.sha256()
for i in range(base):
 c[i]=original[i];nodes[i]=original[i][node_map];digest.update(np.asarray(original[i]).tobytes())
 if (i+1)%8==0:
  c.flush();nodes.flush()
  for x in [c,nodes,original]:
   if hasattr(x._mmap,'madvise'):x._mmap.madvise(mmap.MADV_DONTNEED)
if p.get('base_checkpoint'):
 cp=np.load(P/p['base_checkpoint']);assert int(cp['completed_steps'])==base;state={k:cp[k].copy() for k in ['u','u_old','v_old','a_old']}
else:
 old=np.zeros(ndof);v=old.copy();acc=old.copy();dt=cfg['dt_s']
 for u in c[:base]:
  anew=(u-(old+dt*v+(.5-.36)*dt**2*acc))/(.36*dt**2);v=v+(1-.7)*dt*acc+.7*dt*anew;acc=anew;old=np.array(u,copy=True)
 state=dict(u=old.copy(),u_old=old,v_old=v,a_old=acc)
lineage=[]
for link in p.get('probe_chain',[]):
 lp=P/link['protocol'];q=json.loads(lp.read_text());r=json.loads((P/q['output']).read_text());a=json.loads((P/link['audit']).read_text());assert r['protocol_sha256']==sha(lp)==a['protocol_sha256'];assert a['passed'] and (r['all_requested_steps_accepted'] or link.get('allow_verified_prefix',False) and a['accepted_prefix_only'])
 f=np.load(P/q['fields'],mmap_mode='r');assert len(f)==len(r['records'])==len(a['records']);assert sha(P/q['fields'])==a['fields_sha256'] and sha(P/q['checkpoint'])==r['probe_checkpoint_sha256']
 for j,(row,proof) in enumerate(zip(r['records'],a['records'])):
  i=len(steps);assert row['step']==i and row['accepted'] and row['SNES_reason']>0 and proof['native_residual_N']<1e-8
  c[i]=f[j];nodes[i]=f[j][node_map];steps.append(dict(step=i,time_s=row['time_s'],SNES_reason=row['SNES_reason'],residual_l2_N=proof['native_residual_N'],nonlinear_iterations=row.get('nonlinear_iterations'),nonlinear_solve_seconds=row.get('seconds',row.get('nonlinear_seconds')),inherited_probe=link['protocol'],independent_audit=link['audit'],independent_audit_sha256=sha(P/link['audit'])))
  if (i+1)%8==0:
   c.flush();nodes.flush()
   for x in [c,nodes,f]:
    if hasattr(x._mmap,'madvise'):x._mmap.madvise(mmap.MADV_DONTNEED)
  positivity.append(dict(step=i,time_s=row['time_s'],**{s:dict(passed=True,minimum_Bernstein_bound=proof[s+'_minimum_Bernstein_bound'],original_unresolved_elements=proof.get('original_'+s+'_unresolved_elements',0),detailed_audit=link['audit'],detailed_audit_sha256=sha(P/link['audit']),audit_record=j) for s in ['endpoint','stress']}))
 cp=np.load(P/q['checkpoint']);state={k:cp[k].copy() for k in ['u','u_old','v_old','a_old']};lineage.append(dict(**link,protocol_sha256=sha(lp),output_sha256=sha(P/q['output']),audit_sha256=sha(P/link['audit']),checkpoint_sha256=sha(P/q['checkpoint'])))
count=len(steps);assert count==p['seed_steps'] and len(positivity)==count;np.testing.assert_array_equal(state['u_old'],c[count-1]);c.flush();nodes.flush();shutil.copyfile(source/'inputs.npz',root/'inputs.npz');np.savez_compressed(root/'restart.npz',completed_steps=count,**state)
for i,x in enumerate(positivity):assert x['step']==i and all(x[s]['passed'] and x[s]['minimum_Bernstein_bound']>1e-9 for s in ['endpoint','stress'])
atomic_json(root/'positivity.json',positivity);atomic_json(root/'manifest.json',dict(complete=False,completed_steps=count,n_steps=n,config=cfg,config_sha256=sha(P/p['config']),steps=steps,dofs=ndof,tetrahedra=len(inputs['cells']),seed_steps=count,prefix_positivity_passed=True,prefix_independent_native_audit_passed=False,source_reference=p['source_reference'],seed_protocol=args.protocol,seed_protocol_sha256=sha(pp),source_manifest_sha256=sha(source/'manifest.json'),source_inputs_sha256=sha(source/'inputs.npz'),source_prefix_count=base,source_prefix_data_sha256=digest.hexdigest(),lineage=lineage,resource_protocol=p['resource_protocol'],cumulative_native_wall_seconds=0.))
print('NEW REGULARIZED REFERENCE SEEDED',cfg['name'],count,flush=True)
