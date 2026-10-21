"""Create a separate traced reference; preserve originals and lost-reason provenance."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args()
p=json.loads((P/args.protocol).read_text());cfg=json.loads((P/p['config']).read_text())
source=P/'data/processed'/p['source_reference'];sm=json.loads((source/'manifest.json').read_text());sc=sm['config']
for key in sc:
 if key not in ['name','scope']:assert cfg[key]==sc[key],key
for path in p.get('required_audits',[]):
 a=json.loads((P/path).read_text());assert a['complete'] and a['passed'],path
root=P/'data/processed'/cfg['name'];root.mkdir(parents=True,exist_ok=True);mp=root/'manifest.json'
if mp.exists():
 m=json.loads(mp.read_text());assert m['config_sha256']==sha(P/p['config']) and m['seed_protocol_sha256']==sha(P/args.protocol)
 print('REUSE PARALLEL REFERENCE SEED',m['completed_steps'],flush=True);raise SystemExit(0)
assert shutil.disk_usage(P).free>100*2**30
inputs=np.load(source/'inputs.npz');n=len(inputs['time']);ndof=len(inputs['dof_coordinates']);count=p['seed_steps']
original_count=min(count,p.get('original_native_accepted_steps',count))
old=np.load(source/'displacement_coefficients.npy',mmap_mode='r')
coeff=np.lib.format.open_memmap(root/'displacement_coefficients.npy',mode='w+',dtype=np.float64,shape=(n,ndof))
node_map=np.full((len(inputs['coords']),3),-1,dtype=np.int64)
for c in range(3):
 for j in range(4):node_map[inputs['cells'][:,j],c]=inputs['canonical_cell_dofs'][:,j,c]
assert (node_map>=0).all()
nodes=np.lib.format.open_memmap(root/'displacements.npy',mode='w+',dtype=np.float64,shape=(n,len(node_map),3))
digest=hashlib.sha256()
for i in range(original_count):
 coeff[i]=old[i];nodes[i]=old[i][node_map];digest.update(np.asarray(old[i]).tobytes())
steps=list(sm['steps'][:original_count]);lineage=[]
if len(steps)<original_count:
 assert p.get('recovered_tail_protocol')
 tp=json.loads((P/p['recovered_tail_protocol']).read_text());tr=json.loads((P/tp['output']).read_text())
 ta=json.loads((P/p['recovered_tail_audit']).read_text());assert ta['passed'];tail=np.load(P/tp['fields'],mmap_mode='r')
 failure=json.loads((source/'failure.json').read_text());assert failure['accepted_steps_before_failure']>=original_count
 start=len(steps);assert tr['start_step']==start and len(tail)==original_count-start
 for j,r in enumerate(ta['records']):
  i=start+j;np.testing.assert_array_equal(coeff[i],tail[j]);assert r['native_residual_N']<1e-8
  steps.append(dict(step=i,time_s=float(inputs['time'][i]),SNES_reason=None,residual_l2_N=r['native_residual_N'],
   native_reason_record_lost=True,native_acceptance_provenance=dict(
    failure_path=str((source/'failure.json').relative_to(P)),failure_sha256=sha(source/'failure.json'),
    independent_audit=p['recovered_tail_audit'],independent_audit_sha256=sha(P/p['recovered_tail_audit']),audit_record=j,
    accepted_fields=tp['fields'],accepted_fields_sha256=sha(P/tp['fields']))))
for relative in p.get('accepted_probe_protocols',[]):
 pp=json.loads((P/relative).read_text());r=json.loads((P/pp['output']).read_text())
 apath=Path(pp['output']).with_name(Path(pp['output']).stem+'_verification_resolved.json')
 a=json.loads((P/apath).read_text());assert a['passed'];fields=np.load(P/pp['fields'],mmap_mode='r')
 assert sha(P/pp['fields'])==a['fields_sha256'] and sha(P/pp['checkpoint'])==r['probe_checkpoint_sha256']
 for j,row in enumerate(a['records']):
  i=len(steps);record=r['records'][j];assert record['accepted'] and record['step']==i
  coeff[i]=fields[j];nodes[i]=fields[j][node_map]
  steps.append(dict(step=i,time_s=record['time_s'],SNES_reason=record['SNES_reason'],residual_l2_N=row['native_residual_N'],
   nonlinear_solve_seconds=record['nonlinear_seconds'],inherited_probe=relative,independent_audit=str(apath),independent_audit_sha256=sha(P/apath)))
 lineage.append(dict(protocol=relative,protocol_sha256=sha(P/relative),audit=str(apath),audit_sha256=sha(P/apath)))
assert len(steps)==count;coeff.flush();nodes.flush();shutil.copyfile(source/'inputs.npz',root/'inputs.npz')
if p.get('accepted_probe_protocols'):
 last=json.loads((P/p['accepted_probe_protocols'][-1]).read_text());cp=np.load(P/last['checkpoint'])
 state={k:cp[k] for k in ['u','u_old','v_old','a_old']}
else:
 uold=np.zeros(ndof);v=np.zeros(ndof);a=np.zeros(ndof);dt=cfg['dt_s']
 for u in coeff[:count]:
  anew=(u-uold-dt*v-(.5-.36)*dt**2*a)/(.36*dt**2)
  v=v+dt*(.3*a+.7*anew);a=anew;uold=u.copy()
 state=dict(u=uold.copy(),u_old=uold,v_old=v,a_old=a)
np.testing.assert_array_equal(state['u_old'],coeff[count-1]);np.savez_compressed(root/'restart.npz',completed_steps=count,**state)
m=dict(complete=False,completed_steps=count,n_steps=n,config=cfg,config_sha256=sha(P/p['config']),steps=steps,
 dofs=ndof,tetrahedra=len(inputs['cells']),seed_steps=count,source_reference=p['source_reference'],
 seed_protocol=args.protocol,seed_protocol_sha256=sha(P/args.protocol),prefix_positivity_passed=False,
 source_prefix_count=original_count,source_prefix_data_sha256=digest.hexdigest(),
 source_manifest_sha256=sha(source/'manifest.json'),source_inputs_sha256=sha(source/'inputs.npz'),lineage=lineage,
 seed_source_sha256=sha(Path(__file__)),positivity_scope='Seed is native-accepted/traced; its entire inherited prefix must pass whole-element positivity before new generation.')
mp.write_text(json.dumps(m,indent=2)+'\n');print('PARALLEL SEED CREATED',cfg['name'],count,flush=True)
