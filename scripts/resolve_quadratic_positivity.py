"""Resolve inconclusive cubic Bernstein bounds by deterministic tetrahedral bisection."""
from pathlib import Path
import argparse,hashlib,itertools,json,math
import numpy as np
P=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--max-depth',type=int,default=12);ap.add_argument('--probe-protocol');args=ap.parse_args()
cp=P/args.config;cfg=json.loads(cp.read_text());root=P/'data/processed'/cfg['name']
mp=root/'manifest.json';m=json.loads(mp.read_text());bp=P/f'experiments/metrics/{cfg["name"]}_bernstein.json'
audit=json.loads(bp.read_text()) if not args.probe_protocol else None
initial_old=None;target_stem=cfg['name'];proof_hashes=None
if args.probe_protocol:
 protocol=json.loads((P/args.probe_protocol).read_text());assert protocol['reference']==cfg['name']
 bp=(P/protocol['output']).with_name(Path(protocol['output']).stem+'_verification.json')
 native_audit=json.loads(bp.read_text());assert native_audit['complete']
 fields=P/protocol['fields'];inputs=root/'inputs.npz'
 assert hashlib.sha256(fields.read_bytes()).hexdigest()==native_audit['fields_sha256']
 assert hashlib.sha256(inputs.read_bytes()).hexdigest()==native_audit['inputs_sha256']
 initial_old=np.load(root/'restart.npz')['u_old']
 assert hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest()==native_audit['original_checkpoint_sha256']
 assert protocol.get('timestep_divisor',1)==1 and 'qualification_step' not in protocol
 resumed_hash=None
 if protocol.get('resume_probe_protocol'):
  probe=json.loads((P/protocol['output']).read_text())
  assert probe['finished'] and probe['all_requested_steps_accepted']
  assert probe['protocol_sha256']==native_audit['protocol_sha256']==hashlib.sha256((P/args.probe_protocol).read_bytes()).hexdigest()
  parent=json.loads((P/protocol['resume_probe_protocol']).read_text())
  assert parent['reference']==protocol['reference'] and parent.get('timestep_divisor',1)==1
  parent_result=json.loads((P/parent['output']).read_text())
  resumed_path=P/parent['checkpoint'];resumed_hash=hashlib.sha256(resumed_path.read_bytes()).hexdigest()
  assert resumed_hash==probe['resumed_checkpoint_sha256']==parent_result['probe_checkpoint_sha256']
  parent_state=np.load(resumed_path);count=int(parent_state['completed_probe_steps'])
  assert all(r['accepted'] for r in parent_result['records'][:count])
  assert probe['start_step']==parent_result['start_step']+count
  initial_old=parent_state['u_old'].copy()
 audit=dict(complete=True,positive_margin=1e-9,records=[])
 for i,r in enumerate(native_audit['records']):
  row=dict(step=i,time_s=r['time_s'])
  for state in ['endpoint','stress']:
   row[state+'_unresolved_elements']=r[state+'_unresolved_elements']
   row[state+'_minimum_Bernstein_lower_bound']=r[state+'_minimum_Bernstein_bound']
  audit['records'].append(row)
 proof_hashes=dict(fields_sha256=native_audit['fields_sha256'],inputs_sha256=native_audit['inputs_sha256'])
 if resumed_hash is not None:proof_hashes['resumed_checkpoint_sha256']=resumed_hash
 target_stem=Path(protocol['output']).stem
else:
 assert m['complete'] and audit['complete'] and audit['output_sha256']==m['output_sha256']
assert m['config_sha256']==hashlib.sha256(cp.read_bytes()).hexdigest()
if not args.probe_protocol:
 for name,digest in m['output_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
u=np.load(fields if args.probe_protocol else root/'displacement_coefficients.npy',mmap_mode='r');data=np.load(root/'inputs.npz');X=data['coords'][data['cells']]
inv=np.linalg.inv((X[:,1:]-X[:,:1]).transpose(0,2,1));gl=np.concatenate((-inv.sum(1,keepdims=True),inv),axis=1)
edges=list(itertools.combinations(range(4),2));grads=np.stack([np.concatenate(((4*b[None,:,None]-1)*gl,np.stack([4*(b[i]*gl[:,j]+b[j]*gl[:,i]) for i,j in edges],axis=1)),axis=1) for b in np.eye(4)],axis=1)
multi=np.array([a for a in itertools.product(range(4),repeat=4) if sum(a)==3]);lattice=multi/3
mult=np.array([math.factorial(3)/math.prod(math.factorial(int(i)) for i in a) for a in multi])
def basis(b):return mult[None,:]*np.prod(np.asarray(b)[:,None,:]**multi[None,:,:],axis=-1)
transform=np.linalg.inv(basis(lattice));margin=audit['positive_margin'];records=[]
def resolve(F,physical):
 stack=[(np.eye(4),0)];low=float('inf');leaves=0;depth_seen=0
 while stack:
  vertices,depth=stack.pop();depth_seen=max(depth_seen,depth)
  points=lattice@vertices;values=np.linalg.det(np.einsum('qv,vij->qij',points,F));coeff=values@transform.T
  np.testing.assert_allclose(basis(lattice)@coeff,values,rtol=1e-11,atol=1e-11)
  if values.min()<=0:
   index=int(values.argmin());return dict(classification='inverted_witness',detF=float(values[index]),barycentric_witness=points[index].tolist(),depth=depth)
  if coeff.min()>margin:
   low=min(low,float(coeff.min()));leaves+=1;continue
  if depth==args.max_depth:return dict(classification='unresolved',depth=depth,minimum_sampled_detF=float(values.min()))
  positions=vertices@physical;edge=max(edges,key=lambda pair:float(np.sum((positions[pair[0]]-positions[pair[1]])**2)))
  a,b=edge;mid=(vertices[a]+vertices[b])/2
  first=vertices.copy();second=vertices.copy();first[a]=mid;second[b]=mid
  stack.extend([(first,depth+1),(second,depth+1)])
 return dict(classification='positive_bound',lower_bound=low,leaf_tetrahedra=leaves,maximum_depth=depth_seen)
for record in audit['records']:
 step=record['step']
 for state in ['endpoint','stress']:
  if not record[state+'_unresolved_elements']:continue
  field=u[step] if state=='endpoint' else .4*(u[step-1] if step else (initial_old if initial_old is not None else np.zeros(u.shape[1])))+.6*u[step]
  F=np.eye(3)+np.einsum('eki,evkj->evij',field[data['canonical_cell_dofs']],grads)
  coeff=np.linalg.det(np.einsum('qv,evij->eqij',lattice,F))@transform.T;lower=coeff.min(axis=1);indices=np.flatnonzero(lower<=margin)
  assert len(indices)==record[state+'_unresolved_elements']
  elements=[dict(element=int(i),**resolve(F[i],X[i])) for i in indices]
  positive_bounds=[float(lower[lower>margin].min())]+[e['lower_bound'] for e in elements if e['classification']=='positive_bound']
  records.append(dict(step=step,time_s=record['time_s'],state=state,original_unresolved_elements=len(indices),elements=elements,minimum_positive_lower_bound=min(positive_bounds)))
  print(step,state,[(e['element'],e['classification']) for e in elements],flush=True)
passed=all(e['classification']=='positive_bound' for r in records for e in r['elements'])
merged={}
for state in ['endpoint','stress']:
 values=[r[state+'_minimum_Bernstein_lower_bound'] for r in audit['records'] if not r[state+'_unresolved_elements']]
 values += [r['minimum_positive_lower_bound'] for r in records if r['state']==state]
 merged[state]=min(values)
out=dict(complete=True,passed=passed,reference_name=cfg['name'],config_sha256=m['config_sha256'],output_sha256=proof_hashes if args.probe_protocol else m['output_sha256'],original_audit_sha256=hashlib.sha256(bp.read_bytes()).hexdigest(),executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),positive_margin=margin,max_depth=args.max_depth,records=records,merged_minimum_lower_bound=merged if passed else None,scope='Supplement to the retained original inconclusive bound. Longest-rest-edge midpoint bisection partitions each unresolved tetrahedron; affine F is restricted to child vertices and the same cubic Bernstein criterion/margin is applied on every leaf. A nonpositive directly evaluated determinant is an inversion witness. Float64 algebra, not interval arithmetic; no relaxation of the physical positivity requirement.')
(P/f'experiments/metrics/{target_stem}_bernstein_refinement.json').write_text(json.dumps(out,indent=2,allow_nan=False));print('REFINED POSITIVITY PASSED',passed,flush=True)
