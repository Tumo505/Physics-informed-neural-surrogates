"""Collect solver provenance without fitting models or inspecting test errors."""
from pathlib import Path
import json,hashlib,numpy as np,argparse
P=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--require-complete',action='store_true');args=ap.parse_args();cfg=json.loads((P/'configs/physics/dynamic_study.json').read_text());records=[];failed=[]
for c in cfg['cases']:
 if c['split']=='failed_timestep':continue
 root=P/f'data/processed/cardiac_dynamic_v1/{c["id"]}';path=root/'manifest.json'
 if not path.exists():records.append({'case':c,'complete':False,'completed_steps':0});continue
 m=json.loads(path.read_text());assert m['case']==c;assert all(np.isfinite(s['residual_l2_N']) and s['residual_l2_N']<1e-6 for s in m['steps'])
 if m['complete']:
  for key,file in [('trajectory_sha256','trajectory.npz'),('geometry_sha256','geometry.npz')]:assert hashlib.sha256((root/file).read_bytes()).hexdigest()==m[key]
 rec={'case':c,'complete':m['complete'],'completed_steps':m['completed_steps'],'seconds':m['seconds'],'maximum_residual_l2_N':max(s['residual_l2_N'] for s in m['steps']),'minimum_recorded_endpoint_detF':min((s.get('minimum_detF',1.) for s in m['steps'])),'trajectory_sha256':m.get('trajectory_sha256'),'geometry_sha256':m.get('geometry_sha256'),'source_sha256':m['source_sha256'],'source_hash_scope':m.get('source_hash_capture','checkpoint-time source snapshot; solver retry/metadata options evolved during cohort; fixed physical case and external source commit unchanged')};records.append(rec)
for path in (P/'data/processed/cardiac_dynamic_v1').glob('*failed_dt001/manifest.json'):
 m=json.loads(path.read_text());failed.append({'case':m['case'],'recorded_steps':m['completed_steps'],'failure':'10ms attempt failed independent residual/Jacobian audit; entire timestep protocol rejected before training, shapes retained for5ms retry'})
coarse=P/'data/processed/cardiac_dynamic_v1/convergence_dt0.01/manifest.json'
if coarse.exists():failed.append({'case':json.loads(coarse.read_text())['case'],'failure':'10ms full-cycle baseline diverged; retained, not a converged reference'})
complete=all(r['complete'] for r in records);result={'complete':complete,'planned_accepted_trajectories':len(records),'completed_accepted_trajectories':sum(r['complete'] for r in records),'recorded_valid_steps':sum(r['completed_steps'] for r in records),'records':records,'failed_10ms_attempts':failed,'scope':cfg['scope'],'split_protocol':cfg['protocol'],'config_sha256':hashlib.sha256((P/'configs/physics/dynamic_study.json').read_bytes()).hexdigest()};(P/'experiments/manifests/cardiac_dynamic_v1.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ['records','failed_10ms_attempts','split_protocol']},flush=True)
if args.require_complete:assert complete
