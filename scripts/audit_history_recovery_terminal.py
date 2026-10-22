"""Preserve failed homotopy evidence; no solve or physical endpoint promotion."""
import argparse,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.bernstein import BernsteinAudit
import hashlib

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args()
pp=P/args.protocol;p=json.loads(pp.read_text());rp=P/p['output'];r=json.loads(rp.read_text())
assert r['finished'] and not r['all_requested_steps_accepted'] and not r['records']
assert sha(pp)==r['protocol_sha256']
assert all(sha(P/k)==h for k,h in p['frozen_source_sha256'].items())
parent=json.loads((P/p['resume_probe_protocol']).read_text());prior=json.loads((P/p['previous_history_protocol']).read_text())
cp=np.load(P/parent['checkpoint']);previous=np.load(P/prior['checkpoint'])
assert sha(P/parent['checkpoint'])==r['resumed_checkpoint_sha256'] and sha(P/prior['checkpoint'])==r['previous_history_checkpoint_sha256']
paths=[pp,rp,P/parent['checkpoint'],P/prior['checkpoint']];checks=[]
for key in ['engineering_checkpoint','iterate_checkpoint']:
 path=P/p[key];x=np.load(path);paths.append(path);fraction=float(x['fraction'])
 assert int(x['step'])==r['start_step'] and 0<fraction<1
 assert not bool(x['physical_label' if key=='engineering_checkpoint' else 'accepted'])
 for name in ['u_old','v_old','a_old']:
  expected=(1-fraction)*previous[name]+fraction*cp[name]
  np.testing.assert_allclose(x[name],expected,rtol=1e-13,atol=1e-14)
 check=dict(checkpoint=key,fraction=fraction,physical_label=False,exact_interpolated_histories_passed=True)
 if key=='engineering_checkpoint':
  proof=BernsteinAudit(np.load(P/'data/processed'/p['reference']/'inputs.npz')).pair(x['u'],x['u_old'])
  last=r['engineering_roots'][-1];assert fraction==last['fraction']
  for state in ['endpoint','stress']:
   np.testing.assert_allclose(proof[state]['minimum_Bernstein_bound'],last['proof'][state]['minimum_Bernstein_bound'],rtol=0,atol=1e-14)
  check['replayed_whole_element_proof']=proof
  matching=[a for a in r['attempts'] if a['fraction']==fraction and a['accepted_engineering_root']];assert matching
  check['recorded_native_force_N']=matching[-1]['native_residual_N'];check['recorded_SNES_reason']=matching[-1]['SNES_reason']
 checks.append(check)
freeze=P/'experiments/manifests/dynamic_final_freeze.json';f=json.loads(freeze.read_text());assert all(sha(P/k)==h for k,h in f['checkpoint_sha256'].items())
out=rp.with_name(rp.stem+'_terminal_integrity.json');assert not out.exists()
result=dict(complete=True,passed=True,recovery_passed=False,full_cycle=False,checks=checks,
 evidence_sha256={str(path.relative_to(P)):sha(path) for path in paths},source_sha256=sha(Path(__file__)),all20_original_models_unchanged=True,
 scope='Read-only terminal checkpoint/hash/history interpolation and deterministic whole-element proof replay. Recorded native norms/reasons are preserved, not newly independently assembled here. No exact fraction1 physical endpoint exists. Integrity pass is not nonlinear recovery or scientific validation.')
out.write_text(json.dumps(result,indent=2)+'\n');print('TERMINAL INTEGRITY PASSED; PHYSICAL RECOVERY FAILED',flush=True)
