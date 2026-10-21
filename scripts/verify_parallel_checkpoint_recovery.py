"""Controlled metadata-loss injection on the disposable two-step qualification only."""
import hashlib,json,os,subprocess
from pathlib import Path
P=Path(__file__).resolve().parents[1]
pp=P/'configs/physics/dynamic_quadratic_parallel_stream_coarse_v2.json';p=json.loads(pp.read_text());cfg=json.loads((P/p['config']).read_text())
assert cfg['name']=='cardiac_quadratic_parallel_stream_coarse_check_v2'
verified=json.loads((P/'experiments/metrics/dynamic_quadratic_parallel_stream_coarse_qualification_v2_verification.json').read_text());assert verified['complete'] and verified['passed']
root=P/'data/processed'/cfg['name'];mp=root/'manifest.json';original=json.loads(mp.read_text());assert original['completed_steps']==402 and original['pilot_complete']
assert not (P/'experiments/metrics/quadratic_parallel_checkpoint_recovery.json').exists()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
before={n:sha(root/n) for n in ['displacement_coefficients.npy','displacements.npy','restart.npz','positivity.json']}
(P/'experiments/logs/parallel_stream_manifest_before_injected_loss.json').write_text(json.dumps(original,indent=2))
damaged=dict(original,completed_steps=401,pilot_complete=False,steps=original['steps'][:-1]);mp.write_text(json.dumps(damaged,indent=2))
env=os.environ.copy();env.update(FI_PROVIDER='tcp',FI_TCP_IFACE='en0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 MAMBA_ROOT_PREFIX=str(P/'.mamba'),XDG_CACHE_HOME=str(P/'.cache'),DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'))
cmd=[str(P/'.tools/bin/micromamba'),'run','-p',str(P/'.fenics'),'mpiexec','-n','6','python','scripts/generate_parallel_reference.py','--protocol',str(pp.relative_to(P))]
with (P/'experiments/logs/quadratic_parallel_checkpoint_recovery.log').open('w') as log:subprocess.run(cmd,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120,check=True)
after=json.loads(mp.read_text());assert after['completed_steps']==402 and after['steps']==original['steps']
assert after['journal_recoveries'][-1]['step']==401 and not after['journal_recoveries'][-1]['physical_step_recomputed']
assert all(sha(root/n)==h for n,h in before.items())
result=dict(complete=True,passed=True,restored_step=401,native_acceptance_record_restored=True,
 coefficients_nodes_histories_and_positivity_bytes_unchanged=True,physical_steps_recomputed=0,
 source_sha256=sha(Path(__file__)),generator_sha256=sha(P/'scripts/generate_parallel_reference.py'),
 scope='Injected loss of the final manifest update after durable native checkpoint/journal on a disposable independently verified coarse2-step qualification. Original references/models untouched. Verified exact restoration of actual positive SNES record without fabrication or physical recomputation; not a guarantee against arbitrary storage failure.')
(P/'experiments/metrics/quadratic_parallel_checkpoint_recovery.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
