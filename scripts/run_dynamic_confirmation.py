"""Bound added compute: wait for primary cohort, then generate confirmation."""
from pathlib import Path
import json,time,subprocess,sys,hashlib
P=Path(__file__).resolve().parents[1];cfg=json.loads((P/'configs/physics/dynamic_study.json').read_text());cases=[c for c in cfg['cases'] if c['split'] not in ['convergence','failed_timestep']];freeze=json.loads((P/'experiments/manifests/dynamic_final_freeze.json').read_text());config=P/'configs/physics/dynamic_confirmation.json';declaration={'checkpoint_sha256':freeze['checkpoint_sha256'],'confirmation_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),'protocol':'independent additional test draws, models unchanged; no confirmation labels used for fitting/selection; same20 calibration scores/radii as primary evaluation'};(P/'experiments/manifests/dynamic_confirmation_freeze.json').write_text(json.dumps(declaration,indent=2))
while True:
 ready=0
 for c in cases:
  try:ready+=bool(json.loads((P/f'data/processed/cardiac_dynamic_v1/{c["id"]}/manifest.json').read_text()).get('complete'))
  except (FileNotFoundError,json.JSONDecodeError):pass
 if ready==len(cases):break
 print('WAIT PRIMARY COHORT BEFORE EXTRA COMPUTE',ready,'/',len(cases),flush=True);time.sleep(30)
subprocess.run([sys.executable,'scripts/run_dynamic_cohort.py','--group','confirmation','--workers','5','--config','configs/physics/dynamic_confirmation.json'],cwd=P,check=True)
print('CONFIRMATION REFERENCES COMPLETE',flush=True)
