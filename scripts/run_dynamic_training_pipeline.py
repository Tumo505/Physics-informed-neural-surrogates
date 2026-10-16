"""Wait for reference train/selection data; run gated full and limited studies."""
from pathlib import Path
import json,subprocess,sys,time,os
P=Path(__file__).resolve().parents[1];cfg=json.loads((P/'configs/physics/dynamic_study.json').read_text());cases=[c for c in cfg['cases'] if c['split'] in ['train','validation']]
while True:
 ready=[]
 for c in cases:
  path=P/f'data/processed/cardiac_dynamic_v1/{c["id"]}/manifest.json'
  try:ready.append(json.loads(path.read_text()).get('complete',False))
  except (FileNotFoundError,json.JSONDecodeError):ready.append(False)
 if all(ready):break
 print('WAITING TRAIN/VALIDATION FE',sum(ready),'/',len(ready),flush=True);time.sleep(30)
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2');py=str(P/'.venv/bin/python')
for cmd in [[py,'scripts/verify_dynamic_backend.py'],*[x for regime in ['full','limited'] for x in [[py,'scripts/prepare_dynamic_study.py','--regime',regime],[py,'scripts/train_dynamic_study.py','--regime',regime]]]]:
 subprocess.run(cmd,cwd=P,env=env,check=True)
print('BOTH DYNAMIC STUDIES COMPLETE',flush=True)
