"""Wait for all solver/training gates; then freeze, evaluate and benchmark once."""
from pathlib import Path
import json,time,subprocess,os
P=Path(__file__).resolve().parents[1];cfg=json.loads((P/'configs/physics/dynamic_study.json').read_text());tcfg=json.loads((P/'configs/physics/dynamic_training.json').read_text());cases=[c for c in cfg['cases'] if c['split']!='failed_timestep'];py=str(P/'.venv/bin/python');env=os.environ.copy();env.update(FI_PROVIDER='tcp',FI_TCP_IFACE='en0',XDG_CACHE_HOME=str(P/'.cache'),MAMBA_ROOT_PREFIX=str(P/'.mamba'),DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2');fem=[str(P/'.tools/bin/micromamba'),'run','-p',str(P/'.fenics'),'python']
def complete(path):
 try:return bool(json.loads(path.read_text()).get('complete'))
 except (FileNotFoundError,json.JSONDecodeError):return False
while True:
 n=sum(complete(P/f'data/processed/cardiac_dynamic_v1/{c["id"]}/manifest.json') for c in cases);models=sum(complete(P/f'experiments/manifests/{tcfg["name"]}_{r}.json') for r in ['full','limited'])
 evaluated=complete(P/'experiments/metrics/dynamic_final_evaluation.json')
 confirmed=complete(P/'experiments/metrics/dynamic_confirmation_evaluation.json')
 if n==len(cases) and models==2 and evaluated and confirmed:break
 print('WAIT FINAL GATES',n,'/',len(cases),'reference trajectories;',models,'/2 ensemble studies; frozen evaluation complete',evaluated,'confirmation complete',confirmed,flush=True);time.sleep(30)
commands=[[py,'scripts/audit_dynamic_study.py','--require-complete'],[py,'scripts/analyze_dynamic_convergence.py'],[*fem,'scripts/analyze_dynamic_mesh.py'],[py,'scripts/benchmark_dynamic_study.py'],[py,'scripts/plot_dynamic_study.py'],[py,'scripts/plot_dynamic_confirmation.py'],[py,'scripts/verify_dynamic_artifacts.py'],[py,'scripts/write_reports.py']]
for cmd in commands:
 print('EXECUTE',cmd[-1],flush=True);subprocess.run(cmd,cwd=P,env=env,check=True)
print('DYNAMIC FINAL EXPERIMENTS COMPLETE',flush=True)
