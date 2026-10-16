"""Bounded independent native solver processes; no shared solver state."""
from pathlib import Path
import json,subprocess,os,sys,concurrent.futures,argparse
P=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['cohort','convergence','confirmation'],required=True);ap.add_argument('--config',default='configs/physics/dynamic_study.json');ap.add_argument('--workers',type=int,default=3);args=ap.parse_args();cfg=json.loads((P/args.config).read_text())
def run(c):
 env=os.environ.copy();env.update(FI_PROVIDER='tcp',FI_TCP_IFACE='en0',XDG_CACHE_HOME=str(P/'.cache'),DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'),MAMBA_ROOT_PREFIX=str(P/'.mamba'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');log=P/f'experiments/logs/dynamic_{c["id"]}.log'
 with log.open('w') as f:r=subprocess.run([str(P/'.tools/bin/micromamba'),'run','-p',str(P/'.fenics'),'python',str(P/'scripts/generate_dynamic_study.py'),'--case',c['id'],'--config',args.config],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=P)
 print(c['id'],'exit',r.returncode,flush=True)
 if r.returncode:raise RuntimeError(str(log))
 return c['id']
with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run,[c for c in cfg['cases'] if (c['split']=='confirmation' if args.group=='confirmation' else c['split']=='convergence' if args.group=='convergence' else c['split'] in ['train','validation','calibration','test','load_shift','shift'])]))
print('GROUP COMPLETE',args.group,flush=True)
