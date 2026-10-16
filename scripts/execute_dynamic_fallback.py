"""Route an out-of-support request to an independent, measured FE trajectory."""
from pathlib import Path
import json,sys,subprocess,os,time
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.eval.fallback import InputDomainGate
cfg=json.loads((P/'configs/physics/dynamic_study.json').read_text());case=next(c for c in cfg['cases'] if c['split']=='shift');bounds={'radial':(.95,1.05),'longitudinal':(.95,1.05),'amplitude':(.9,1.1),'timing':(.95,1.05)};gate=InputDomainGate(bounds);inputs=dict(zip(['radial','longitudinal','amplitude','timing'],[*case['scales'],case['amplitude'],case['timing_scale']]))
def fem():
 env=os.environ.copy();env.update(FI_PROVIDER='tcp',FI_TCP_IFACE='en0',XDG_CACHE_HOME=str(P/'.cache'),DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'),MAMBA_ROOT_PREFIX=str(P/'.mamba'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');start=time.perf_counter();subprocess.run([str(P/'.tools/bin/micromamba'),'run','-p',str(P/'.fenics'),'python',str(P/'scripts/generate_dynamic_study.py'),'--case',case['id'],'--output-suffix','_fallback'],env=env,cwd=P,check=True);return time.perf_counter()-start
cached=(P/f'data/processed/cardiac_dynamic_v1/{case["id"]}_fallback/trajectory.npz').exists();decision,elapsed=gate.dispatch(inputs,lambda:(_ for _ in ()).throw(AssertionError('Shifted request must not reach NN')),fem);result={'case':case,'decision':decision,'wall_seconds':elapsed,'cache_existed_before_request':cached,'scope':'one actual input-gated full-cycle FE fallback; solver receives only geometry/loading descriptors and zero initial histories, no reference labels'};(P/'experiments/metrics/dynamic_fallback_execution.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
