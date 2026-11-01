"""Independent serial reassembly, norms, histories and whole-element proof on known MPI outputs."""
import argparse,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P/'src'),str(P/'hpc')]
from lv_surrogate.physics.parallel_fem import ParallelP2
from lv_surrogate.physics.reference_fem import residual
from lv_surrogate.physics.bernstein import BernsteinAudit
from platform_native import norm_check
from registry import atomic,sha
ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--probe',required=True);ap.add_argument('--fields',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();s=json.loads((P/a.spec).read_text());snapshot=json.loads((P/s['snapshot']).read_text());root=P/'data/processed'/snapshot['reference'];m=json.loads((root/'manifest.json').read_text());data=np.load(root/'inputs.npz');adapter=ParallelP2(P,snapshot['source_reference'],m['config']);assert adapter.comm.size==1;proof=json.loads((P/a.probe).read_text());assert proof['complete'] and proof['passed'] and proof['fields_sha256']==sha(P/a.fields);cp=np.load(P/s['known_start_checkpoint']);start=int(cp['completed_steps']);f=np.load(P/a.fields);assert len(f)==2;saved=np.load(P/a.checkpoint);out=P/a.output;assert not out.exists()
for k in ['u_old','v_old','a_old']:adapter.assign(k,cp[k])
audit=BernsteinAudit(data);records=[]
for j,u in enumerate(f):
 i=start+j;adapter.assign('u',u);adapter.problem.parameters['p'].assign(float(data['pressure'][i]));force=float(residual(adapter.problem).norm('l2'));assert np.isfinite(force) and force<1e-8;norm=norm_check(adapter,data,u);old=adapter.gather('u_old');positive=audit.pair(u,old);records.append(dict(step=i,native_residual_N=force,independent_norm_relative_error=norm,whole_element_proof=positive));adapter.problem._update_fields()
errors={}
for k in ['u','u_old','v_old','a_old']:
 actual=adapter.gather(k);np.testing.assert_allclose(actual,saved[k],rtol=1e-7,atol=1e-10);errors[k]=float(np.linalg.norm(actual-saved[k])/max(np.linalg.norm(saved[k]),1e-30))
atomic(out,dict(complete=True,passed=True,known_field_only=True,new_physical_state_generated=False,records=records,history_relative_errors=errors,probe_sha256=sha(P/a.probe),fields_sha256=sha(P/a.fields),checkpoint_sha256=sha(P/a.checkpoint),scope='Independent serialnativeforce/norm/history/whole-element qualification of two knownMPIequations only;notfullcycleorMLvalidation.'))
