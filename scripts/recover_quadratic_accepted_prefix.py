"""Recover native-accepted, uncheckpointed fields without accepting the failed iterate."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
cp=P/args.config;cfg=json.loads(cp.read_text());root=P/'data/processed'/cfg['name']
failed=json.loads((root/'failure.json').read_text());m=json.loads((root/'manifest.json').read_text())
a=int(m['completed_steps']);b=int(failed['accepted_steps_before_failure']);assert a<b
assert failed['config_sha256']==m['config_sha256']==hashlib.sha256(cp.read_bytes()).hexdigest()
snapshot=root/'UNACCEPTED_nonlinear_iterate.npz';snap=dict(np.load(snapshot));assert not bool(snap['accepted']) and int(snap['step'])==b
assert str(snap['config_sha256'])==m['config_sha256']
saved=dict(np.load(root/'restart.npz'));assert int(saved['completed_steps'])==a
inputs=np.load(root/'inputs.npz');u=np.load(root/'displacement_coefficients.npy',mmap_mode='r')
stem=cfg['name']+'_accepted_tail';output=P/f'experiments/metrics/{stem}.json';protocol_path=P/f'configs/physics/{stem}.json'
assert not output.exists() and not protocol_path.exists()
fields_path=P/f'experiments/logs/{stem}_fields.npy';np.save(fields_path,np.asarray(u[a:b]))
dt=cfg['dt_s'];old=saved['u_old'].copy();v=saved['v_old'].copy();acc=saved['a_old'].copy()
# Pinned benchmark generalized-alpha defaults: alpha_f=.4, alpha_m=.2.
gamma=.5+.4-.2;beta=(gamma+.5)**2/4
for field in u[a:b]:
 new_a=(field-(old+dt*v+(.5-beta)*dt**2*acc))/(beta*dt**2)
 v=v+(1-gamma)*dt*acc+gamma*dt*new_a;acc=new_a;old=field.copy()
checks={}
for name,value in [('u_old',old),('v_old',v),('a_old',acc)]:
 np.testing.assert_allclose(value,snap[name],rtol=1e-7,atol=1e-10)
 checks[name]=dict(maximum_absolute_difference=float(abs(value-snap[name]).max()),relative_L2_difference=float(np.linalg.norm(value-snap[name])/max(np.linalg.norm(snap[name]),1e-30)))
checkpoint=P/f'experiments/logs/{stem}_restart.npz'
protocol=dict(reference=cfg['name'],timestep_divisor=1,steps_after_checkpoint=b-a,output=str(output.relative_to(P)),checkpoint=str(checkpoint.relative_to(P)),fields=str(fields_path.relative_to(P)),scope='Recover only the native-accepted coefficient rows after the durable checkpoint and before the explicitly failed step. Native failure metadata establishes acceptance count; individual lost SNES records are not reconstructed. Independently reassemble all forces, whole-element positivity and norms before any reuse. Never use the failed iterate as an accepted field. Histories independently reconstructed with pinned generalized-alpha constants and checked against the fixed histories stored with the failed iterate.')
protocol_path.write_text(json.dumps(protocol,indent=2)+'\n');ph=hashlib.sha256(protocol_path.read_bytes()).hexdigest()
np.savez_compressed(checkpoint,completed_probe_steps=b-a,protocol_sha256=ph,original_checkpoint_sha256=hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest(),u=snap['u_old'],**{n:snap[n] for n in ['u_old','v_old','a_old']})
result=dict(complete=True,finished=True,engineering_only=True,full_cycle=False,all_requested_steps_accepted=True,acceptance_provenance='Native failure accepted_steps_before_failure; independent audit required, no fabricated per-step SNES records',protocol=protocol,protocol_sha256=ph,reference_config_sha256=m['config_sha256'],checkpoint_sha256=hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest(),probe_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),failed_iterate_snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),native_failure_sha256=hashlib.sha256((root/'failure.json').read_bytes()).hexdigest(),start_step=a,dt_s=dt,changed_time_discretization=False,independent_history_reconstruction=checks,records=[dict(step=i,time_s=float(inputs['time'][i]),accepted=True,native_acceptance_count_provenance=True) for i in range(a,b)])
output.write_text(json.dumps(result,indent=2,allow_nan=False));print('RECOVERED UNQUALIFIED TAIL',a,b,'requires independent audit',flush=True)
