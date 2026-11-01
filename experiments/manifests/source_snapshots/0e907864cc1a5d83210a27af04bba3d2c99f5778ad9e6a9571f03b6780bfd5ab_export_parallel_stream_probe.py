"""Expose only newly solved pilot rows for the independent legacy native verifier."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args()
p=json.loads((P/args.protocol).read_text());sp=json.loads((P/p['stream_protocol']).read_text());cfg=json.loads((P/sp['config']).read_text())
root=P/'data/processed'/cfg['name'];m=json.loads((root/'manifest.json').read_text());start=p['qualification_step'];count=p['steps_after_checkpoint']
assert m['pilot_complete'] and m['completed_steps']==start+count and not(root/'failure.json').exists()
source=P/'data/processed'/p['reference'];sm=json.loads((source/'manifest.json').read_text())
u=np.load(root/'displacement_coefficients.npy',mmap_mode='r');np.save(P/p['fields'],u[start:start+count])
cp=np.load(root/'restart.npz');np.savez_compressed(P/p['checkpoint'],completed_probe_steps=count,**{k:cp[k] for k in ['u','u_old','v_old','a_old']})
def sha(x):return hashlib.sha256(x.read_bytes()).hexdigest()
records=[dict(step=r['step'],time_s=r['time_s'],accepted=True,SNES_reason=r['SNES_reason'],residual_N=r['residual_l2_N']) for r in m['steps'][start:]]
out=dict(complete=True,finished=True,all_requested_steps_accepted=True,records=records,start_step=start,dt_s=cfg['dt_s'],
 full_cycle=False,changed_time_discretization=False,qualification_only=True,protocol_sha256=sha(P/args.protocol),
 checkpoint_sha256=sha(source/'restart.npz'),probe_checkpoint_sha256=sha(P/p['checkpoint']),reference_config_sha256=sm['config_sha256'],
 source_sha256=sha(Path(__file__)),stream_manifest_sha256=sha(root/'manifest.json'),
 scope='Adapter exposes actual newly solved streaming-pilot fields/checkpoint to the independent serial known-reference verifier; inherited rows are not new solves, full-cycle completion is not claimed.')
(P/p['output']).write_text(json.dumps(out,indent=2)+'\n')
