"""Read-only replay qualification for real coarse and loaded multi-step checkpoints."""
from pathlib import Path
import json,sys,hashlib
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.probe_history_replay import load_parent
records=[]
for stem in ['dynamic_quadratic_mpi12_threaded_coarse_v1','dynamic_quadratic_mpi_admissible_consecutive_v1']:
 audit='experiments/metrics/'+stem+'_verification'+('' if 'coarse' in stem else '_resolved')+'.json'
 current,previous,step,metadata=load_parent(P,'configs/physics/'+stem+'.json',audit)
 assert metadata['parent_steps_replayed']==2 and metadata['all_fields_causally_replayed']
 records.append(dict(parent=stem,next_step=step,metadata=metadata))
result=dict(complete=True,passed=True,records=records,source_sha256=hashlib.sha256((P/'src/lv_surrogate/physics/probe_history_replay.py').read_bytes()).hexdigest(),scope='Hash-linked independent parent audit plus causal Newmark replay of every parentfield; final histories match nativecheckpoint,previous histories refer to immediately preceding physical state. Read-only initialization qualification,not a new physical solve or fullcycle/numerical validation.')
out=P/'experiments/metrics/dynamic_quadratic_parent_history_replay_qualification.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print('MULTI-STEP PARENT HISTORY REPLAY PASSED',flush=True)
