"""Verify explicit permission and exact causal replay for a failed-window prefix."""
from pathlib import Path
import sys,json,hashlib
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.probe_history_replay_prefix import load_parent
p='configs/physics/dynamic_quadratic_multistep_window_v1.json';a='experiments/metrics/dynamic_quadratic_multistep_window_v1_verification_resolved.json'
denied=False
try:load_parent(P,p,a)
except AssertionError:denied=True
assert denied
current,previous,step,meta=load_parent(P,p,a,allow_accepted_prefix=True)
assert step==451 and meta['parent_steps_replayed']==7 and meta['accepted_prefix_only_consumed'] and not meta['parent_original_requested_steps_completed']
result=dict(complete=True,passed=True,implicit_prefix_consumption_rejected=True,next_step=step,metadata=meta,source_sha256=hashlib.sha256((P/'src/lv_surrogate/physics/probe_history_replay_prefix.py').read_bytes()).hexdigest(),scope='Only the independently verified actual7-state prefix is consumed with explicit permission; original8-stepwindow remainsFAILED. Nativecurrentcheckpoint histories retained aftercausalreplay; immediateprevioushistory recovered. No new physical solve or label generated.')
out=P/'experiments/metrics/dynamic_quadratic_verified_prefix_replay_qualification.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print('EXPLICIT VERIFIED-PREFIX REPLAY PASSED',flush=True)
