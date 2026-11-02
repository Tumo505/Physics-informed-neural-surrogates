"""Carry spent resources from the preserved, non-generating platform timeout."""
import json,re
from pathlib import Path
from registry import sha,read

def prior_budget(root,run_id,snapshot_sha256):
 if not run_id:return dict(cumulative_seconds=0.,generation_seconds=0.)
 assert re.fullmatch('[A-Za-z0-9_-]+',run_id),'Invalid prior run ID'
 root=Path(root);parent=root/'experiments/hpc'/run_id
 state=json.loads((parent/'state.json').read_text());plan=json.loads((parent/'plan.json').read_text());events=read(parent)
 records=[r for r in events if r['kind']=='artifact' and r['path']==str((parent/'plan.json').relative_to(root))]
 assert records and records[-1]['sha256']==sha(parent/'plan.json'),'Prior immutable plan changed'
 assert plan['snapshot_sha256']==snapshot_sha256,'Prior source snapshot differs'
 assert state['status']=='blocked' and state['terminal_rejection'] and state['phase']=='platform'
 assert state['generation_seconds']==0 and state['completed_phases']==[],'This follow-up does not consume prior generated states'
 assert any('declared stage wall budget' in r for r in state['blockers']),'Only the preserved platform timeout is admitted by this follow-up'
 assert 0<=state['cumulative_seconds']<plan['cumulative_pipeline_budget_hours']*3600
 return dict(prior_run=run_id,cumulative_seconds=state['cumulative_seconds'],generation_seconds=0.,prior_state_sha256=sha(parent/'state.json'),prior_plan_sha256=sha(parent/'plan.json'),prior_ledger_tail=events[-1]['event_sha256'])
