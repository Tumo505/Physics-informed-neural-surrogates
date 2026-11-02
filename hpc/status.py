"""Gateway-safe registered progress; initialization is distinct from job failure."""
import argparse,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'hpc'))
from registry import read
from manuscript import render

def read_status(root,run):
 root=Path(root);run=Path(run)
 if not run.is_absolute():run=root/run
 state=run/'state.json'
 if state.exists():return json.loads(state.read_text()),read(run),True
 launch=root/'.hpc/launches'/run.name
 if not launch.is_dir():return dict(status='not_found',message='No initialized run or registered launch found. Check the run ID.'),[],False
 events=read(launch);submissions=[x for x in events if x['kind']=='job_submission']
 summary=dict(status='awaiting_initialization',message='The first reference slice creates state.json after setup succeeds. Scheduler submission alone does not establish execution or scientific progress.',submitted_jobs=[dict(role=x['role'],job_id=x['stdout'].split()[0]) for x in submissions if x['return_code']==0],scientific_progress=None)
 failures=[x for x in submissions if x['return_code']!=0]
 if failures:summary.update(status='submission_failed',submission_errors=[x['stderr'] for x in failures])
 setup=[x for x in submissions if x['role']=='setup' and x['return_code']==0]
 if setup:
  identifier=setup[-1]['stdout'].split()[0];result=root/'.hpc'/('setup-result-'+identifier+'.json')
  if result.exists():
   summary['setup_result']=json.loads(result.read_text())
   if summary['setup_result']['exit_code']!=0:summary.update(status='setup_failed',message='Setup exited unsuccessfully; dependent jobs cannot proceed. Preserve its logs and inspect the failure.')
 return summary,events,False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=P/a.run;s,events,initialized=read_status(P,run)
 print(json.dumps(s,indent=2));print('Verified ledger events:',len(events))
 if initialized and (run/'plan.json').exists():print('Manuscript:',render(P,run))
 else:print('Scheduler check: qstat -u "$USER". Do not re-launch an already submitted chain.')
if __name__=='__main__':main()
