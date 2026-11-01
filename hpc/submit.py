"""Submit a bounded PBS chain; only successful/cleanly paused jobs release the next slice."""
import argparse,json,subprocess,sys,shlex
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'hpc'))
from registry import event
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--jobs',type=int,default=1);ap.add_argument('--after');ap.add_argument('--hours',type=int);ap.add_argument('--queue');ap.add_argument('--software',help='Override site PBS software resource label');a=ap.parse_args();assert 1<=a.jobs<=10;run=P/a.run;plan=json.loads((run/'plan.json').read_text());hours=a.hours or plan['job_hours'];queue=a.queue or plan['queue'];assert 1<=hours<=168;dependency=a.after;ids=[]
for _ in range(a.jobs):
 n=len(list(run.glob('job-*.pbs')))+1;script=run/f'job-{n:03d}.pbs';script.write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(P))+'\nexport OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nexec '+shlex.quote(str(P/'.venv/bin/python'))+' hpc/run.py --run '+shlex.quote(a.run)+' --slice-seconds '+str(hours*3600)+'\n')
 command=['qsub','-N','lv_'+plan['run_id'][:12],'-q',queue,'-l',f"select=1:ncpus={plan['cpus']}:mem={plan['memory_gb']}gb",'-l',f'walltime={hours:02d}:00:00','-j','oe','-o',str(run/f'pbs-{n:03d}.log')]
 software=a.software or plan.get('software','PBSJobApp')
 command+=['-l','software='+software]
 if plan.get('account'):command+=['-A',plan['account']]
 if dependency:command+=['-W','depend=afterok:'+dependency]
 command.append(str(script));result=subprocess.run(command,cwd=P,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 event(run,'job_submission',argv=command,return_code=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
 if result.returncode:raise SystemExit('PBS rejected submission: '+result.stderr.strip())
 dependency=result.stdout.strip().split()[0];ids.append(dependency)
print('\n'.join(ids));print('Inspect: qstat -u "$USER". A scientific rejection stops afterok dependencies; it is not a completed study.')
