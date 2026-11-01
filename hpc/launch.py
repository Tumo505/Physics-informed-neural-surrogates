"""One gateway submission: repaired setup, then dependency-linked reference/audit slices."""
import argparse,json,re,shlex,subprocess,sys
from pathlib import Path
from registry import atomic,event
ROOT=Path(__file__).resolve().parents[1]

def launch(root,run_id,*,queue='workq',cpus=28,ranks=12,memory_gb=240,hours=24,jobs=8,account=None):
 root=Path(root).resolve()
 assert re.fullmatch('[A-Za-z0-9_-]+',run_id),'Use letters, digits, underscores and hyphens in run ID'
 assert 1<=jobs<=10 and 1<=hours<=168 and 1<=ranks<=cpus and memory_gb>=64
 snapshot=json.loads((root/'experiments/manifests/hpc_migration_snapshot.json').read_text())
 assert snapshot['complete'],'Migration snapshot is incomplete'
 required=[*snapshot['files'],f"data/processed/{snapshot['reference']}/displacement_coefficients.npy"]
 missing=[x for x in required if not(root/x).is_file()]
 if missing:raise SystemExit('Transfer the preserved experiment data before launch. Missing: '+', '.join(missing))
 assert not(root/'experiments/hpc'/run_id).exists(),'Existing run must not be re-launched'
 directory=root/'.hpc/launches'/run_id
 directory.mkdir(parents=True,exist_ok=False) # Atomic duplicate-submission protection.
 revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
 settings=dict(run_id=run_id,queue=queue,cpus=cpus,ranks=ranks,memory_gb=memory_gb,hours=hours,jobs=jobs,account=account,source_commit=revision)
 atomic(directory/'launch.json',settings)
 def submit(argv,role):
  result=subprocess.run(argv,cwd=root,text=True,capture_output=True)
  event(directory,'job_submission',role=role,argv=argv,return_code=result.returncode,stdout=result.stdout.strip(),stderr=result.stderr.strip())
  if result.returncode:raise SystemExit('Submission stopped; preserve '+str(directory)+'. '+result.stderr.strip())
  identifier=result.stdout.strip().split()[0]
  assert re.fullmatch(r'[A-Za-z0-9_.-]+',identifier),'Unexpected scheduler job ID; inspect saved response'
  print(role+': '+identifier,flush=True)
  return identifier
 setup=['qsub','-N','lv_setup','-o',str(directory/'setup-pbs.log')]
 if account:setup+=['-A',account]
 dependency=submit(setup+[str(root/'hpc/setup.pbs')],'setup')
 prepare=[str(root/'.venv/bin/python'),'hpc/prepare.py','--run-id',run_id,'--queue',queue,'--cpus',str(cpus),'--ranks',str(ranks),'--memory-gb',str(memory_gb),'--hours',str(hours),'--software','PBSJobApp','--placement','excl']
 if account:prepare+=['--account',account]
 run='experiments/hpc/'+run_id
 for index in range(1,jobs+1):
  script=directory/f'slice-{index:03d}.pbs'
  lines=['#!/bin/bash','set -euo pipefail','cd '+shlex.quote(str(root)),
         'export PATH="$PWD/.hpc/tools/bin:$PATH"',
         '[[ "$(git rev-parse HEAD)" == '+shlex.quote(revision)+' ]] || { echo "Source changed after launch; stopped" >&2; exit 2; }']
  if index==1:lines.append(shlex.join(prepare))
  lines.append('exec '+shlex.join([str(root/'.venv/bin/python'),'hpc/run.py','--run',run,'--slice-seconds',str(hours*3600)]))
  script.write_text('\n'.join(lines)+'\n')
  command=['qsub','-N','lv_'+run_id[:12],'-q',queue,'-l',f'select=1:ncpus={cpus}:mem={memory_gb}gb','-l','place=excl','-l','software=PBSJobApp','-l',f'walltime={hours:02d}:00:00','-j','oe','-o',str(directory/f'pbs-{index:03d}.log'),'-W','depend=afterok:'+dependency]
  if account:command+=['-A',account]
  dependency=submit(command+[str(script)],'reference_slice_'+str(index))
 print('Queued setup and reference/audit slices. Logout is safe; no repeated healthy-job submissions are needed.')
 print('Failures stop the dependency chain. Budget exhaustion and scientific gates require review; missing full-study stages are not bypassed.')
 return directory

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--queue',default='workq');p.add_argument('--cpus',type=int,default=28);p.add_argument('--ranks',type=int,default=12);p.add_argument('--memory-gb',type=int,default=240);p.add_argument('--hours',type=int,default=24);p.add_argument('--jobs',type=int,default=8);p.add_argument('--account');a=p.parse_args();launch(ROOT,**vars(a))
if __name__=='__main__':main()
