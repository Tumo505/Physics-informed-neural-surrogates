"""Append-only hash-chained HPC run ledger; no credentials or full environment dumps."""
import argparse,datetime,fcntl,hashlib,json,os,platform,subprocess
from pathlib import Path

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.pending.json');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
def read(run):
 p=Path(run)/'events.jsonl';events=[];prior=None
 if not p.exists():return events
 for line in p.read_text().splitlines():
  r=json.loads(line);digest=r.pop('event_sha256');assert r['previous_event_sha256']==prior
  assert hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()==digest
  r['event_sha256']=digest;events.append(r);prior=digest
 return events
def event(run,kind,**data):
 run=Path(run);run.mkdir(parents=True,exist_ok=True)
 with (run/'registry.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);events=read(run);r=dict(time_utc=now(),kind=kind,previous_event_sha256=events[-1]['event_sha256'] if events else None,**data);r['event_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  with (run/'events.jsonl').open('a') as f:f.write(json.dumps(r,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
 return r
def artifact(root,run,path,role):
 p=Path(path);p=p if p.is_absolute() else Path(root)/p;assert p.is_file();return event(run,'artifact',path=str(p.relative_to(root)),role=role,bytes=p.stat().st_size,sha256=sha(p))
def job_metadata():
 names=['PBS_JOBID','PBS_QUEUE','PBS_O_WORKDIR','PBS_NCPUS','PBS_NP','PBS_NODEFILE']
 r={k:os.environ[k] for k in names if k in os.environ};r.update(hostname=platform.node(),platform=platform.platform(),python=platform.python_version())
 if Path('/proc/cpuinfo').exists():
  cpu=Path('/proc/cpuinfo').read_text();models=[x.split(':',1)[1].strip() for x in cpu.splitlines() if x.startswith('model name')];r['cpu_model']=models[0] if models else 'unknown';r['visible_logical_cpus']=os.cpu_count()
 if Path('/proc/meminfo').exists():r['host_physical_RAM_bytes']=int(Path('/proc/meminfo').read_text().split('MemTotal:',1)[1].split()[0])*1024
 if os.environ.get('PBS_NODEFILE'):r['allocated_hosts']=sorted(set(Path(os.environ['PBS_NODEFILE']).read_text().splitlines()))
 return r
def environment_fingerprint(prefix):
 prefix=Path(prefix).resolve();h=hashlib.sha256()
 paths=sorted([*prefix.glob('conda-meta/*.json'),*prefix.glob('lib/python*/site-packages/*.dist-info/METADATA')])
 assert paths,'Environment metadata missing: '+str(prefix)
 for p in paths:h.update(str(p.relative_to(prefix)).encode());h.update(p.read_bytes())
 return h.hexdigest()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['setup','verify']);ap.add_argument('--root',required=True);ap.add_argument('--run');a=ap.parse_args();root=Path(a.root).resolve();run=Path(a.run) if a.run else root/'experiments/hpc/setup'
 if a.action=='verify':print('Verified ledger events:',len(read(run)))
 else:
  event(run,'environment_setup',**job_metadata(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
  for p in sorted((root/'.hpc/setup-records').glob('*')):artifact(root,run,p,'environment')
