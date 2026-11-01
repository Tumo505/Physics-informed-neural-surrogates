"""Publish committed research changes without sending local Markdown-bearing history."""
import argparse,os,subprocess,tempfile
from pathlib import Path
P=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--push',action='store_true');ap.add_argument('--message',default='Update reproducible research code and quantitative evidence');args=ap.parse_args()
def git(*words,input=None,env=None,check=True):
 return subprocess.run(['git',*words],cwd=P,input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,check=check)
def value(*words):return git(*words).stdout.decode().strip()
def excluded(path):return Path(path).suffix.lower() in {'.md','.markdown','.mdown','.mdx'} or path.startswith('paper/')
if args.push:git('fetch','origin')
head=value('rev-parse','HEAD');remote=value('rev-parse','origin/main')
last=git('config','--local','--get','cmbbe.lastPublishedResearchCommit',check=False)
base=last.stdout.decode().strip() if last.returncode==0 else value('merge-base',head,remote)
assert git('merge-base','--is-ancestor',base,head,check=False).returncode==0,'Publication checkpoint must remain an ancestor of the local research history'
paths=[p for p in git('diff','--name-only','-z',base,head).stdout.decode().split('\0') if p];selected=[p for p in paths if not excluded(p)]
if not selected:print('No committed non-Markdown changes to publish.');raise SystemExit(0)
patch=git('diff','--binary','--full-index',base,head,'--',*selected).stdout
fd,index=tempfile.mkstemp(prefix='cmbbe-publication-index-');os.close(fd);os.unlink(index);env=os.environ.copy();env['GIT_INDEX_FILE']=index
try:
 git('read-tree',remote,env=env)
 applied=git('apply','--cached','--3way','-',input=patch,env=env,check=False)
 if applied.returncode:
  raise RuntimeError('Publication merge conflicted; remote and local branches were not changed. '+applied.stderr.decode()[-3000:])
 tree=git('write-tree',env=env).stdout.decode().strip();changed=[p for p in git('diff','--name-only','-z',remote,tree).stdout.decode().split('\0') if p]
 assert not any(excluded(p) for p in changed),'Outgoing snapshot must not modify Markdown or paper/'
 if not changed:
  print('Remote already contains the non-Markdown changes.')
  if args.push:git('config','--local','cmbbe.lastPublishedResearchCommit',head)
  raise SystemExit(0)
 print(f'Checked publication: {len(changed)} non-Markdown files; {sum(excluded(p) for p in paths)} excluded paths; zero Markdown changes.')
 if not args.push:print('Dry run only; pass --push to publish.');raise SystemExit(0)
 message=(args.message+'\n\nLocal research checkpoint: '+head+'\nMarkdown and paper/ excluded; local research history retained.\n').encode()
 published=git('commit-tree',tree,'-p',remote,input=message).stdout.decode().strip()
 outgoing=[p for p in git('diff-tree','--no-commit-id','--name-only','-r','-z',published).stdout.decode().split('\0') if p];assert not any(excluded(p) for p in outgoing)
 git('update-ref','refs/heads/non-markdown-publication',published)
 pushed=git('push','origin','refs/heads/non-markdown-publication:refs/heads/main',check=False)
 if pushed.returncode:raise RuntimeError('Push stopped; publication commit retained on non-markdown-publication. '+pushed.stderr.decode()[-3000:])
 git('config','--local','cmbbe.lastPublishedResearchCommit',head);git('config','--local','cmbbe.lastPublishedRemoteCommit',published)
 print('Published to origin/main:',published)
finally:
 for path in [index,index+'.lock']:
  if os.path.exists(path):os.unlink(path)
