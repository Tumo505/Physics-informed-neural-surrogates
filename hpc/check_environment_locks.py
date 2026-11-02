"""Require the installed conda package URLs to match the resolved Linux locks."""
from pathlib import Path

def urls(path):
 return {x.strip().split('#',1)[0] for x in Path(path).read_text().splitlines() if x.startswith(('https://','http://'))}
def check(root):
 root=Path(root)
 for name in ['native','modern','tools','native-v2']:
  lockname='native' if name=='native-v2' else name
  expected=urls(root/f'hpc/{lockname}-linux64-glibc217.lock');actual=urls(root/f'.hpc/setup-records/{name}-installed.lock');assert expected and expected==actual,f'Installed {name} environment does not match lock; preserve install logs and resolve before computing'
if __name__=='__main__':check(Path(__file__).resolve().parents[1]);print('Installed environment locks match')
