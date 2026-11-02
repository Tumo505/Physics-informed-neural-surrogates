"""Verify pinned addon versions and installed RECORD hashes in native-v2."""
import base64,hashlib,json
from importlib.metadata import distribution
from pathlib import Path

def check(root):
 specs=json.loads((Path(root)/'hpc/native-python-addons-resolution.json').read_text())['packages'];checked=[]
 for item in specs:
  package=distribution(item['name']);assert package.version==item['version'],item['name']+' version mismatch'
  count=0
  for f in package.files or []:
   if f.hash is None:continue
   assert f.hash.mode=='sha256',str(f)
   path=package.locate_file(f);actual=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b'=').decode()
   assert actual==f.hash.value,'Installed addon hash mismatch: '+str(f)
   count+=1
  assert count,'No installed addon file hashes: '+item['name']
  checked.append(dict(name=item['name'],version=package.version,verified_files=count,wheel_sha256=item['sha256']))
 return checked
if __name__=='__main__':print(json.dumps(dict(passed=True,packages=check(Path(__file__).resolve().parents[1])),indent=2))
