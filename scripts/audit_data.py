"""Audit originals; safe extraction of LV archive only, scan array statistics in chunks."""
from pathlib import Path
import hashlib,json,zipfile,platform,subprocess,sys
import numpy as np
P=Path(__file__).resolve().parents[1]; ROOT=P.parent

def digest(p,alg='sha256'):
 h=hashlib.new(alg)
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 out={'access_date':'2026-10-16','python':sys.version,'platform':platform.platform(),'archives':[],'arrays':{},'sources':{},'physics_gate':'Dalton edges cannot define FE equilibrium; separate cardiac verification is recorded when available','units':'LV cm: author main.py evaluation comment; feature meanings unknown unless separately documented'}
 for name,expected,url in [('lvData.zip','96987da42e159c205fa4aeb30cd5526e','https://zenodo.org/records/7075055'),('benchmark_article_data.zip','75602be4777c4ca2262c2bcfd2134b15','https://zenodo.org/records/14260459')]:
  p=ROOT/name
  with zipfile.ZipFile(p) as z:
   members=z.infolist()
   assert all(not Path(m.filename).is_absolute() and '..' not in Path(m.filename).parts for m in members)
   md5=digest(p,'md5'); assert md5==expected,(p,md5)
   info={'path':str(p),'bytes':p.stat().st_size,'expanded_bytes':sum(m.file_size for m in members),'md5':md5,'expected_md5':expected,'source':url,'members':[m.filename for m in members]}
   if name=='lvData.zip':
    z.extractall(P/'data/raw')
   else:
    info['readmes']={m.filename:z.read(m).decode(errors='replace') for m in members if m.filename.endswith('README.md') and m.file_size<30000 and '__MACOSX' not in m.filename}
   out['archives'].append(info)
  print(name,'checksum verified',flush=True)
 for src in ['passive-lv-gnn-emul','cardiac_benchmark']:
  out['sources'][src]={'commit':subprocess.check_output(['git','-C',str(ROOT/src),'rev-parse','HEAD'],text=True).strip(),'license':(ROOT/src/'LICENSE').read_text() if (ROOT/src/'LICENSE').exists() else 'README declares MIT; LICENSE file absent in supplied commit'}
 for base in [P/'data/raw/lvData',ROOT/'passive-lv-gnn-emul/data/lvDataFixedGeom']:
  for p in sorted(base.rglob('*.npy')):
   key=str(p.relative_to(ROOT));
   try:a=np.load(p,mmap_mode='r',allow_pickle=False)
   except ValueError:
    out['arrays'][key]={'object_array':'author clustering labels; trusted-source load only','sha256':digest(p)};continue
   stats={'shape':list(a.shape),'dtype':str(a.dtype),'bytes':p.stat().st_size,'sha256':digest(p),'nonfinite':0,'min':None,'max':None,'mean':None,'std':None}
   sums=sq=cnt=0.; lo=np.inf;hi=-np.inf
   for chunk in np.array_split(a.reshape(-1),max(1,a.size//1000000)):
    b=np.asarray(chunk,dtype=np.float64); valid=np.isfinite(b);stats['nonfinite']+=int((~valid).sum());b=b[valid]
    if len(b):lo=min(lo,b.min());hi=max(hi,b.max());sums+=b.sum();sq+=(b*b).sum();cnt+=len(b)
   if cnt:stats.update(min=float(lo),max=float(hi),mean=float(sums/cnt),std=float(max(0,sq/cnt-(sums/cnt)**2)**.5))
   if a.ndim>=2 and a.shape[-1]<=40 and 'topology' not in p.name :
    flat=a.reshape(-1,a.shape[-1]);stats['per_channel']={'min':np.min(flat,axis=0).tolist(),'max':np.max(flat,axis=0).tolist(),'mean':np.mean(flat,axis=0).tolist(),'std':np.std(flat,axis=0).tolist()}
   out['arrays'][key]=stats
 # Exact duplicate audit based on full coordinates and shape descriptor nearest distances.
 duplicate={}; hashes={}; coeff=[]; labels=[]
 for split in ['train','validation','test']:
  b=P/'data/raw/lvData/rawData'/split;a=np.load(b/'real-node-coords.npy',mmap_mode='r'); counts={}
  for i,x in enumerate(a):
   h=hashlib.sha256(x.tobytes()).hexdigest();hashes.setdefault(h,[]).append(f'{split}:{i}')
  coeff.append(np.load(b/'shape-coeffs.npy')); labels.extend([split]*len(a))
 duplicate['exact_repeated_geometry_groups']=[v for v in hashes.values() if len(v)>1]
 from scipy.spatial import cKDTree
 c=np.concatenate(coeff); scale=c[np.array(labels)=='train'].std(0);c=c/np.maximum(scale,1e-12)
 dist,idx=cKDTree(c).query(c,k=2)
 duplicate['nearest_standardized_shape_distance_quantiles']=np.quantile(dist[:,1],[0,.01,.5,1]).tolist()
 duplicate['near_duplicate_rule']='Euclidean distance in training-standardized shape coefficients <1e-6; descriptive only'
 duplicate['near_duplicate_pairs_lt_1e6']=[[i,int(idx[i,1])] for i in range(len(c)) if dist[i,1]<1e-6 and i<int(idx[i,1])]
 fg=np.load(ROOT/'passive-lv-gnn-emul/data/lvDataFixedGeom/rawData/test/real-node-coords.npy')[0]
 duplicate['fixed_to_varying_nearest']={}
 for sp in ['train','validation','test']:
  xx=np.load(P/f'data/raw/lvData/rawData/{sp}/real-node-coords.npy',mmap_mode='r');ds=[float(np.max(np.abs(x-fg))) for x in xx];j=int(np.argmin(ds));duplicate['fixed_to_varying_nearest'][sp]={'index':j,'max_abs_coord_difference_cm':ds[j]}
 out['split_audit']=duplicate
 out['dataset_licenses']={n:json.loads((P/f'experiments/manifests/zenodo_{n}.json').read_text())['metadata'].get('license') for n in ['7075055','14260459']}
 out['unknowns']=['per-column global/node feature physical definitions','complete Dalton constitutive law and parameter mapping','pressure/traction and boundary labels','volumetric connectivity','original solver reproduction and timings']
 out['cases']={name:{sp:len(np.load(base/f'rawData/{sp}/real-node-displacement.npy',mmap_mode='r')) for sp in ['train','validation','test']} for name,base in [('varying',P/'data/raw/lvData'),('fixed',ROOT/'passive-lv-gnn-emul/data/lvDataFixedGeom')]}
 out['nodal_correspondence']='Common 6784-node indexing; displacement/coordinate shapes agree. No Dalton FE element correspondence.'
 out['reference_convention']='Coordinates reference; displacement endpoint minus reference; x=X+u'
 if (P/'reports/benchmark_audit.json').exists():out['benchmark']=json.loads((P/'reports/benchmark_audit.json').read_text())
 (P/'reports/data_audit.json').write_text(json.dumps(out,indent=2))
 # Detailed audit is machine-readable; consolidated report is the only Markdown report.
 print('audit complete',flush=True)
if __name__=='__main__':main()
