"""Inspect mesh archive selectively; do not extract multi-GB visualization histories."""
from pathlib import Path
import zipfile,json,sys,importlib.util
P=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(P.parent/'benchmark_article_data.zip') as z:
 files=[m for m in z.infolist() if '__MACOSX' not in m.filename]
 meshes=[m for m in files if '/meshes_and_fibers/data/' in m.filename]
 out={'mesh_members':[{'path':m.filename,'bytes':m.file_size} for m in meshes], 'solver_available_in_current_runtime':bool(importlib.util.find_spec('dolfin')),'isolated_fenics_installed':(P/'.fenics/bin/python').exists(),'solver_probe_status':'see reports/solver_probe.md; native cardiac execution exits139 in this environment','formulation':'dynamic Holzapfel–Ogden, inertia + viscosity + follower pressure + Robin springs/friction; not Dalton passive endpoint task','required_for_residual':['exact FE space and DOF-to-node map','fiber/sheet fields and parameter configuration','boundary markers','displacement velocity acceleration histories','time integration and quadrature','trusted solver residual assembly']}
 # Only mesh input files under 50 MB, total cap 150 MB.
 candidates=[m for m in meshes if m.filename.endswith(('.h5','.xdmf')) and ('monoventricle' in m.filename or 'ellipsoid' in m.filename)]
 size=sum(m.file_size for m in candidates)
 out['selected_mesh_bytes']=size
 if size<150_000_000:
  for m in candidates:
   assert '..' not in Path(m.filename).parts
   z.extract(m,P/'data/raw/benchmark')
 out['selected_mesh_files']=[m.filename for m in candidates]
 try:
  import h5py
  schema={}
  for p in (P/'data/raw/benchmark').rglob('*.h5'):
   with h5py.File(p) as h:
    fields={}
    def visit(n,v):
     if isinstance(v,h5py.Dataset):fields[n]={'shape':list(v.shape),'dtype':str(v.dtype)}
    h.visititems(visit);schema[str(p.relative_to(P))]=fields
  out['hdf5_schema']=schema
 except ImportError:out['hdf5_schema']='h5py unavailable'
 (P/'reports/benchmark_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ('mesh_members','hdf5_schema')},indent=2))
