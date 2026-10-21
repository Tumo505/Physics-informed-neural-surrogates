"""Export the exact serial mesh/fibers for partition-independent MPI qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,df
ap=argparse.ArgumentParser();ap.add_argument('--reference',required=True);args=ap.parse_args()
root=P/'data/processed'/args.reference;m=json.loads((root/'manifest.json').read_text());cfg=m['config']
cache=P/'data/processed/parallel_mesh_cache'/args.reference;cache.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
meta=cache/'manifest.json'
if meta.exists():
 d=json.loads(meta.read_text());assert d['config_sha256']==m['config_sha256']
 for n,h in d['output_sha256'].items():assert sha(cache/n)==h
 print('REUSE EXACT MESH CACHE',str(cache),flush=True);raise SystemExit(0)
data=np.load(root/'inputs.npz')
geo,material,_=setup(dt=cfg['dt_s'],geometry_scales=cfg['geometry_scales'],refinement_level=cfg['refinement_level'])
assert df.MPI.size(geo.mesh.mpi_comm())==1
np.testing.assert_allclose(geo.mesh.coordinates(),data['coords'],rtol=0,atol=1e-14)
np.testing.assert_array_equal(geo.mesh.cells(),data['cells'])
for key,f in [('fiber',geo.f0),('sheet',geo.s0)]:
 if key in data:np.testing.assert_allclose(f.compute_vertex_values(geo.mesh).reshape(3,-1).T,data[key],rtol=1e-12,atol=1e-14)
temporary=cache/'mesh.pending.h5'
with df.HDF5File(geo.mesh.mpi_comm(),str(temporary),'w') as h:
 h.write(geo.mesh,'mesh');h.write(geo.ffun,'ffun')
 for name in ['f0','s0','n0']:h.write(getattr(geo,name),name)
temporary.replace(cache/'mesh.h5')
coords=data['dof_coordinates'];components=np.full(len(coords),-1,dtype=np.int8)
for c in range(3):components[data['canonical_cell_dofs'][:,:,c].reshape(-1)]=c
assert (components>=0).all()
np.savez_compressed(cache/'serial_dof_map.npz',coordinates=coords,components=components)
out=dict(complete=True,reference=args.reference,config_sha256=m['config_sha256'],
 inputs_sha256=sha(root/'inputs.npz'),markers=geo.markers,
 material_parameters={k:float(v) for k,v in material.parameters.items()},
 output_sha256={n:sha(cache/n) for n in ['mesh.h5','serial_dof_map.npz']},
 source_sha256=sha(Path(__file__)),scope='Exact serial tetrahedra, facet markers and continuous P1 fibers; MPI must load this cache without regenerating or refining a different mesh.')
meta.write_text(json.dumps(out,indent=2)+'\n');print('EXACT MESH CACHE READY',str(cache),flush=True)
