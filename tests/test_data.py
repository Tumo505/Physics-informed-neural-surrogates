from pathlib import Path
import numpy as np
from lv_surrogate.data.dalton import Dalton,AuthorPreprocess
P=Path(__file__).resolve().parents[1]
def test_included_author_data_and_preprocessing():
 root=P/'external/passive-lv-gnn-emul/data/lvDataFixedGeom';d=Dalton(root,'test');c=d[0];assert c['real-node-displacement'].shape==(6784,3);assert d.edges.shape[1]==2 and d.edges.min()>=0 and d.edges.max()<6784
 v,e,t,z=AuthorPreprocess(root)(c)
 for name,a in [('augmented-node-features',v),('edge-features',e),('global-features',t[0])]:np.testing.assert_allclose(a,np.load(root/f'processedData/test/{name}.npy')[0],rtol=2e-5,atol=1e-5)
def test_registered_pilot_split_disjoint():
 import json
 s=json.loads((P/'data/splits/pilot.json').read_text());assert not set(s['validation'])&set(s['calibration']);assert len(set(s['train']))==64;assert all(0<=i<2250 for i in s['train'])
 # Indices are local to original author partitions; identical integer values in train/test are different cases.
 assert max(s['calibration'])<150 and max(s['test'])<567
