"""CPU smoke uses included author fixed-geometry data; no archive/download needed."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from lv_surrogate.data.dalton import Dalton,AuthorPreprocess
from lv_surrogate.models.reference import read_weights,fixed_predict
P=Path(__file__).resolve().parents[1];R=P/'external/passive-lv-gnn-emul';b=R/'data/lvDataFixedGeom';d=Dalton(b,'test');case=d[0];v,e,t,z=AuthorPreprocess(b)(case);assert case['real-node-displacement'].shape==(6784,3)
w=R/'emulationResults/trainedParameters/lvDataFixedGeom';lat=np.load(w/'latent_nodal_values.npy');y=fixed_predict(t,read_weights(w/'theta_encode_params.pkl'),read_weights(w/'trainedNetworkParams.pkl'),lat);assert y.shape==(6784,3) and np.isfinite(y).all();print('Loaded real held-out fixed-LV graph and predicted 6784 × 3 displacement components; CPU smoke passed.')
