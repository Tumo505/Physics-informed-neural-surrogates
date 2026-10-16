from pathlib import Path
import sys,copy,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from lv_surrogate.models.multiscale import MultiscaleSurrogate
P=Path(__file__).resolve().parents[1];D=P/'data/processed/full';torch.set_num_threads(2);torch.manual_seed(11)
x=torch.from_numpy(np.load(D/'train_points.npy',mmap_mode='r')[:2,:128].copy());c=torch.from_numpy(np.load(D/'train_coarse.npy',mmap_mode='r')[:2].copy());g=torch.from_numpy(np.load(D/'train_globals.npy',mmap_mode='r')[:2].copy());e=torch.from_numpy(np.load(D/'train_edges.npy',mmap_mode='r')[:2].copy());top=torch.from_numpy(np.load(D/'topology.npy'));parents=torch.from_numpy(np.load(D/'parents.npy')[:128]);model=MultiscaleSurrogate();gpu=copy.deepcopy(model).to('mps');inputs=[x,c,g,e,top,parents];a=model(*inputs);b=gpu(*[v.to('mps') for v in inputs]);a.square().mean().backward();b.square().mean().backward();diff=(a-b.cpu()).abs().max().item();gd=max((p.grad-q.grad.cpu()).abs().max().item() for p,q in zip(model.parameters(),gpu.parameters()));assert diff<1e-4 and gd<1e-4,(diff,gd)
# Query chunking must preserve the full-field evaluation exactly (fixed graph embedding).
model.eval()
with torch.no_grad():
 latent=model.encode(c,g,e,top);chunk=model.decode(x[:,:64],g,latent,parents[:64]);torch.testing.assert_close(chunk,a[:,:64],atol=1e-6,rtol=1e-6)
out={'cpu_mps_prediction_max_abs':diff,'cpu_mps_parameter_gradient_max_abs':gd,'query_chunk_consistency':'passed','torch':torch.__version__};(P/'experiments/metrics/full_backend.json').write_text(json.dumps(out,indent=2));print(out)
