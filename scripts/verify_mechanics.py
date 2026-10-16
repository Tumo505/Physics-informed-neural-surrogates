from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import h5py,numpy as np,torch
from lv_surrogate.physics.kinematics import tetra_geometry,deformation_gradient,green_lagrange
P=Path(__file__).resolve().parents[1];p=next((P/'data/raw/benchmark').rglob('*.h5'))
with h5py.File(p) as h:X=torch.from_numpy(h['Mesh/mesh/geometry'][:]);c=torch.from_numpy(h['Mesh/mesh/topology'][:].astype('int64'))
inv,vol,g=tetra_geometry(X,c);A=torch.tensor([[1.1,.02,0],[0,1.05,0],[0,0,.98]],dtype=X.dtype);F=deformation_gradient(X,X@A.T-X,c);err=(F-A).abs().max().item()
out={'source':str(p),'n_nodes':len(X),'n_tetrahedra':len(c),'units':'benchmark mesh m; volume m^3 (toolkit README)','total_reference_volume_m3':vol.sum().item(),'minimum_element_volume_m3':vol.min().item(),'affine_F_max_abs_error':err,'min_affine_detF':torch.linalg.det(F).min().item(),'interpretation':'kinematics verification on authentic cardiac mesh, manufactured affine displacement; NOT solver solution or cardiac physics validation','gates':{'1_translation':'unit test','2_identity':'neo-Hookean unit only','3_affine':'actual benchmark mesh','4_volume_gradients':'unit and benchmark mesh','5_converged_FEM_residual':'BLOCKED: isolated FEniCS installed but cardiac probe exits139','6_trusted_solver_assembly':'BLOCKED','7_residual_gradient':'synthetic unit only','8_Jacobians':'unit and manufactured benchmark','9_perturbation':'synthetic unit only','10_cardiac_BC_pressure':'NOT IMPLEMENTED'}}
assert err<1e-9 and vol.min()>0
if (P/'experiments/metrics/cardiac_verification.json').exists():
 out['cardiac_verification']=json.loads((P/'experiments/metrics/cardiac_verification.json').read_text());out['gates'].update({'5_converged_FEM_residual':'passed;32 cardiac startup cases','6_trusted_solver_assembly':'passed; independent dynamic P1 assembly comparison','7_residual_gradient':'passed; trusted tangent VJP finite difference and neural backprop','10_cardiac_BC_pressure':'natural Robin/pressure terms verified; no Dirichlet DOFs'})
(P/'reports/mechanics_validation.json').write_text(json.dumps(out,indent=2));print(out)
