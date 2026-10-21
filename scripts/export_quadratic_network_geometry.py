"""Native P2 geometry for a frozen-network gradient check on training inputs only."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from FIAT.reference_element import UFCTriangle
from FIAT.quadrature_schemes import create_quadrature
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,export_geometry,LVProblem
from lv_surrogate.physics.quadratic import QuadraticKinematics
cfg=json.loads((P/'configs/physics/dynamic_study.json').read_text())
case=next(c for c in cfg['cases'] if c['split']=='train')
geo,mat,base=setup(dt=.005,geometry_scales=case['scales'])
params=dict(base.parameters);params['function_space']='P_2'
problem=LVProblem(geo,mat,parameters=params,solver_parameters={'linear_solver':'mumps','petsc':{'ksp_type':'preonly','pc_type':'none'}})
arrays=export_geometry(geo,mat,base);kin=QuadraticKinematics(problem.u_space,arrays['qpoints'],arrays['qweights'])
arrays['canonical_cell_dofs']=kin.maps
quad=create_quadrature(UFCTriangle(),6);arrays['face_qpoints']=quad.get_points();arrays['face_qweights']=quad.get_weights()
for key in ['alpha_f','alpha_m','alpha_epi','alpha_top','beta_epi','beta_top','rho']:arrays[key]=float(problem.parameters[key])
arrays.update(dt=.005,activation=0.,dof_count=problem.u_space.dim())
path=P/'experiments/logs/quadratic_network_geometry.npz';assert not path.exists();np.savez_compressed(path,**arrays)
result=dict(complete=True,training_case=case,path=str(path.relative_to(P)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Native coarse P2 displacement/test space on the first prespecified original training geometry; no new labels,calibration or test fields.')
(P/'experiments/metrics/quadratic_network_geometry.json').write_text(json.dumps(result,indent=2)+'\n')
print('P2 NETWORK GEOMETRY READY',case['id'],flush=True)
