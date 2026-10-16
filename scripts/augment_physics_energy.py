"""Endpoint elastic strain-energy errors on frozen cardiac test predictions."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from lv_surrogate.physics.cardiac_residual import CardiacFirstStep
from lv_surrogate.physics.kinematics import deformation_gradient
P=Path(__file__).resolve().parents[1];torch.set_num_threads(4);a=np.load(P/'data/processed/cardiac/cohort.npz');assembler=CardiacFirstStep(a);result=json.loads((P/'experiments/metrics/physics_experiment.json').read_text())
def energy(u):
 F=deformation_gradient(assembler.X,torch.tensor(u,dtype=torch.float64),assembler.cells);Fq=F[:,None].expand(-1,len(assembler.weights),-1,-1);return float((assembler.vol[:,None]*assembler.energy_density(Fq)*assembler.weights).sum())
for name,r in result['results'].items():
 pred=np.load(P/f'experiments/metrics/physics_{name}_predictions.npz');errors=[]
 for row,u,truth in zip(r['per_case'],pred['predictions'],pred['truth']):
  estimate=energy(u);reference=energy(truth);error=abs(estimate-reference);row.update(endpoint_elastic_energy_J=estimate,reference_endpoint_elastic_energy_J=reference,absolute_endpoint_elastic_energy_error_J=error,relative_endpoint_elastic_energy_error=error/abs(reference) if abs(reference)>1e-12 else None);errors.append(error)
 r['mean_absolute_endpoint_elastic_energy_error_J']=float(np.mean(errors))
result['energy_scope']='passive elastic stored energy at endpoint; excludes kinetic energy and viscous dissipation'
baseline=result['results']['data_only'];physics=result['results'][result['lambda_selection']['selected']];result['H1_observed']={'residual_ratio_physics_over_data':physics['mean_normalized_residual_rms']/baseline['mean_normalized_residual_rms'],'displacement_RMSE_ratio_physics_over_data':physics['mean_component_rmse_m']/baseline['mean_component_rmse_m'],'conclusion':'No residual improvement observed in this matched fixed-geometry startup pilot; do not claim H1 support.' if physics['mean_normalized_residual_rms']>=baseline['mean_normalized_residual_rms'] else 'Residual reduction observed; one matched seed and one geometry remain a limited pilot.'}
(P/'experiments/metrics/physics_experiment.json').write_text(json.dumps(result,indent=2));print(result['H1_observed'])
