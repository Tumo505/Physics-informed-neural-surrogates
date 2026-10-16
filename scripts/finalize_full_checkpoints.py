"""Validation-only selection across initial and low-LR continuation checkpoints."""
from pathlib import Path
import json,torch,yaml,hashlib,shutil
P=Path(__file__).resolve().parents[1];cfg=yaml.safe_load((P/'configs/gnn/full_low_lr.yaml').read_text());root=P/'experiments/checkpoints';selected=[]
for kind,seeds in [('gnn',cfg['seeds']),('point',cfg['baseline_seeds'])]:
 for seed in seeds:
  latest=torch.load(root/cfg['run_name']/f'{kind}_{seed}_latest.pt',map_location='cpu',weights_only=False)
  if not latest['complete']:raise RuntimeError('Training not complete; cannot freeze selection')
  parent=root/cfg['initialize_best_from']/f'{kind}_{seed}_best.pt';current=root/cfg['run_name']/f'{kind}_{seed}_best.pt';a=torch.load(parent,map_location='cpu',weights_only=False);b=torch.load(current,map_location='cpu',weights_only=False)
  source=parent if a['validation_mse_cm2']<b['validation_mse_cm2'] else current
  record={'kind':kind,'seed':seed,'selected_source':str(source.relative_to(P)),'validation_component_rmse_cm':min(a['validation_mse_cm2'],b['validation_mse_cm2'])**.5}
  if source==parent:shutil.copy2(parent,current)
  record['frozen_sha256']=hashlib.sha256(current.read_bytes()).hexdigest();selected.append(record)
(P/'experiments/manifests/full_selection.json').write_text(json.dumps({'rule':'minimum full-node validation component MSE across both stages; includes original checkpoint, never forced to accept continuation','validation_ids':list(range(16)),'calibration_or_test_used':False,'selected':selected},indent=2));print(selected)
