"""Freeze validation-only architecture choice; never read calibration/test arrays."""
from pathlib import Path
import json,argparse
import yaml,torch
P=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser();ap.add_argument('--expand',action='store_true');args=ap.parse_args();candidates={}
for name in ['control_v4','capacity_v4','fine_graph_v4']:
 cfg=yaml.safe_load((P/f'configs/gnn/{name}.yaml').read_text());r=json.loads((P/f'experiments/metrics/{name}/gnn_11.json').read_text())
 if not r['complete']:raise RuntimeError(f'{name} incomplete')
 candidates[name]={'validation_component_rmse_cm':r['best_validation_component_rmse_cm'],'parameters':r['parameters'],'epochs_completed':r['epochs_completed'],'config':cfg}
old=torch.load(P/'experiments/checkpoints/full_multiscale_v2/gnn_11_best.pt',weights_only=False,map_location='cpu')['validation_mse_cm2']**.5
selected=min(candidates,key=lambda k:candidates[k]['validation_component_rmse_cm']);improved=candidates[selected]['validation_component_rmse_cm']<old*(1-1e-6)
result={'candidates':candidates,'selected':selected,'previous_seed11_validation_rmse_cm':old,'improves_previous_seed11':improved,'decision_scope':'one-seed screening; selected family expanded only if it improves prior seed11 validation accuracy; all test comparisons are repeated benchmark evaluations','calibration_or_test_opened':False}
(P/'experiments/metrics/graph_study_selection.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))
if args.expand and improved:
 cfg=candidates[selected]['config'];cfg['seeds']=[11,22,33,44,55];cfg['screening_selection']='frozen in experiments/metrics/graph_study_selection.json before new calibration/test reads';(P/'configs/gnn/selected_v4.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
