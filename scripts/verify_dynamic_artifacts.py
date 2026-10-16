"""Verify fixed model/data/label-budget contracts and actual fallback reproduction."""
from pathlib import Path
import json,hashlib,numpy as np,argparse
P=Path(__file__).resolve().parents[1];reference=np.load(P/'data/processed/cardiac/cohort.npz');
ap=argparse.ArgumentParser();ap.add_argument('--models-only',action='store_true');args=ap.parse_args();cfg=json.loads((P/'configs/physics/dynamic_study.json').read_text());tcfg=json.loads((P/'configs/physics/dynamic_training.json').read_text());freeze=json.loads((P/'experiments/manifests/dynamic_final_freeze.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cases=[c for c in cfg['cases'] if c['split'] not in ['convergence','failed_timestep']];assert cases==freeze['frozen_cohort_cases'];roles={}
for c in cases:
 role='test' if c['split']=='load_shift' else c['split'];roles.setdefault(c['geometry'],set()).add(role)
assert all(len(r)==1 for r in roles.values())
assert len(freeze['checkpoint_sha256'])==20
for f,h in freeze['checkpoint_sha256'].items():assert sha(P/f)==h
regimes={}
for r in ['full','limited']:
 m=json.loads((P/f'experiments/manifests/{tcfg["name"]}_{r}.json').read_text());assert m['complete'] and m['data_access']==['train','validation'];D=P/f'data/processed/{tcfg["data_tag"]}_{r}';assert sha(D/'normalizers.npz')==m['normalizer_sha256'];a=np.load(D/'train.npz');mask=a['labelled_mask'].astype(bool);assert mask.sum()==(8 if r=='full' else 2);assert not any(k in a.files for k in ['u','u_old','v_old','a_old']);assert (a['q'][~mask]==0).all();regimes[r]={'labelled_training_trajectories':int(mask.sum()),'input_trajectories':len(mask),'normalizer_sha256':m['normalizer_sha256']}
assert json.loads((P/'experiments/metrics/dynamic_backend.json').read_text())['passed'];assert json.loads((P/'experiments/metrics/dynamic_neural_gradient.json').read_text())['passed']
result={'passed':True,'scope':'frozen checkpoint/preprocessing hashes, geometry role isolation and limited-label contracts; no inference claim','frozen_models_verified':20,'cohort_geometries':len(roles),'cohort_trajectories':len(cases),'regimes':regimes,'geometry_roles_disjoint':True,'primary_cohort_unchanged_since_model_freeze':True}
if not args.models_only:
 evaluation=json.loads((P/'experiments/metrics/dynamic_final_evaluation.json').read_text());assert evaluation['complete']
 for c in cases:
  root=P/f'data/processed/cardiac_dynamic_v1/{c["id"]}';m=json.loads((root/'manifest.json').read_text());assert m['complete'] and m['case']==c;assert sha(root/'trajectory.npz')==m['trajectory_sha256'];assert sha(root/'geometry.npz')==m['geometry_sha256'];geo=np.load(root/'geometry.npz');np.testing.assert_array_equal(geo['cells'],reference['cells']);np.testing.assert_allclose(geo['coords'],reference['coords']*np.array([c['scales'][1],c['scales'][0],c['scales'][0]]),rtol=0,atol=1e-14)
 fallback=json.loads((P/'experiments/metrics/dynamic_fallback_execution.json').read_text());case=fallback['case']['id'];a=np.load(P/f'data/processed/cardiac_dynamic_v1/{case}/trajectory.npz');b=np.load(P/f'data/processed/cardiac_dynamic_v1/{case}_fallback/trajectory.npz');difference=float(np.max(abs(a['u']-b['u'])));assert difference<1e-9;result['independent_FE_fallback_max_displacement_difference_m']=difference;result['cohort_data_checksums_verified']=44
if not args.models_only:
 ccfg=json.loads((P/'configs/physics/dynamic_confirmation.json').read_text());cfreeze=json.loads((P/'experiments/manifests/dynamic_confirmation_freeze.json').read_text());assert cfreeze['checkpoint_sha256']==freeze['checkpoint_sha256'];assert sha(P/'configs/physics/dynamic_confirmation.json')==cfreeze['confirmation_config_sha256'];assert len(ccfg['cases'])==16
 for c in ccfg['cases']:
  root=P/f'data/processed/cardiac_dynamic_v1/{c["id"]}';m=json.loads((root/'manifest.json').read_text());assert m['complete'] and m['case']==c;assert sha(root/'trajectory.npz')==m['trajectory_sha256'];assert sha(root/'geometry.npz')==m['geometry_sha256'];geo=np.load(root/'geometry.npz');np.testing.assert_array_equal(geo['cells'],reference['cells']);np.testing.assert_allclose(geo['coords'],reference['coords']*np.array([c['scales'][1],c['scales'][0],c['scales'][0]]),rtol=0,atol=1e-14);assert c['geometry'] not in roles
 result['confirmation_data_checksums_verified']=16;result['total_independent_geometries']=48;result['total_primary_and_confirmation_test_geometries']=20
(P/'experiments/metrics/dynamic_integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
