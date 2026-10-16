"""Verify frozen model hashes, train-only transforms and geometry split provenance."""
from pathlib import Path
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parents[1]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
load=lambda f:json.loads((P/f).read_text())
name='cardiac_geometry_refinement_v3';run=load(f'experiments/manifests/{name}.json');freeze=load(f'experiments/manifests/{name}_evaluation_freeze.json');cohort=load('experiments/manifests/cardiac_confirmation_v1.json');cache=load('experiments/manifests/confirmation_ml.json');original=load('experiments/manifests/geometry_ml.json')
assert run['complete'] and cohort['complete']
assert run['data_access']==['train','validation']
for stem,digest in freeze['checkpoint_sha256'].items():assert sha(P/f'experiments/checkpoints/{name}/{stem}')==digest
assert len(freeze['checkpoint_sha256'])==10
for folder in ['cardiac_geometry_ml','cardiac_confirmation_ml']:assert sha(P/f'data/processed/{folder}/normalizers.npz')==original['normalizer_sha256']==run['normalizer_sha256']==cache['parent_normalizer_sha256']==freeze['normalizer_sha256']
assert sha(P/'experiments/manifests/cardiac_confirmation_v1.json')==cache['source_cohort_manifest_sha256']
ids=[]
for split,r in cache['splits'].items():
 path=P/f'data/processed/cardiac_confirmation_ml/{split}.npz';assert sha(path)==r['sha256'];a=np.load(path);current=np.unique(a['geometry_ids']);np.testing.assert_array_equal(current,np.arange(*r['geometry_ids']));assert len(a['targets'])==4*len(current);ids.extend(current)
assert len(ids)==len(set(ids))==52
for rec in cohort['records']:assert sha(P/f'data/processed/cardiac_confirmation_v1/geometry_{rec["geometry_id"]:03d}.npz')==rec['file_sha256']
result={'passed':True,'frozen_checkpoints_verified':10,'FE_geometry_checksums_verified':52,'disjoint_confirmation_geometry_splits':True,'training_normalizers_unchanged':True,'normalizer_sha256':original['normalizer_sha256'],'training_data_access':['train','validation'],'scope':'artifact integrity and split/provenance audit; not a claim of model accuracy or anatomical independence'}
(P/'experiments/metrics/confirmation_integrity.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
