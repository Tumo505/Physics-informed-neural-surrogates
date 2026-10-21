"""Fail-closed preregistration/reference gate; never read new test response fields."""
import argparse
import hashlib
import json
import math
from pathlib import Path

P = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--require-ready', action='store_true')
args = ap.parse_args()
path = P/'configs/physics/dynamic_qualified_followup.json'
cfg = json.loads(path.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()
assert sha(P/cfg['original_model_freeze']['path']) == cfg['original_model_freeze']['sha256']
cases = cfg['cases']; assert len({c['id'] for c in cases}) == len(cases)
groups = {}
for case in cases:
    groups.setdefault(case['split'], set()).add(case['geometry_id'])
    assert min(case['scales'])>0 and min(case['amplitude'],case['timing_scale'])>0
for a in groups:
    for b in groups:
        if a!=b: assert groups[a].isdisjoint(groups[b]), 'Whole geometry split leakage'
assert len(groups['calibration']) == cfg['uncertainty']['calibration_geometries'] == 20
assert len(groups['test']) == 20
old = json.loads((P/'configs/physics/dynamic_study.json').read_text())['cases']
old += json.loads((P/'configs/physics/dynamic_confirmation.json').read_text())['cases']
old_geometry = {tuple(c['scales']) for c in old}
for case in cases:
    if case['split'] in ['calibration','test','ood_test']:
        assert tuple(case['scales']) not in old_geometry, 'New evaluation geometry was inspected previously'
blockers = []
reference_path = P/cfg['reference_gate']['manifest']
if not reference_path.exists():
    blockers.append('No qualified reference freeze: current mesh and Q6 integration controls fail; R2 is incomplete.')
else:
    freeze = json.loads(reference_path.read_text())
    assert freeze['protocol_sha256'] == sha(path)
    for control in freeze['controls']:
        metric_path = P/control['path']; assert sha(metric_path) == control['sha256']
        metric = json.loads(metric_path.read_text())
        assert metric['complete'] and not metric['pilot']
        assert metric['tolerance_relative_L2'] == cfg['reference_gate']['relative_sensitivity_tolerance']
        if not metric['passes_declared_one_percent_sensitivity']:
            blockers.append('Failed control: '+control['path'])
    for factor in cfg['reference_gate']['required_controls']:
        assert any(c['factor']==factor for c in freeze['controls']), factor
    levels = {c['refinement_level'] for c in freeze['mesh_levels']}
    assert len(levels)>=cfg['reference_gate']['minimum_mesh_levels']
    for item in freeze['references']:
        root = P/'data/processed'/item['name']
        manifest = json.loads((root/'manifest.json').read_text())
        verification_path = P/item['verification']
        assert sha(verification_path) == item['verification_sha256']
        verification = json.loads(verification_path.read_text())
        assert manifest['complete'] and not (root/'failure.json').exists()
        assert verification['complete'] and verification['passed']
        assert verification['config_sha256'] == manifest['config_sha256']
        assert verification['output_sha256'] == manifest['output_sha256']
        for name,digest in manifest['output_sha256'].items(): assert sha(root/name)==digest
    # A separately checked backend/transfer audit is mandatory; a manifest flag alone is insufficient.
    for key in ['network_gradient_audit','canonical_transfer_audit']:
        item = freeze[key]; assert sha(P/item['path']) == item['sha256']
        audit = json.loads((P/item['path']).read_text())
        assert audit['complete'] and audit['passed']
result = dict(complete=True, protocol_sha256=sha(path), ready_to_train=not blockers,
    split_geometry_counts={k:len(v) for k,v in groups.items()},
    old_evaluation_geometries_excluded=True, calibration_order_statistics={
        str(coverage):math.ceil((20+1)*coverage) for coverage in cfg['uncertainty']['nominal_coverages']},
    blockers=blockers, test_response_fields_read=False,
    scope='Readiness audit only. No reference, network, calibration or final-test experiment is claimed by this report.')
(P/'experiments/metrics/dynamic_qualified_followup_readiness.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
if args.require_ready and blockers: raise SystemExit(2)
