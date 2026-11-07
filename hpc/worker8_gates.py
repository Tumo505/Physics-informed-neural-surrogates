"""Check actual two-equation evidence, hashes and native admission before seeding."""
import json
import math
from pathlib import Path
from registry import sha


def positive(proof, workers=None):
    for name in ['endpoint', 'stress']:
        p = proof[name]
        assert p['passed'] and math.isfinite(p['minimum_Bernstein_bound']) and p['minimum_Bernstein_bound'] > 1e-9
        if workers is not None:
            assert p['evaluation_workers'] == workers


def cofactor_benchmark_gate(result, steps):
    assert result['complete'] and result['proof_benchmark_passed'] and result['stop_reason'] is None
    assert result['reference_continuation_allowed'] is False and result['physical_new_state_generated'] is False
    assert len(steps) == 2 and steps[1] == steps[0] + 1
    assert [row['step'] for row in result['records']] == steps
    for row in result['records']:
        assert row['proof_agreement'] is True
        times = row['pair_seconds']
        assert all(math.isfinite(times[key]) for key in ('cofactor', 'legacy'))
        assert 0 < times['cofactor'] < times['legacy']


def known_gate(root, probe_path, verification_path, fields_path, checkpoint_path, implementation='legacy'):
    root = Path(root)
    probe = json.loads((root / probe_path).read_text())
    independent = json.loads((root / verification_path).read_text())
    assert probe['complete'] and probe['passed'] and independent['complete'] and independent['passed']
    assert len(probe['records']) == len(independent['records']) == 2
    assert independent['probe_sha256'] == sha(root / probe_path)
    for name, path in [('fields_sha256', fields_path), ('checkpoint_sha256', checkpoint_path)]:
        assert probe[name] == independent[name] == sha(root / path), 'Known-field artifact hash mismatch'
    assert set(independent['history_relative_errors']) == {'u', 'u_old', 'v_old', 'a_old'}
    assert all(math.isfinite(x) and x >= 0 for x in independent['history_relative_errors'].values())
    start = probe['start_step']
    for offset, (row, audit) in enumerate(zip(probe['records'], independent['records'])):
        assert row['step'] == audit['step'] == start + offset
        assert row['SNES_reason'] > 0 and row['native_updates'] == 0
        assert math.isfinite(row['native_force_N']) and 0 <= row['native_force_N'] < 1e-8
        assert math.isfinite(audit['native_residual_N']) and 0 <= audit['native_residual_N'] < 1e-8
        assert math.isfinite(row['known_field_max_difference_m']) and 0 <= row['known_field_max_difference_m'] < 1e-9
        assert math.isfinite(audit['independent_norm_relative_error']) and 0 <= audit['independent_norm_relative_error'] < 1e-11
        assert math.isfinite(row['custom_trace']['custom_force_N']) and 0 <= row['custom_trace']['custom_force_N'] < 1e-9
        positive(row['whole_element_proof'], workers=8)
        if implementation == 'cofactor':
            assert row['whole_element_proof'].get('proof_implementation') == 'cofactor', 'Missing cofactor probe proof'
        positive(audit['whole_element_proof'])
    return dict(known_pair_passed=True, independent_native_audit_passed=True,
                new_physical_states=0, full_cycle_validation=False)
