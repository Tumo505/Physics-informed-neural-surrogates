"""Gate a new restart representation without waiving the failed original history check."""
from pathlib import Path
import json,sys
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.durable_reference import sha
pp=P/'configs/physics/dynamic_native_history_reconciliation_20261101.json';p=json.loads(pp.read_text());q=json.loads((P/p['output']).read_text());assert q['complete'] and q['passed'] and not q['physical_step_generated'] and not q['original_audit_passed'] and q['original_pointwise_tolerance_unchanged']
assert sha(pp)==q['protocol_sha256'];assert sha(P/p['canonical_checkpoint'])==q['native_replay_checkpoint_sha256'];assert sha(P/p['failed_audit'])==q['failed_audit_sha256']
root=P/'data/processed'/p['reference'];assert sha(root/'restart.npz')==q['original_checkpoint_sha256'];assert sha(root/'positivity.json')==q['positivity_sha256']
a=json.loads((P/p['failed_audit']).read_text());assert not a['passed'] and len(a['records'])==458 and all(x['native_residual_N']<1e-8 for x in a['records'])
cp=np.load(P/p['canonical_checkpoint']);assert int(cp['completed_steps'])==458
for name in ['u','u_old','v_old','a_old']:assert np.isfinite(cp[name]).all()
np.testing.assert_array_equal(cp['u'],cp['u_old']);fields=np.load(root/'displacement_coefficients.npy',mmap_mode='r');np.testing.assert_array_equal(cp['u_old'],fields[457]);assert all(x['native_residual_N']<1e-8 for x in q['repeated_independent_force_checks'])
print('Candidate/original provenance verified; NEW unchanged-tolerance full native audit still required.')
