"""Compare the streaming proof implementation with four independent serial audits."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.bernstein import BernsteinAudit
stems=['dynamic_quadratic_third_history_resume_v1','dynamic_quadratic_third_stress_predictor_pilot',
       'dynamic_quadratic_third_stress_predictor_consecutive','dynamic_quadratic_third_best_predictor_pilot']
audit=None;records=[]
for stem in stems:
    p=json.loads((P/f'configs/physics/{stem}.json').read_text())
    ref=json.loads((P/f'experiments/metrics/{stem}_verification_resolved.json').read_text());assert ref['passed']
    parent=json.loads((P/p['resume_probe_protocol']).read_text());old=np.load(P/parent['checkpoint'])['u_old']
    if audit is None:audit=BernsteinAudit(np.load(P/'data/processed'/p['reference']/'inputs.npz'))
    u=np.load(P/p['fields'],mmap_mode='r')[0];tick=time.perf_counter();result=audit.pair(u,old)
    for state in ['endpoint','stress']:
        np.testing.assert_allclose(result[state]['minimum_Bernstein_bound'],ref['records'][0][state+'_minimum_Bernstein_bound'],rtol=1e-8,atol=1e-10)
        assert result[state]['original_unresolved_elements']==ref['records'][0].get('original_'+state+'_unresolved_elements',0)
    records.append(dict(protocol=stem,passed=True,seconds=time.perf_counter()-tick,
        bounds={k:v['minimum_Bernstein_bound'] for k,v in result.items()}))
    print('CHUNKED WHOLE-ELEMENT CHECK',records[-1],flush=True)
out=dict(complete=True,passed=True,records=records,positive_margin=1e-9,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    operator_sha256=hashlib.sha256((P/'src/lv_surrogate/physics/bernstein.py').read_bytes()).hexdigest(),
    scope='Chunked memory-bounded algebraic positivity matches four retained independently assembled/native-map serial audits, including their original unresolved counts and recursively certified bounds. Same1e-9 float64 margin; not interval arithmetic or full-reference qualification.')
(P/'experiments/metrics/quadratic_chunked_positivity.json').write_text(json.dumps(out,indent=2)+'\n')
