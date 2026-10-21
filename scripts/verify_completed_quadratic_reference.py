"""Check immutable P2 outputs and independent physical L2 norms, without solving."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import sys

import numpy as np
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import UFCTetrahedron

P = Path(__file__).resolve().parents[1]
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
sys.path.insert(0, str(P / 'src'))
from lv_surrogate.physics.reference_fem import setup, df
from lv_surrogate.physics.quadratic import QuadraticKinematics

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
ap.add_argument('--reassemble-force-subset',action='store_true')
ap.add_argument('--output')
args = ap.parse_args()
config = P / args.config
cfg = json.loads(config.read_text())
root = P / 'data/processed' / cfg['name']
m = json.loads((root / 'manifest.json').read_text())
assert m['complete'] and m['completed_steps'] == m['n_steps']
assert not (root / 'failure.json').exists()
assert m['config_sha256'] == hashlib.sha256(config.read_bytes()).hexdigest()
for name, digest in m['output_sha256'].items():
    assert file_sha(root/name) == digest
steps = m['steps']
assert len(steps) == m['n_steps']
assert all(np.isfinite(s['residual_l2_N']) and s['residual_l2_N']<1e-6 for s in steps)
lost_reason_checks=[]
coefficients=np.load(root/'displacement_coefficients.npy',mmap_mode='r')
for s in steps:
    if s['SNES_reason'] is not None:
        assert s['SNES_reason']>0
        continue
    assert s.get('native_reason_record_lost') and s.get('native_acceptance_provenance')
    p=s['native_acceptance_provenance']
    for key,digest in [('failure_path','failure_sha256'),('independent_audit','independent_audit_sha256'),('accepted_fields','accepted_fields_sha256')]:
        assert file_sha(P/p[key])==p[digest]
    failure=json.loads((P/p['failure_path']).read_text());audit=json.loads((P/p['independent_audit']).read_text())
    assert failure['persisted_checkpoint_steps']<=s['step']<failure['accepted_steps_before_failure']
    assert audit['complete'] and audit['passed'] and audit['records'][p['audit_record']]['native_residual_N']<1e-6
    np.testing.assert_array_equal(coefficients[s['step']],np.load(P/p['accepted_fields'],mmap_mode='r')[p['audit_record']])
    lost_reason_checks.append(dict(step=s['step'],positive_native_acceptance_count_proven=True,individual_SNES_reason_recreated=False))
b = json.loads((P / 'experiments/metrics' / (cfg['name'] + '_bernstein.json')).read_text())
assert b['complete'] and b['checked_steps'] == m['n_steps']
assert b['output_sha256'] == m['output_sha256']
positivity_refinement = None
endpoint_bound=b['minimum_endpoint_lower_bound'];stress_bound=b['minimum_stress_lower_bound']
if not b['whole_element_positivity_sufficient_condition_passed']:
    rp=P/'experiments/metrics'/(cfg['name']+'_bernstein_refinement.json')
    refinement=json.loads(rp.read_text())
    assert refinement['complete'] and refinement['passed']
    assert refinement['original_audit_sha256']==hashlib.sha256((P/'experiments/metrics'/(cfg['name']+'_bernstein.json')).read_bytes()).hexdigest()
    assert refinement['output_sha256']==m['output_sha256'] and refinement['config_sha256']==m['config_sha256']
    assert refinement['positive_margin']==b['positive_margin']
    for state in ['endpoint','stress']:
        assert sum(r['original_unresolved_elements'] for r in refinement['records'] if r['state']==state)==b['unresolved_'+state+'_element_states']
    assert all(e['classification']=='positive_bound' and e['lower_bound']>b['positive_margin'] for r in refinement['records'] for e in r['elements'])
    endpoint_bound=refinement['merged_minimum_lower_bound']['endpoint']
    stress_bound=refinement['merged_minimum_lower_bound']['stress']
    positivity_refinement=dict(path=str(rp.relative_to(P)),sha256=hashlib.sha256(rp.read_bytes()).hexdigest())
geo, _, _ = setup(dt=cfg['dt_s'], refinement_level=cfg['refinement_level'],
                  geometry_scales=cfg['geometry_scales'])
V = df.VectorFunctionSpace(geo.mesh, 'CG', 2)
inputs = np.load(root / 'inputs.npz')
np.testing.assert_allclose(V.tabulate_dof_coordinates(), inputs['dof_coordinates'], atol=1e-14, rtol=0)
np.testing.assert_array_equal(geo.mesh.cells(), inputs['cells'])
q = create_quadrature(UFCTetrahedron(), 4)
kin = QuadraticKinematics(V, q.get_points(), q.get_weights())
np.testing.assert_array_equal(kin.maps, inputs['canonical_cell_dofs'])
bary = kin.quad_bary
basis = np.concatenate((bary * (2*bary-1), np.stack(
    [4*bary[:, i]*bary[:, j] for i, j in kin.edges], axis=1)), axis=1)
u = np.load(root / 'displacement_coefficients.npy', mmap_mode='r')
field = df.Function(V)
checks = []
for i in np.linspace(0, len(steps)-1, 4, dtype=int):
    field.vector().set_local(u[i]); field.vector().apply('insert')
    values = np.einsum('qk,ekc->eqc', basis, u[i][kin.maps])
    independent = float(np.einsum('e,q,eqc,eqc->', kin.measure, kin.weights, values, values))
    native = float(df.assemble(df.inner(field, field) * df.dx(metadata={'quadrature_degree': 4})))
    np.testing.assert_allclose(independent, native, rtol=1e-11, atol=1e-24)
    checks.append(dict(step=int(i), time_s=steps[i]['time_s'],
                       physical_L2_squared_m5=native,
                       independent_relative_error=abs(independent-native)/max(native, 1e-30)))
result = dict(complete=True, reference_name=cfg['name'], passed=True,
              config_sha256=m['config_sha256'], output_sha256=m['output_sha256'],
              verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              accepted_steps=len(steps), maximum_native_residual_N=max(s['residual_l2_N'] for s in steps),
              minimum_endpoint_Bernstein_bound=endpoint_bound,
              minimum_stress_Bernstein_bound=stress_bound, positivity_refinement=positivity_refinement,norm_checks=checks,
              scope='Native convergence records, immutable hashes, whole-element float64 sufficient positivity audit and independent canonical P2 quadrature versus UFL norms. Does not establish mesh/time/quadrature convergence or continuum accuracy.')
result['lost_individual_reason_provenance_checks']=lost_reason_checks
if args.reassemble_force_subset:
    from lv_surrogate.physics.quadratic_solver import make_quadratic_problem
    from lv_surrogate.physics.reference_fem import residual
    geo,material,base=setup(dt=cfg['dt_s'],refinement_level=cfg['refinement_level'],geometry_scales=cfg['geometry_scales'])
    params=dict(base.parameters);params['function_space']='P_2'
    problem=make_quadratic_problem(geo,material,params,'gamg')
    chosen=set(np.linspace(0,len(steps)-1,4,dtype=int).tolist()+[s['step'] for s in lost_reason_checks[:1]])
    old=np.zeros(u.shape[1]);v=old.copy();a=old.copy();forces=[];dt=cfg['dt_s']
    for i,current in enumerate(u):
        if i in chosen:
            for name,value in [('u',current),('u_old',old),('v_old',v),('a_old',a)]:
                f=getattr(problem,name);f.vector().set_local(value);f.vector().apply('insert')
            problem.parameters['p'].assign(float(inputs['pressure'][i]));R=float(residual(problem).norm('l2'))
            assert np.isfinite(R) and R<1e-6
            _,integral=kin.audit(current)
            native=float(df.assemble(df.det(df.Identity(3)+df.grad(problem.u))*df.dx(metadata={'quadrature_degree':4})))
            np.testing.assert_allclose(integral,native,rtol=1e-11,atol=1e-14)
            forces.append(dict(step=i,native_residual_N=R,determinant_integral_relative_error=abs(integral-native)/max(abs(native),1e-30)))
        anew=(current-old-dt*v-(.5-.36)*dt**2*a)/(.36*dt**2)
        v=v+dt*(.3*a+.7*anew);a=anew;old=current.copy()
        if (i+1)%8==0 and hasattr(u._mmap,'madvise'):u._mmap.madvise(mmap.MADV_DONTNEED)
    result['independent_force_and_integral_checks']=forces
(P / 'experiments/manifests' / (cfg['name'] + '.json')).write_text(json.dumps(m, indent=2, allow_nan=False))
(P/(args.output or ('experiments/metrics/'+cfg['name']+'_verification.json'))).write_text(json.dumps(result, indent=2, allow_nan=False))
print(json.dumps(result, indent=2, allow_nan=False))
