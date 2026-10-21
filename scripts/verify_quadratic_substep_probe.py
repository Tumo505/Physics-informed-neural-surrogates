"""Independent native and whole-element verification of the saved local probe."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from FIAT.quadrature_schemes import create_quadrature
from FIAT.reference_element import UFCTetrahedron

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'src'))
from lv_surrogate.physics.reference_fem import setup, residual, df, pressure_function, default_parameters_benchmark1
from lv_surrogate.physics.quadratic_solver import make_quadratic_problem
from lv_surrogate.physics.quadratic import QuadraticKinematics

ap = argparse.ArgumentParser()
ap.add_argument('--protocol', default='configs/physics/dynamic_quadratic_refined_substep_extended_probe.json')
ap.add_argument('--output')
ap.add_argument('--accepted-prefix', action='store_true')
args = ap.parse_args()
protocol_path = P/args.protocol
protocol = json.loads(protocol_path.read_text())
probe = json.loads((P/protocol['output']).read_text())
assert probe['finished']
if not args.accepted_prefix:
    assert probe['all_requested_steps_accepted']
assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == probe['protocol_sha256']
assert hashlib.sha256((P/protocol['checkpoint']).read_bytes()).hexdigest() == probe['probe_checkpoint_sha256']
root = P/'data/processed'/protocol['reference']
m = json.loads((root/'manifest.json').read_text()); cfg = m['config']
assert hashlib.sha256((root/'restart.npz').read_bytes()).hexdigest() == probe['checkpoint_sha256']
saved = np.load(root/'restart.npz'); data = np.load(root/'inputs.npz')
if 'resume_probe_protocol' in protocol:
    previous_protocol = json.loads((P/protocol['resume_probe_protocol']).read_text())
    resumed_path = P/previous_protocol['checkpoint']
    assert hashlib.sha256(resumed_path.read_bytes()).hexdigest() == probe['resumed_checkpoint_sha256']
    saved = np.load(resumed_path)
u = np.load(P/protocol['fields'], mmap_mode='r')
assert len(u) == len(probe['records']) == protocol['steps_after_checkpoint']
if args.accepted_prefix:
    count = 0
    for r in probe['records']:
        if not r['accepted']:
            break
        count += 1
    assert count>0
    u = u[:count]
    probe = dict(probe,records=probe['records'][:count])
dt = probe['dt_s']
geo, material, base = setup(dt=dt, refinement_level=cfg['refinement_level'], geometry_scales=cfg['geometry_scales'])
parameters = dict(base.parameters); parameters['function_space'] = 'P_2'
df.parameters['form_compiler']['quadrature_degree'] = cfg['quadrature_degree']
problem = make_quadratic_problem(geo, material, parameters, 'mumps')
np.testing.assert_allclose(problem.u_space.tabulate_dof_coordinates(), data['dof_coordinates'], atol=1e-14, rtol=0)
np.testing.assert_array_equal(geo.mesh.cells(), data['cells'])
for name in ['u', 'u_old', 'v_old', 'a_old']:
    field = getattr(problem, name); field.vector().set_local(saved[name]); field.vector().apply('insert')
if 'qualification_step' in protocol:
    assert m['complete'] and not (root/'failure.json').exists()
    for name, digest in m['output_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    # Independent native history replay, rather than the probe's NumPy recurrence.
    for name in ['u', 'u_old', 'v_old', 'a_old']:
        getattr(problem, name).vector().zero()
    archived = np.load(root/'displacement_coefficients.npy', mmap_mode='r')
    for current in archived[:protocol['qualification_step']]:
        problem.u.vector().set_local(current); problem.u.vector().apply('insert')
        problem._update_fields()
    expected = archived[protocol['qualification_step']]
    np.testing.assert_allclose(u[0], expected, rtol=1e-6, atol=1e-9)
kin = QuadraticKinematics(problem.u_space, data['qpoints'], data['qweights'])
np.testing.assert_array_equal(kin.maps, data['canonical_cell_dofs'])
gradients = np.stack([np.concatenate(((4*bary[None,:,None]-1)*kin.grad_lambda,
    np.stack([4*(bary[i]*kin.grad_lambda[:,j]+bary[j]*kin.grad_lambda[:,i])
              for i,j in kin.edges],axis=1)),axis=1) for bary in np.eye(4)],axis=1)
multi = np.array([a for a in itertools.product(range(4),repeat=4) if sum(a)==3])
multinomial = np.array([math.factorial(3)/math.prod(math.factorial(int(i)) for i in a) for a in multi])
def basis(bary):
    return multinomial[None,:]*np.prod(np.asarray(bary)[:,None,:]**multi[None,:,:],axis=-1)
lattice = multi/3; transform = np.linalg.inv(basis(lattice))
check_bary = np.random.default_rng(2027).dirichlet(np.ones(4),size=7)
q = create_quadrature(UFCTetrahedron(),4)
bary = np.c_[1-q.get_points().sum(1),q.get_points()]
shape = np.concatenate((bary*(2*bary-1),np.stack([4*bary[:,i]*bary[:,j] for i,j in kin.edges],axis=1)),axis=1)
records = []; fixed = {0, len(u)//3, 2*len(u)//3, len(u)-1}
times = np.array([r['time_s'] for r in probe['records']])
if not probe.get('changed_time_discretization'):
    indices = probe['start_step']+np.arange(len(u))
    np.testing.assert_allclose(times,data['time'][indices],atol=1e-12,rtol=0)
    pressures = data['pressure'][indices]
else:
    par = default_parameters_benchmark1()
    for key in ['t_sys_pre','t_dias_pre','gamma']:
        par[key] *= cfg['timing_scale']
    pressures = np.atleast_1d(cfg['amplitude']*pressure_function((0,cfg['duration_s']),parameters=par,t_eval=times-.4*dt))
for i,current in enumerate(u):
    old = problem.u_old.vector().get_local().copy()
    problem.u.vector().set_local(current); problem.u.vector().apply('insert')
    problem.parameters['p'].assign(float(pressures[i]))
    R = float(residual(problem).norm('l2')); assert np.isfinite(R) and R<1e-6
    record = dict(time_s=float(times[i]),native_residual_N=R)
    for state,field in [('endpoint',current),('stress',.4*old+.6*current)]:
        vertex_F = np.eye(3)+np.einsum('eki,evkj->evij',field[kin.maps],gradients)
        coefficients = np.linalg.det(np.einsum('qv,evij->eqij',lattice,vertex_F))@transform.T
        assert np.isfinite(coefficients).all()
        lower = coefficients.min(axis=1)
        record[state+'_minimum_Bernstein_bound'] = float(lower.min())
        record[state+'_unresolved_elements'] = int((lower<=1e-9).sum())
        if i in fixed:
            direct = np.linalg.det(np.einsum('qv,evij->eqij',check_bary,vertex_F))
            np.testing.assert_allclose(direct,coefficients@basis(check_bary).T,rtol=1e-11,atol=1e-11)
    if i in fixed:
        values = np.einsum('qk,ekc->eqc',shape,current[kin.maps])
        independent = float(np.einsum('e,q,eqc,eqc->',kin.measure,q.get_weights(),values,values))
        native = float(df.assemble(df.inner(problem.u,problem.u)*df.dx(metadata={'quadrature_degree':4})))
        np.testing.assert_allclose(independent,native,rtol=1e-11,atol=1e-24)
        record['independent_L2_norm_relative_error'] = abs(independent-native)/max(native,1e-30)
    records.append(record); problem._update_fields()
    print('VERIFIED LOCAL PROBE',i+1,'/',len(u),flush=True)
passed = all(r['endpoint_unresolved_elements']==r['stress_unresolved_elements']==0 for r in records)
final_checkpoint=np.load(P/protocol['checkpoint'])
history_checks={}
for name in ['u','u_old','v_old','a_old']:
    actual=getattr(problem,name).vector().get_local();expected=final_checkpoint[name]
    np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-10)
    history_checks[name]=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30))
result = dict(complete=True,passed=passed,full_cycle=False,records=records,dt_s=dt,
    final_checkpoint_history_relative_errors=history_checks,
    accepted_prefix_only=args.accepted_prefix,
    changed_time_discretization=probe.get('changed_time_discretization'),
    protocol_sha256=probe['protocol_sha256'], original_checkpoint_sha256=probe['checkpoint_sha256'],
    fields_sha256=hashlib.sha256((P/protocol['fields']).read_bytes()).hexdigest(),
    inputs_sha256=hashlib.sha256((root/'inputs.npz').read_bytes()).hexdigest(),
    executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope=f'{len(u)} local probe steps with inherited nonzero histories, not a full-cycle reference. The recorded protocol specifies whether time discretization changed. All native forces are independently reassembled; P2 coordinate/component maps match native space. Endpoint/stress whole-element cubic Bernstein bounds checked with1e-9 float64 margin; random polynomial identities and independent canonical-quadrature/UFL displacement norms checked at {len(fixed)} fixed frames. Not an interval-arithmetic proof, mesh-convergence or continuum claim.')
default_output = ('experiments/metrics/dynamic_quadratic_refined_substep_extended_verification.json'
    if args.protocol == 'configs/physics/dynamic_quadratic_refined_substep_extended_probe.json'
    else str(Path(protocol['output']).with_name(Path(protocol['output']).stem+'_verification.json')))
(P/(args.output or default_output)).write_text(json.dumps(result,indent=2,allow_nan=False))
print('LOCAL VERIFICATION PASSED',passed,flush=True)
