"""Native P2 force/tangent/history oracles at three fixed verification states."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from FIAT.reference_element import UFCTriangle
from FIAT.quadrature_schemes import create_quadrature

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'src'))
from lv_surrogate.physics.reference_fem import setup, LVProblem, export_geometry, df, residual
from lv_surrogate.physics.quadratic import QuadraticKinematics

root = P/'data/processed/cardiac_quadratic_reference_dt00125_v1'
m = json.loads((root/'manifest.json').read_text()); assert m['complete']
cfg = m['config']; dt = cfg['dt_s']
inputs = np.load(root/'inputs.npz')
u = np.load(root/'displacement_coefficients.npy', mmap_mode='r')
geo, material, base = setup(dt=dt, geometry_scales=cfg['geometry_scales'])
params = dict(base.parameters); params['function_space'] = 'P_2'
problem = LVProblem(geo, material, parameters=params,
    solver_parameters={'linear_solver':'mumps','petsc':{'ksp_type':'preonly','pc_type':'none'}})
np.testing.assert_allclose(problem.u_space.tabulate_dof_coordinates(), inputs['dof_coordinates'], rtol=0, atol=1e-14)
arrays = export_geometry(geo, material, base)
arrays['canonical_cell_dofs'] = inputs['canonical_cell_dofs']
triangle = create_quadrature(UFCTriangle(), 6)
arrays['face_qpoints'] = triangle.get_points(); arrays['face_qweights'] = triangle.get_weights()
arrays['activation'] = 0.; arrays['dt'] = dt
for key in ['alpha_f','alpha_m','alpha_epi','alpha_top','beta_epi','beta_top','rho']:
    arrays[key] = float(problem.parameters[key])
kin = QuadraticKinematics(problem.u_space, arrays['qpoints'], arrays['qweights'])
affine = df.interpolate(df.Expression(('0.001*x[0]+0.002*x[1]',
    '-0.001*x[1]+0.001*x[2]', '0.001*x[0]-0.001*x[2]'), degree=1), problem.u_space).vector().get_local()
previous = np.zeros(u.shape[1]); velocity = previous.copy(); acceleration = previous.copy()
beta = float(problem._beta); gamma = float(problem._gamma)
frames = [0, 223, 399]; cases = []; metadata = []


def assign(field, value):
    field.vector().set_local(value); field.vector().apply('insert')


for i in range(max(frames)+1):
    current = u[i]
    if i in frames:
        for key, value in [('u', current),('u_old', previous),('v_old', velocity),('a_old', acceleration)]:
            assign(getattr(problem, key), value)
        problem.parameters['p'].assign(float(inputs['pressure'][i]))
        known_R = float(residual(problem).norm('l2')); assert known_R < 1e-6
        trial = .997*current+affine
        assign(problem.u, trial)
        minimum, _ = kin.audit(.4*previous+.6*trial); assert minimum > 0
        native_R = residual(problem)
        native_J = df.assemble(problem._problem._J)
        rng = np.random.default_rng(1442027+i)
        direction = rng.standard_normal(len(current)); direction /= np.linalg.norm(direction)
        d = problem.u.vector().copy(); d.set_local(direction); d.apply('insert')
        jv = native_R.copy(); native_J.mult(d, jv)
        gradient = native_R.copy(); df.as_backend_type(native_J).mat().multTranspose(
            df.as_backend_type(native_R).vec(), df.as_backend_type(gradient).vec())
        fields = [('u', trial, 1.), ('u_old', previous, .9),
                  ('v_old', velocity, .1/dt), ('a_old', acceleration, .05/dt**2)]
        eps = 1e-6; residuals = []
        for sign in [1, -1]:
            for key, value, multiplier in fields:
                assign(getattr(problem, key), value+sign*eps*multiplier*direction)
            residuals.append(residual(problem).get_local())
        for key, value, multiplier in fields:
            assign(getattr(problem, key), value)
        cases.append(dict(u=trial, u_old=previous.copy(), v_old=velocity.copy(), a_old=acceleration.copy(),
            direction=direction, residual=native_R.get_local(), Jv=jv.get_local(),
            gradient=gradient.get_local(), history_chain_FD=(residuals[0]-residuals[1])/(2*eps)))
        metadata.append(dict(frame=i, time_s=float(inputs['time'][i]), pressure_Pa=float(inputs['pressure'][i]),
            known_reference_residual_l2_N=known_R, minimum_stress_sampled_detF=minimum,
            native_trial_residual_l2_N=float(native_R.norm('l2')), history_chain_FD_epsilon=eps))
        print('EXPORTED P2 ORACLE', i, flush=True)
    new_a = (current-(previous+dt*velocity+(.5-beta)*dt**2*acceleration))/(beta*dt**2)
    velocity = velocity+(1-gamma)*dt*acceleration+gamma*dt*new_a
    acceleration = new_a; previous = current
for key in cases[0]:
    arrays['case_'+key] = np.stack([c[key] for c in cases])
arrays['case_pressure'] = np.array([r['pressure_Pa'] for r in metadata])
target = P/'experiments/logs/quadratic_backend_cases.npz'
np.savez_compressed(target, **arrays)
record = dict(complete=True, frames=frames, cases=metadata,
    oracle_path=str(target.relative_to(P)), oracle_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    reference_output_sha256=m['output_sha256'],
    scope='Three predeclared same-geometry P2 states: startup and nonzero deformed histories. Trial u=.997*u_reference plus the fixed small affine perturbation; these are positive-J operator-verification fields, not new FE labels. Exact native force/Jv/J-transpose-R and centered coupled-history force differences are exported. No neural models or test-case selection changed.')
(P/'experiments/metrics/quadratic_backend_oracle.json').write_text(json.dumps(record, indent=2, allow_nan=False))
