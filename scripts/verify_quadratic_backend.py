"""Independent CPU/GPU P2 weak-form and derivative verification, no training."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'src'))
from lv_surrogate.physics.quadratic_residual import CardiacQuadratic

torch.set_num_threads(2)
manifest = json.loads((P/'experiments/metrics/quadratic_backend_oracle.json').read_text())
oracle = P/manifest['oracle_path']
assert manifest['complete'] and hashlib.sha256(oracle.read_bytes()).hexdigest() == manifest['oracle_sha256']
arrays = dict(np.load(oracle))
dt = float(arrays['dt'])
records = []
levels = [('cpu64', torch.float64, 'cpu'), ('cpu32', torch.float32, 'cpu')]
if torch.backends.mps.is_available():
    levels.append(('mps32', torch.float32, 'mps'))


def relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


for label, dtype, device in levels:
    mechanics = CardiacQuadratic(arrays, dtype=dtype, dt=dt).to(device)
    for i, metadata in enumerate(manifest['cases']):
        u = torch.tensor(arrays['case_u'][i], dtype=dtype, device=device, requires_grad=True)
        history = {k: torch.tensor(arrays['case_'+k][i], dtype=dtype, device=device)
                   for k in ['u_old','v_old','a_old']}
        direction = torch.tensor(arrays['case_direction'][i], dtype=dtype, device=device)
        pressure = float(arrays['case_pressure'][i])
        R = mechanics(u, pressure, create_graph=True, history=history)
        gradient = torch.autograd.grad(.5*R.square().sum(), u)[0]
        force_error = relative(R.detach().cpu().numpy(), arrays['case_residual'][i])
        gradient_error = relative(gradient.detach().cpu().numpy(), arrays['case_gradient'][i])
        record = dict(device_precision=label, frame=metadata['frame'],
            force_relative_error=force_error, loss_gradient_relative_error=gradient_error,
            minimum_endpoint_quadrature_detF=float(mechanics.detF(u).min().detach().cpu()))
        if label == 'cpu64':
            _, jv = torch.autograd.functional.jvp(
                lambda z: mechanics(z, pressure, create_graph=True, history=history),
                (u.detach(),), (direction,))
            record['tangent_action_relative_error'] = relative(jv.detach().numpy(), arrays['case_Jv'][i])
            def coupled(eta):
                hh = {k: history[k]+eta*mult*direction for k, mult in
                      [('u_old', .9), ('v_old', .1/dt), ('a_old', .05/dt**2)]}
                return mechanics(u.detach()+eta*direction, pressure, create_graph=True, history=hh)
            _, derivative = torch.autograd.functional.jvp(coupled,
                torch.zeros((), dtype=dtype), torch.ones((), dtype=dtype))
            record['coupled_history_action_relative_error_vs_native_FD'] = relative(
                derivative.detach().numpy(), arrays['case_history_chain_FD'][i])
            # Check the pointwise analytic Piola independently against dW/dF.
            ua = mechanics.alpha_f*history['u_old']+(1-mechanics.alpha_f)*u.detach()
            F = (mechanics.I+torch.einsum('eai,eqaj->eqij', ua[mechanics.maps], mechanics.grad)).detach().requires_grad_(True)
            automatic = torch.autograd.grad(mechanics.energy_density(F).sum(), F)[0]
            analytic = mechanics.piola(F)
            record['piola_relative_error_vs_energy_autodiff'] = relative(
                analytic.detach().numpy(), automatic.detach().numpy())
            record['passed'] = bool(max(force_error, gradient_error,
                record['tangent_action_relative_error'], record['piola_relative_error_vs_energy_autodiff']) < 1e-7
                and record['coupled_history_action_relative_error_vs_native_FD'] < 1e-4)
        else:
            record['passed'] = bool(max(force_error, gradient_error) < .01)
        records.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        del u, R, gradient
    del mechanics
result = dict(complete=True, passed=all(r['passed'] for r in records),
    mps_available=torch.backends.mps.is_available(), records=records,
    native_oracle_sha256=manifest['oracle_sha256'], native_exporter_sha256=manifest['source_sha256'],
    executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    operator_sha256=hashlib.sha256((P/'src/lv_surrogate/physics/quadratic_residual.py').read_bytes()).hexdigest(),
    declared_tolerances=dict(cpu64_force_tangent_gradient_piola=1e-7, native_history_FD=1e-4, float32_force_gradient=.01),
    scope='Standalone passive P2 operator on three fixed perturbed baseline fields with nonzero deformed histories. Native P2 weak forms supply force, tangent actions and transpose-gradient oracles. Combined-history derivatives use centered native force differences. CPU32 and MPS32 checks quantify finite-precision behavior at these fields, not uniform error bounds or accuracy near exactly vanishing residuals. Volume and boundary P2 quadrature, follower pressure, Robin terms, viscosity and source Newmark inertia are included. No neural model or calibration changed; a trained P2-network directional-gradient check remains required before any new physics-loss result.')
(P/'experiments/metrics/quadratic_backend.json').write_text(json.dumps(result, indent=2, allow_nan=False))
assert result['passed'], 'P2 backend verification failed; retain diagnostic metrics and do not train'
