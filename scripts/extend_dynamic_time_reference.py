"""Add the exact1s endpoint without altering the archived999-step reference."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P/'src'))
from lv_surrogate.physics.reference_fem import (setup, assign_nodes, assign_history, residual,
    pressure_function, default_parameters_benchmark1, vertex_map)

source = P/'data/processed/cardiac_passive_trajectory_v1'
state_path = source/'state.npz'
last_path = source/'step_0998.npz'
out = P/'data/processed/cardiac_dynamic_endpoint_1s'
out.mkdir(exist_ok=True)
metric = P/'experiments/metrics/dynamic_time_endpoint_1s.json'
hashes = {str(p.relative_to(P)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [state_path, last_path]}
if metric.exists():
    old = json.loads(metric.read_text())
    assert old['predecessor_sha256'] == hashes
    assert hashlib.sha256((out/'endpoint.npz').read_bytes()).hexdigest() == old['endpoint_sha256']
    print('REUSED VERIFIED1s ENDPOINT', flush=True)
    sys.exit(0)
state = dict(np.load(state_path)); last = dict(np.load(last_path))
assert int(state['step']) == 998 and float(last['time']) == .999
np.testing.assert_array_equal(state['u'], last['u'])
geo, _, problem = setup(dt=.001)
assign_nodes(problem, state['u'])
assign_history(problem, state['u'], state['v'], state['a'])
pressures = pressure_function((0, 1), parameters=default_parameters_benchmark1(),
                             t_eval=np.array([.999, 1.0])-.4*.001)
np.testing.assert_allclose(pressures[0], last['pressure'], rtol=0, atol=1e-8)
problem.parameters['p'].assign(float(pressures[1]))
tick = time.perf_counter(); iterations, _ = problem.solver.solve(); seconds = time.perf_counter()-tick
reason = int(problem.solver._snes.getConvergedReason())
force = float(residual(problem).norm('l2'))
u = problem.u.vector().get_local()[vertex_map(problem)].reshape(-1, 3)
X = geo.mesh.coordinates(); C = geo.mesh.cells()
inverse = np.linalg.inv((X[C[:, 1:]]-X[C[:, :1]]).transpose(0, 2, 1))
J = np.linalg.det(np.eye(3)+np.einsum('eij,ejk->eik',
                 (u[C[:, 1:]]-u[C[:, :1]]).transpose(0, 2, 1), inverse))
assert reason>0 and np.isfinite(force) and force<1e-6
assert np.isfinite(u).all() and np.isfinite(J).all() and J.min()>0
np.savez_compressed(out/'endpoint.npz', u=u, u_old=state['u'], v_old=state['v'],
                    a_old=state['a'], pressure=float(pressures[1]), time=1., dt=.001)
result = dict(complete=True, predecessor_sha256=hashes,
    endpoint_sha256=hashlib.sha256((out/'endpoint.npz').read_bytes()).hexdigest(),
    time_s=1., dt_s=.001, SNES_reason=reason, residual_l2_N=force,
    minimum_detF=float(J.min()), nonlinear_seconds=seconds,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='One exact history continuation of archived P1/passive1ms trajectory to1s. '
          'Original999 frames/state remain unchanged; not another independent geometry.')
metric.write_text(json.dumps(result, indent=2, allow_nan=False))
print('VERIFIED1s ENDPOINT COMPLETE', result, flush=True)
