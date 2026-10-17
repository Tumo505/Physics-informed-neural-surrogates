"""Execute a fixed validation geometry/loading pair at two native P2 mesh levels."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

P = Path(__file__).resolve().parents[1]
protocol_path = P/'configs/physics/dynamic_quadratic_geometry_controls.json'
protocol = json.loads(protocol_path.read_text())
ap = argparse.ArgumentParser()
ap.add_argument('--case', required=True, choices=protocol['case_ids'])
args = ap.parse_args()
source_path = P/protocol['source_cases']
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == protocol['source_cases_sha256']
case = next(c for c in json.loads(source_path.read_text())['cases'] if c['id'] == args.case)
assert case['split'] == 'validation'
lock = (P/f'experiments/logs/quadratic_geometry_{args.case}.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX)
derived = P/'data/processed/quadratic_geometry_controls/configs'/args.case
derived.mkdir(parents=True, exist_ok=True)
base = json.loads((P/'configs/physics/dynamic_quadratic_reference_gamg.json').read_text())
configs = []
for level in protocol['refinement_levels']:
    cfg = dict(base)
    cfg.update(name=f'cardiac_quadratic_{args.case}_r{level}_q{protocol["quadrature_degree"]}_v1',
        dt_s=protocol['dt_s'], quadrature_degree=protocol['quadrature_degree'],
        native_solver=protocol['native_solver'], refinement_level=level,
        geometry_scales=case['scales'], amplitude=case['amplitude'], timing_scale=case['timing_scale'],
        validation_source_case=case, validation_protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        scope=protocol['scope'])
    path = derived/f'refinement_{level}.json'
    path.write_text(json.dumps(cfg, indent=2, allow_nan=False)+'\n')
    configs.append(str(path.relative_to(P)))
state_path = P/f'experiments/logs/quadratic_geometry_{args.case}_workflow.json'
state = dict(complete=False, case=case, derived_configs=configs, phases=[],
    protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
env = os.environ.copy()
env.update(FI_PROVIDER='tcp', FI_TCP_IFACE='en0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
    VECLIB_MAXIMUM_THREADS='2', XDG_CACHE_HOME=str(P/'.cache'), MAMBA_ROOT_PREFIX=str(P/'.mamba'),
    DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'), MPLCONFIGDIR=str(P/'.cache/matplotlib'))
native = [str(P/'.tools/bin/micromamba'),'run','-p',str(P/'.fenics'),'python']
modern = [str(P/'.venv/bin/python')]


def persist():
    state_path.write_text(json.dumps(state, indent=2, allow_nan=False))


def run(phase, command):
    state['active_phase'] = phase; persist(); started = time.perf_counter()
    with (P/f'experiments/logs/quadratic_geometry_{args.case}_{phase}.log').open('a') as log:
        result = subprocess.run(command, cwd=P, env=env, stdout=log, stderr=subprocess.STDOUT)
    state['phases'].append(dict(phase=phase, return_code=result.returncode,
                               wall_seconds=time.perf_counter()-started)); persist()
    if result.returncode:
        raise RuntimeError(f'{phase} failed with exit code{result.returncode}; reference failure/log retained')


persist()
try:
    for config in configs:
        name = json.loads((P/config).read_text())['name']
        if (P/'data/processed'/name/'failure.json').exists():
            raise RuntimeError(f'Rejected native attempt retained for{name}; automatic retry disabled')
    run('coarse', native+['scripts/generate_quadratic_reference.py','--config',configs[0]])
    run('coarse_positivity', modern+['scripts/audit_quadratic_positivity.py','--config',configs[0]])
    run('mesh_control', modern+['scripts/run_quadratic_validation.py','mesh',
        '--baseline-config',configs[0],'--candidate-config',configs[1],
        '--output-stem',f'dynamic_quadratic_validation_{args.case}'])
    state['complete'] = True; state['active_phase'] = None; persist()
    run('report', modern+['scripts/write_reports.py'])
    state['active_phase'] = None; persist()
except Exception as error:
    state['complete'] = False; state['failure'] = str(error); persist()
    subprocess.run(modern+['scripts/write_reports.py'], cwd=P, env=env)
    raise
