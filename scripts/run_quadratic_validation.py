"""Execute one controlled FE refinement, verify norms and refresh existing report."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

P = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('factor', choices=['time', 'mesh', 'solver'])
args = ap.parse_args()
baseline = 'configs/physics/dynamic_quadratic_reference_dt00125.json'
candidate = 'configs/physics/dynamic_quadratic_reference_'+dict(time='dt001',mesh='refined',solver='gamg')[args.factor]+'.json'
name = json.loads((P/candidate).read_text())['name']
stem = 'dynamic_quadratic_controlled_'+args.factor
output = P/'experiments/metrics'/f'{stem}_pipeline.json'
lock = (P/'experiments/logs'/f'{stem}.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX)
env = os.environ.copy()
env.update(FI_PROVIDER='tcp', FI_TCP_IFACE='en0', OMP_NUM_THREADS='2',
    OPENBLAS_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2',
    XDG_CACHE_HOME=str(P/'.cache'), MAMBA_ROOT_PREFIX=str(P/'.mamba'),
    DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'), MPLCONFIGDIR=str(P/'.cache/matplotlib'))
native = [str(P/'.tools/bin/micromamba'), 'run', '-p', str(P/'.fenics'), 'python']
modern = [str(P/'.venv/bin/python')]
state = dict(complete=False, factor=args.factor, reference=name, phases=[],
    config_sha256={c: hashlib.sha256((P/c).read_bytes()).hexdigest() for c in [baseline, candidate]},
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope='Controlled numerical refinement only. Does not refit models or calibrate uncertainty.')


def persist():
    output.write_text(json.dumps(state, indent=2, allow_nan=False))


def run(label, command):
    state['active_phase'] = label
    persist()
    start = time.perf_counter()
    with (P/'experiments/logs'/f'{stem}_{label}.log').open('a') as log:
        completed = subprocess.run(command, cwd=P, env=env, stdout=log, stderr=subprocess.STDOUT)
    state['phases'].append(dict(phase=label, return_code=completed.returncode,
                               measured_wall_seconds=time.perf_counter()-start))
    persist()
    if completed.returncode:
        raise RuntimeError(f'{label} failed with return code {completed.returncode}; see preserved log')


persist()
try:
    if (P/'data/processed'/name/'failure.json').exists():
        raise RuntimeError('A rejected trajectory is preserved; automatic retries are disabled')
    if args.factor == 'mesh':
        progress = P/'data/processed'/name/'manifest.json'
        completed_reference = progress.exists() and json.loads(progress.read_text()).get('complete')
        if not completed_reference:
            run('startup', native+['scripts/generate_quadratic_reference.py', '--config', candidate,
                                  '--max-steps', '2'])
            run('transfer_pilot', native+['scripts/compare_quadratic_references.py',
                '--config-a', baseline, '--config-b', candidate, '--pilot',
                '--output', f'experiments/metrics/{stem}_pilot.json'])
    # Native generator locks its own reference directory and safely reuses an
    # independently running matching generation after checking all output hashes.
    run('generation', native+['scripts/generate_quadratic_reference.py', '--config', candidate])
    if args.factor == 'solver':
        run('equivalence', modern+['scripts/compare_quadratic_solvers.py'])
    else:
        run('norms', native+['scripts/compare_quadratic_references.py', '--config-a', baseline,
            '--config-b', candidate, '--output', f'experiments/metrics/{stem}.json'])
    run('positivity', modern+['scripts/audit_quadratic_positivity.py', '--config', candidate])
    run('figures', modern+['scripts/plot_quadratic_validation.py'])
    state['complete'] = True  # Workflow completed; sensitivity tolerance may FAIL.
    state['active_phase'] = None
    persist()
    run('report', modern+['scripts/write_reports.py'])
    state['active_phase'] = None
    persist()
except Exception as error:
    state['complete'] = False
    state['failure'] = str(error)
    persist()
    subprocess.run(modern+['scripts/write_reports.py'], cwd=P, env=env)
    raise
