"""Wait for one existing probe, independently audit success and refresh the report."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

P = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--protocol', required=True)
args = ap.parse_args()
protocol = json.loads((P/args.protocol).read_text())
stem = Path(protocol['output']).stem
state_path = P/f'experiments/logs/{stem}_audit_workflow.json'
state = dict(phase='waiting_for_probe', full_cycle_validated=False)
def persist():
    state_path.write_text(json.dumps(state,indent=2,allow_nan=False))
persist()
lock = (P/f'experiments/logs/{stem}.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX)
result = json.loads((P/protocol['output']).read_text())
fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
env = os.environ.copy()
env.update(FI_PROVIDER='tcp',FI_TCP_IFACE='en0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
    MAMBA_ROOT_PREFIX=str(P/'.mamba'),XDG_CACHE_HOME=str(P/'.cache'),
    DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'))
native = [str(P/'.tools/bin/micromamba'),'run','-p',str(P/'.fenics'),'python']
modern = [str(P/'.venv/bin/python')]
try:
    if result.get('finished') and result.get('all_requested_steps_accepted'):
        state['phase'] = 'independent_native_audit'; persist()
        output = f'experiments/metrics/{stem}_verification.json'
        with (P/f'experiments/logs/{stem}_verification.log').open('w') as log:
            subprocess.run(native+['scripts/verify_quadratic_substep_probe.py',
                '--protocol',args.protocol,'--output',output],cwd=P,env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        verification = json.loads((P/output).read_text())
        assert verification['complete'] and verification['passed'], 'Independent local audit failed'
        state.update(phase='local_recovery_verified',local_recovery_verified=True)
    else:
        state.update(phase='recovery_unsuccessful',local_recovery_verified=False)
except Exception as error:
    state.update(phase='audit_failed',failure=str(error),local_recovery_verified=False)
finally:
    persist()
    subprocess.run(modern+['scripts/write_reports.py'],cwd=P,env=env,check=True)
print(json.dumps(state,indent=2),flush=True)
