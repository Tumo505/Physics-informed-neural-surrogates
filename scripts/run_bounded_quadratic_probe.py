"""Run one immutable probe and independent audit with process-tree resource guards."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

P = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--protocol', required=True)
ap.add_argument('--max-phase-seconds', type=float, default=4500)
ap.add_argument('--max-rss-gib', type=float, default=30)
ap.add_argument('--min-free-disk-gib', type=float, default=100)
args = ap.parse_args()
assert min(args.max_phase_seconds, args.max_rss_gib, args.min_free_disk_gib)>0
protocol = json.loads((P/args.protocol).read_text())
stem = Path(protocol['output']).stem
lock = (P/f'experiments/logs/{stem}_runner.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (P/protocol['output']).exists(), 'Preserve existing attempt; never auto-retry'
path = P/f'experiments/metrics/{stem}_workflow.json'
state = dict(complete=False, full_cycle_validated=False, protocol=args.protocol,
    limits=vars(args), phases=[])
def persist():
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(state, indent=2)); temporary.replace(path)
env = os.environ.copy()
env.update(FI_PROVIDER='tcp', FI_TCP_IFACE='en0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
    MAMBA_ROOT_PREFIX=str(P/'.mamba'), XDG_CACHE_HOME=str(P/'.cache'),
    DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'))
def run(phase, command):
    state['active_phase'] = phase; persist()
    tick = time.perf_counter(); peak = 0; stop = None
    with (P/f'experiments/logs/{stem}_{phase}.log').open('w') as log:
        child = subprocess.Popen(command, cwd=P, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                rows = subprocess.check_output(['ps','-axo','pid=,ppid=,rss='], text=True)
                processes = [tuple(map(int, row.split())) for row in rows.splitlines() if row.strip()]
                descendants = {child.pid}
                while True:
                    found = {pid for pid, parent, rss in processes if parent in descendants}
                    if found <= descendants: break
                    descendants.update(found)
                peak = max(peak, sum(rss*1024 for pid,parent,rss in processes if pid in descendants))
                if time.perf_counter()-tick > args.max_phase_seconds: stop = 'phase_wall_limit'
                if peak > args.max_rss_gib*2**30: stop = 'descendant_RSS_limit'
                if shutil.disk_usage(P).free < args.min_free_disk_gib*2**30: stop = 'free_disk_reserve'
                if stop: break
                time.sleep(2)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
        code = child.wait()
    state['phases'].append(dict(phase=phase, return_code=code, stop_reason=stop,
        wall_seconds=time.perf_counter()-tick, sampled_peak_RSS_bytes=peak))
    persist()
    if code or stop: raise RuntimeError(f'{phase}: {stop or code}')
try:
    native = [str(P/'.tools/bin/micromamba'), 'run', '-p', str(P/'.fenics'), 'python']
    run('solve', native+['scripts/probe_quadratic_strict_stopping.py', '--protocol', args.protocol])
    run('audit', [str(P/'.venv/bin/python'), 'scripts/finish_quadratic_probe.py', '--protocol', args.protocol])
    audit = json.loads((P/f'experiments/logs/{stem}_audit_workflow.json').read_text())
    state.update(complete=True, local_recovery_verified=audit.get('local_recovery_verified', False),
        active_phase='finished')
except Exception as error:
    state.update(complete=True, local_recovery_verified=False, failure=str(error), active_phase='stopped')
finally:
    persist()
    subprocess.run([str(P/'.venv/bin/python'), 'scripts/write_reports.py'], cwd=P, env=env, check=True)
print(json.dumps(state, indent=2), flush=True)
