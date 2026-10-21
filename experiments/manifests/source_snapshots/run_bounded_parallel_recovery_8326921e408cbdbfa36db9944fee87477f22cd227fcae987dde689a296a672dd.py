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
ap.add_argument('--max-workflow-seconds',type=float)
ap.add_argument('--resume',action='store_true')
args = ap.parse_args()
assert min(args.max_phase_seconds, args.max_rss_gib, args.min_free_disk_gib)>0
protocol = json.loads((P/args.protocol).read_text())
stem = Path(protocol['output']).stem
lock = (P/f'experiments/logs/{stem}_runner.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (P/protocol['output']).exists(), 'Preserve existing attempt; never auto-retry'
path = P/f'experiments/metrics/{stem}_workflow.json'
previous_elapsed=0.;previous_phases=[]
if path.exists():
    assert args.resume, 'Workflow already exists; preserve it or explicitly resume its remaining budget'
    previous=json.loads(path.read_text());assert not previous['complete']
    assert previous['limits'].get('max_workflow_seconds')==args.max_workflow_seconds
    previous_elapsed=previous.get('total_wall_seconds',sum(x['wall_seconds'] for x in previous['phases']))
    assert not args.max_workflow_seconds or previous_elapsed<args.max_workflow_seconds, 'Original cumulative workflow budget exhausted'
    import hashlib
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (P/f'experiments/logs/{stem}_workflow_before_resume_{digest[:12]}.json').write_text(json.dumps(previous,indent=2))
    previous_phases=previous['phases']
state = dict(complete=False, full_cycle_validated=False, protocol=args.protocol,
    limits=vars(args), phases=previous_phases)
workflow_started=time.perf_counter()
def persist():
    state['total_wall_seconds']=previous_elapsed+time.perf_counter()-workflow_started
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(state, indent=2)); temporary.replace(path)
env = os.environ.copy()
env.update(FI_PROVIDER='tcp', FI_TCP_IFACE='en0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
    MAMBA_ROOT_PREFIX=str(P/'.mamba'), XDG_CACHE_HOME=str(P/'.cache'),
    DIJITSO_CACHE_DIR=str(P/'.fenics-cache-activated'))
if protocol.get('parallel_ranks'):
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
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
                if args.max_workflow_seconds and previous_elapsed+time.perf_counter()-workflow_started>args.max_workflow_seconds:stop='workflow_wall_limit'
                if peak > args.max_rss_gib*2**30: stop = 'descendant_RSS_limit'
                if shutil.disk_usage(P).free < args.min_free_disk_gib*2**30: stop = 'free_disk_reserve'
                if stop: break
                persist()
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
    if protocol.get('stream_protocol'):
        stream=json.loads((P/protocol['stream_protocol']).read_text())
        run('seed',[str(P/'.venv/bin/python'),'scripts/seed_parallel_reference.py','--protocol',protocol['stream_protocol']])
        run('prefix_positivity',[str(P/'.venv/bin/python'),'scripts/audit_parallel_prefix.py','--protocol',protocol['stream_protocol']])
        run('solve',native[:-1]+['mpiexec','-n',str(stream['parallel_ranks']),'python',
            'scripts/generate_parallel_reference.py','--protocol',protocol['stream_protocol']])
        if not protocol.get('full_reference_stream'):
            run('export_probe',[str(P/'.venv/bin/python'),'scripts/export_parallel_stream_probe.py','--protocol',args.protocol])
    elif protocol.get('parallel_ranks'):
        run('solve', native[:-1]+['mpiexec','-n',str(protocol['parallel_ranks']),
            'python','scripts/probe_parallel_history_recovery.py','--protocol',args.protocol])
    else:
        run('solve', native+['scripts/probe_quadratic_strict_stopping.py', '--protocol', args.protocol])
    if protocol.get('full_reference_stream'):
        cfg=json.loads((P/stream['config']).read_text())
        run('whole_element_summary',[str(P/'.venv/bin/python'),'scripts/finalize_parallel_reference.py','--protocol',protocol['stream_protocol']])
        run('independent_native_audit',native+['scripts/verify_completed_quadratic_reference.py','--config',stream['config'],'--reassemble-force-subset'])
        v=json.loads((P/f'experiments/metrics/{cfg["name"]}_verification.json').read_text());assert v['passed']
        if protocol.get('mesh_baseline_config'):
            run('mesh_comparison',native+['scripts/compare_quadratic_references.py','--config-a',protocol['mesh_baseline_config'],
                '--config-b',stream['config'],'--output',protocol['mesh_comparison_output']])
        result=dict(complete=True,full_reference_native_verified=True,study_validation_passed=False,
            scope='Completed native reference/audits; numerical-convergence,geometry/loading,trained-model/calibration/frozen-test gates remain separate.')
        (P/protocol['output']).write_text(json.dumps(result,indent=2))
        state.update(complete=True,native_reference_verified=True,active_phase='finished')
    else:
        result=json.loads((P/protocol['output']).read_text())
        assert result['finished'] and result['all_requested_steps_accepted'],'Homotopy did not reach all exact physical endpoints'
        run('audit',native+['scripts/verify_quadratic_substep_probe.py','--protocol',args.protocol])
        audit_path=P/f'experiments/metrics/{stem}_verification.json'
        audit=json.loads(audit_path.read_text())
        if not audit['passed']:
            assert 'qualification_step' not in protocol,'Known coarse whole-element qualification failed'
            config=protocol['reference_config']
            run('resolve_positivity',[str(P/'.venv/bin/python'),'scripts/resolve_quadratic_positivity.py','--config',config,'--probe-protocol',args.protocol])
            run('compose_audit',[str(P/'.venv/bin/python'),'scripts/combine_quadratic_probe_audits.py','--protocol',args.protocol])
            audit=json.loads((P/f'experiments/metrics/{stem}_verification_resolved.json').read_text())
        assert audit['passed'] and all(r['native_residual_N']<1e-8 for r in audit['records'])
        state.update(complete=True,local_recovery_verified=True,active_phase='finished')
except Exception as error:
    state.update(complete=True, local_recovery_verified=False, failure=str(error), active_phase='stopped')
finally:
    persist()
    subprocess.run([str(P/'.venv/bin/python'), 'scripts/write_reports.py'], cwd=P, env=env, check=True)
print(json.dumps(state, indent=2), flush=True)
