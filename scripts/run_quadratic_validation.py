"""Execute one controlled FE refinement, verify norms and refresh existing report."""
import argparse
import fcntl
import hashlib
import json
import os
import signal
import shutil
from pathlib import Path
import subprocess
import time

P = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('factor', choices=['time', 'mesh', 'solver', 'mesh_gamg', 'quadrature', 'reference'])
ap.add_argument('--baseline-config')
ap.add_argument('--candidate-config')
ap.add_argument('--output-stem')
ap.add_argument('--wait-for-workflow', help='Wait for this existing workflow lock, then require its successful audited completion')
ap.add_argument('--require-krylov-probe', help='Require a completed three-state same-root backend probe')
ap.add_argument('--reuse-positivity-audit',action='store_true',help='Reuse a complete positivity audit only when its config, frame count and output hashes match')
ap.add_argument('--max-wall-hours',type=float)
ap.add_argument('--max-rss-gib',type=float)
ap.add_argument('--min-free-disk-gib',type=float)
args = ap.parse_args()
baseline = 'configs/physics/dynamic_quadratic_reference_'+('gamg' if args.factor in ['mesh_gamg','quadrature'] else 'dt00125')+'.json'
candidate = 'configs/physics/dynamic_quadratic_reference_'+dict(time='dt001',mesh='refined',solver='gamg',mesh_gamg='refined_gamg',quadrature='quadrature8',reference='halfstep_gamg')[args.factor]+'.json'
baseline = args.baseline_config or baseline
candidate = args.candidate_config or candidate
if args.factor == 'reference':
    baseline = candidate
name = json.loads((P/candidate).read_text())['name']
stem = args.output_stem or 'dynamic_quadratic_controlled_'+args.factor
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
workflow_start=time.perf_counter()
state['resource_limits']=dict(wall_hours=args.max_wall_hours,total_descendant_RSS_GiB=args.max_rss_gib,
    minimum_free_disk_GiB=args.min_free_disk_gib,sampling_interval_s=2)
assert all(v is None or v>0 for v in [args.max_wall_hours,args.max_rss_gib,args.min_free_disk_gib])


def persist():
    output.write_text(json.dumps(state, indent=2, allow_nan=False))


def run(label, command):
    state['active_phase'] = label
    persist()
    start = time.perf_counter()
    peak_rss=0;stop_reason=None
    with (P/'experiments/logs'/f'{stem}_{label}.log').open('a') as log:
        if any(v is not None for v in [args.max_wall_hours,args.max_rss_gib,args.min_free_disk_gib]):
            child=subprocess.Popen(command,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                rows=subprocess.check_output(['ps','-axo','pid=,ppid=,rss='],text=True)
                processes=[tuple(map(int,line.split())) for line in rows.splitlines() if line.strip()]
                descendants={child.pid}
                while True:
                    found={pid for pid,parent,rss in processes if parent in descendants}
                    if found<=descendants:break
                    descendants.update(found)
                rss=sum(rss*1024 for pid,parent,rss in processes if pid in descendants)
                peak_rss=max(peak_rss,rss)
                if args.max_wall_hours and time.perf_counter()-workflow_start>args.max_wall_hours*3600:stop_reason='predeclared_workflow_wall_limit'
                if args.max_rss_gib and rss>args.max_rss_gib*2**30:stop_reason='predeclared_descendant_RSS_limit'
                if args.min_free_disk_gib and shutil.disk_usage(P).free<args.min_free_disk_gib*2**30:stop_reason='predeclared_free_disk_reserve'
                if stop_reason:
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        child.wait()
                    break
                time.sleep(2)
            returncode=child.wait()
        else:
            returncode=subprocess.run(command,cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    state['phases'].append(dict(phase=label, return_code=returncode,
                               measured_wall_seconds=time.perf_counter()-start,
                               sampled_peak_descendant_RSS_bytes=peak_rss if args.max_rss_gib else None,
                               stop_reason=stop_reason))
    persist()
    if returncode or stop_reason:
        raise RuntimeError(f'{label} stopped: {stop_reason or "process_failure"}, return code {returncode}; see preserved log')


persist()
try:
    if args.wait_for_workflow:
        state['active_phase'] = 'waiting_for_audited_baseline'; persist()
        prerequisite_lock = (P/'experiments/logs'/f'{args.wait_for_workflow}.lock').open('w')
        fcntl.flock(prerequisite_lock, fcntl.LOCK_EX)
        prerequisite_path = P/'experiments/metrics'/f'{args.wait_for_workflow}_pipeline.json'
        prerequisite = json.loads(prerequisite_path.read_text())
        assert prerequisite['complete'] and not prerequisite.get('failure'), 'Baseline workflow did not complete successfully'
        baseline_name = json.loads((P/baseline).read_text())['name']
        verification = json.loads((P/'experiments/metrics'/f'{baseline_name}_verification.json').read_text())
        baseline_manifest = json.loads((P/'data/processed'/baseline_name/'manifest.json').read_text())
        assert verification['passed'] and baseline_manifest['complete']
        assert verification['output_sha256'] == baseline_manifest['output_sha256']
        state['baseline_workflow_sha256'] = hashlib.sha256(prerequisite_path.read_bytes()).hexdigest()
        fcntl.flock(prerequisite_lock, fcntl.LOCK_UN); prerequisite_lock.close()
    if args.require_krylov_probe:
        path = P/args.require_krylov_probe
        probe = json.loads(path.read_text())
        assert probe['complete'] and probe['requested_steps'] == [9,11,13]
        assert len(probe['records']) == 3 and all(r['accepted'] for r in probe['records'])
        assert probe['local_changed_grid_probe']
        state['required_krylov_probe_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        persist()
    if (P/'data/processed'/name/'failure.json').exists():
        raise RuntimeError('A rejected trajectory is preserved; automatic retries are disabled')
    if args.factor in ['mesh', 'mesh_gamg']:
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
    elif args.factor != 'reference':
        run('norms', native+['scripts/compare_quadratic_references.py', '--config-a', baseline,
            '--config-b', candidate, '--output', f'experiments/metrics/{stem}.json'])
    audit_path=P/'experiments/metrics'/f'{name}_bernstein.json'
    current_manifest=json.loads((P/'data/processed'/name/'manifest.json').read_text())
    reused=False
    if args.reuse_positivity_audit and audit_path.exists():
        audit=json.loads(audit_path.read_text())
        reused=bool(audit['complete'] and audit['checked_steps']==current_manifest['n_steps']
            and audit['config_sha256']==current_manifest['config_sha256']
            and audit['output_sha256']==current_manifest['output_sha256'])
    if reused:
        state['reused_positivity_audit_sha256']=hashlib.sha256(audit_path.read_bytes()).hexdigest();persist()
    else:
        run('positivity', modern+['scripts/audit_quadratic_positivity.py', '--config', candidate])
    if not json.loads(audit_path.read_text())['whole_element_positivity_sufficient_condition_passed']:
        run('positivity_resolution',modern+['scripts/resolve_quadratic_positivity.py','--config',candidate])
    run('reference_verification', native+['scripts/verify_completed_quadratic_reference.py', '--config', candidate])
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
