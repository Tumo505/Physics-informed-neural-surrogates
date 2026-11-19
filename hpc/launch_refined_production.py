"""Continue the completed quarter-step trajectory in a separate checkpointed PBS chain."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

import compare_reference_timesteps_555 as refined
from registry import artifact, atomic, event, read, sha
from submit_probe_profile import assert_idle, registered

ROOT, TARGET, NATIVE, GPU, CUDA = refined.ROOT, refined.TARGET, refined.NATIVE, refined.GPU, refined.CUDA
load = refined.load
WALL_SECONDS = 168 * 3600


def compact_record(row):
    return {key: value for key, value in row.items() if key not in ('patch', 'whole_element_proof')}


def production_schedule(config):
    import numpy as np
    factor = 4
    stop = int(round(config['duration_s']/config['dt_s']))
    assert stop == 1600
    dt = config['dt_s']/factor
    times = 550*config['dt_s'] + np.arange(1, (stop-550)*factor+1)*dt
    assert abs(times[-1]-config['duration_s']) < 1e-12
    return dt, times, times-.4*dt


def validate_row(row, index, digest, schedule):
    assert row['substep'] == index and row['input_checkpoint_sha256'] == digest
    assert row['time_s'] == schedule['endpoint_times'][index-1]
    assert row['pressure_time_s'] == schedule['pressure_times'][index-1]
    assert row['pressure_Pa'] == schedule['pressures'][index-1]
    assert 0 <= row['custom_force_N'] < 1e-9 and 0 <= row['native_force_N'] < 1e-8
    assert row['native_updates'] == 0


def production_records(run, *, verify_all_checkpoints=False, check_deadline=None):
    """Use registered acceptance commits, never an unaccepted Newton iterate."""
    plan = load(run / 'plan.json')
    schedule = load(run / 'pressure.json')['factors']['4']
    ledger = {item['path']: item['sha256'] for item in read(run) if item['kind'] == 'artifact'}
    commits = sorted((run / 'factor-4').glob('substep-*/accepted.json'),
                     key=lambda path: int(path.parent.name.split('-')[1]))
    records = []
    digest = sha(run / 'seed-550.npz')
    for index, path in enumerate(commits, 1):
        if check_deadline is not None:
            check_deadline()
        assert path.parent.name == f'substep-{index:03d}', 'Gap in accepted checkpoint chain'
        assert ledger[str(path.relative_to(ROOT))] == sha(path), 'Changed acceptance commit'
        row = load(path)
        validate_row(row, index, digest, schedule)
        trace_name = row.get('trace_file', 'trace.json')
        assert Path(trace_name).name == trace_name
        assert sha(path.parent / trace_name) == row['trace_sha256']
        if verify_all_checkpoints or index == len(commits):
            assert sha(path.parent / 'checkpoint.npz') == row['checkpoint_sha256']
        records.append(compact_record(row))
        digest = row['checkpoint_sha256']
    assert plan['resume']['completed_substeps'] <= len(records) <= len(schedule['endpoint_times'])
    return records


def initialize(run, parent):
    for name in ('plan.json', 'result.json', 'seed-550.npz', 'patch.npz', 'pressure.json'):
        registered(ROOT, parent, parent / name)
    assert load(parent / 'state.json')['status'] == 'comparison_complete'
    rows = refined.accepted_prefix(parent, 4, completed=True)
    assert len(rows) == 20 and rows[-1]['time_s'] == 555*load(TARGET / 'manifest.json')['config']['dt_s']
    inherited = load(parent / 'plan.json')
    assert inherited['gpu_library_sha256'] == sha(GPU / 'lib/libpetsc.so')
    assert inherited['source_checkpoint_sha256'] == sha(TARGET / 'restart.npz')
    assert load(TARGET / 'manifest.json')['completed_steps'] == 554
    frozen = dict(inherited['frozen'])
    # This launcher extends the qualified refined runner; pin the new implementation explicitly.
    for path in (Path(__file__).resolve(), Path(refined.__file__).resolve(),
                 Path(__file__).resolve().with_name('refined_positivity_policy.py'),
                 Path(__file__).resolve().with_name('refined_constrained_direction.py'),
                 Path(__file__).resolve().with_name('cuda_determinant_direction.py'),
                 Path(__file__).resolve().parents[1] / 'src/lv_surrogate/physics/adaptive_bernstein.py'):
        frozen[str(path)] = sha(path)
    for path, digest in frozen.items():
        assert sha(Path(path)) == digest, path
    restored = refined.restored_checkpoint(parent / 'factor-4/substep-020/checkpoint.npz', rows[-1], 4)
    assert len(restored['u']) > 0
    run.mkdir()
    for source_name, destination in (('seed-550.npz', 'seed-550.npz'), ('patch.npz', 'patch.npz'),
                                     ('pressure.json', 'prefix-pressure.json')):
        path = run / destination
        shutil.copyfile(parent / source_name, path)
        assert sha(path) == sha(parent / source_name)
        artifact(ROOT, run, path, 'Refined production seed data')
        frozen[str(path)] = sha(path)
    for row in rows:
        relative = Path(f'factor-4/substep-{row["substep"]:03d}')
        step = run / relative
        step.mkdir(parents=True)
        for name, digest in (('accepted.json', sha(parent / relative / 'accepted.json')),
                             (row.get('trace_file', 'trace.json'), row['trace_sha256']),
                             ('checkpoint.npz', row['checkpoint_sha256'])):
            path = step / name
            shutil.copyfile(parent / relative / name, path)
            assert sha(path) == digest
            artifact(ROOT, run, path, 'Completed refined prefix')
            frozen[str(path)] = digest
    frozen[str(parent / 'result.json')] = sha(parent / 'result.json')
    config = load(TARGET / 'manifest.json')['config']
    dt, times, _ = production_schedule(config)
    plan = dict(inherited, refined_production=True, factors=[4], stop_state=1600,
                parent_refinement=parent.name, dt_s=dt, requested_substeps=len(times),
                walltime_seconds=WALL_SECONDS, frozen=frozen,
                resume=dict(parent_run=parent.name, factor=4, completed_substeps=20),
                scope='Separate quarter-timestep continuation through the original final time; '
                      'inherits the state-550 histories and completed refinement. '
                      'No claim of full-cycle time convergence or admission to the original reference.')
    atomic(run / 'plan.json', plan)
    artifact(ROOT, run, run / 'plan.json', 'Full refined production plan')
    command = [str(ROOT / '.venv/bin/python'), str(Path(__file__).resolve()), 'execute', '--run-id', run.name]
    script = run / 'job.pbs'
    script.write_text('#!/bin/bash\nset -eu\ncd '+shlex.quote(str(ROOT))+'\nexec '+shlex.join(command)+'\n')
    artifact(ROOT, run, script, 'Checkpointed refined production worker')
    atomic(run / 'state.json', dict(status='prepared', slice=0, completed_substeps=20,
                                   requested_substeps=len(times), original_reference_steps=554))


def check_depth_evidence(report, spec):
    assert report['reproduced'] and not report['mismatches']
    assert report['reference_state_accepted'] is False
    for key, value in (('substep', 21), ('element', 813578), ('stage', 'endpoint'),
                       ('margin', 1e-9), ('original_depth', 20), ('retry_depth', 24)):
        assert report[key] == value
        if key != 'margin':
            assert spec[key] == value
    assert report['iteration'] == spec['iteration'] and report['alpha'] == spec['alpha']
    cell = report['cell_audit']
    assert cell['independent_gradient_agrees']
    estimate = cell['independent_cartesian']['optimized_minimum_estimate']
    assert math.isfinite(estimate) and estimate > report['margin']
    entries = cell['captured_gradient']['bernstein']
    assert [(row['depth'], row['outcome']) for row in entries] == [
        (24, 'unresolved'), (28, 'unresolved'), (32, 'certified_positive')]
    proof = entries[-1]
    assert math.isfinite(proof['lower_bound']) and proof['lower_bound'] > report['margin']
    assert 24 < proof['maximum_depth'] <= 32 and proof['leaf_tetrahedra'] > 0
    return dict(depth=32, margin=report['margin'], element=report['element'],
                substep=report['substep'], lower_bound=proof['lower_bound'])


def initialize_certified_retry(run, replay):
    for name in ('plan.json', 'positivity-replay.json', 'rejected-cell.npz'):
        registered(ROOT, replay, replay / name)
    assert load(replay / 'state.json')['status'] == 'diagnostic_complete'
    evidence = load(replay / 'plan.json')
    policy = check_depth_evidence(load(replay / 'positivity-replay.json'), evidence['positivity_replay'])
    failed = ROOT / 'experiments/hpc' / evidence['diagnostic_parent']
    registered(ROOT, failed, failed / 'plan.json')
    registered(ROOT, failed, failed / 'pressure.json')
    assert load(failed / 'state.json')['status'] == 'blocked'
    records = production_records(failed)
    assert len(records) == 20
    assert sha(replay / 'factor-4/substep-020/checkpoint.npz') == records[-1]['checkpoint_sha256']
    # The diagnostic plan pins the failed production plan and the exact measured trial.
    for path, digest in evidence['frozen'].items():
        path = Path(path)
        if path == failed / 'plan.json' or path.is_relative_to(failed / 'factor-4'):
            assert sha(path) == digest, path
    source_plan = load(failed / 'plan.json')
    parent = ROOT / 'experiments/hpc' / source_plan['parent_refinement']
    qualified = refined.accepted_prefix(parent, 4, completed=True)
    assert [row['checkpoint_sha256'] for row in qualified] == [row['checkpoint_sha256'] for row in records]
    initialize(run, parent)
    shutil.copyfile(failed / 'pressure.json', run / 'pressure.json')
    assert sha(run / 'pressure.json') == sha(failed / 'pressure.json')
    artifact(ROOT, run, run / 'pressure.json', 'Preserved pressure schedule for certified production retry')
    plan = load(run / 'plan.json')
    policy.update(from_run=replay.name, result_sha256=sha(replay / 'positivity-replay.json'),
                  rejected_cell_sha256=sha(replay / 'rejected-cell.npz'),
                  start_checkpoint_sha256=records[-1]['checkpoint_sha256'])
    plan['certified_depth_retry'] = policy
    for path in (replay / 'plan.json', replay / 'positivity-replay.json', replay / 'rejected-cell.npz',
                 failed / 'plan.json', run / 'pressure.json'):
        plan['frozen'][str(path)] = sha(path)
    atomic(run / 'plan.json', plan)
    artifact(ROOT, run, run / 'plan.json', 'Evidence-gated depth-32 refined production retry')


def check_saved_cell_frames(diagnostic):
    import numpy as np
    from lv_surrogate.physics.adaptive_bernstein import resolve_positive

    report = load(diagnostic / 'cell-path.json')
    assert report['complete'] and report['reference_state_accepted'] is False
    assert report['element'] == 813578 and report['substep'] == 21 and report['margin'] == 1e-9
    assert report['matched_prefix_directions'] >= 8
    index = load(diagnostic / 'cell-frames/index.json')
    frames = {row['path']: row for row in index['frames']}
    checks = []
    improved = 0
    for frame in report['frames']:
        assert frames[frame['path']]['sha256'] == frame['sha256']
        path = diagnostic / frame['path']
        assert path.resolve().is_relative_to((diagnostic / 'cell-frames').resolve())
        assert sha(path) == frame['sha256']
        assert frame['cell_audit']['independent_gradient_agrees']
        if frame['stage'] != 'endpoint':
            continue
        row = dict(kind=frame['kind'], iteration=frame['iteration'], path=frame['path'])
        with np.load(path, allow_pickle=False) as data:
            try:
                proof = resolve_positive(data['deformation'], data['vertices'])
                row.update(outcome='certified_positive', proof=proof)
            except ValueError as exc:
                row.update(outcome='rejected', reason=str(exc))
        if frame['kind'] in ('checkpoint', 'initializer', 'accepted_newton'):
            assert row['outcome'] == 'certified_positive', ('Saved positive field failed interval proof', row)
        elif (frame['kind'] == 'first_unresolved_trial' and frame['iteration'] < report['matched_prefix_directions']
              and row['outcome'] == 'certified_positive'):
            improved += 1
        checks.append(row)
        print('SAVED CELL PROOF', row['kind'], row['iteration'], row['outcome'], flush=True)
    assert improved > 0, 'No previously blocked trial in the matched path was newly certified; no job submitted'
    assert any(row['kind'] == 'checkpoint' for row in checks)
    return dict(passed=True, margin=1e-9, algorithm='determinant_adaptive_interval_bernstein',
                improved_matched_trials=improved, checks=checks)


def initialize_adaptive_retry(run, diagnostic):
    for name in ('plan.json', 'cell-path.json', 'cell-frames/index.json'):
        registered(ROOT, diagnostic, diagnostic / name)
    assert load(diagnostic / 'state.json')['status'] == 'diagnostic_complete'
    evidence = load(diagnostic / 'plan.json')
    assert evidence['certified_depth_retry']['depth'] == 32 and evidence['cell_path_audit']['substep'] == 21
    proof_check = check_saved_cell_frames(diagnostic)
    failed = ROOT / 'experiments/hpc' / evidence['diagnostic_parent']
    registered(ROOT, failed, failed / 'plan.json')
    registered(ROOT, failed, failed / 'pressure.json')
    assert load(failed / 'state.json')['status'] in ('blocked', 'running')
    records = production_records(failed)
    assert len(records) == 20
    assert sha(diagnostic / 'factor-4/substep-020/checkpoint.npz') == records[-1]['checkpoint_sha256']
    source = load(failed / 'plan.json')
    parent = ROOT / 'experiments/hpc' / source['parent_refinement']
    qualified = refined.accepted_prefix(parent, 4, completed=True)
    assert [row['checkpoint_sha256'] for row in qualified] == [row['checkpoint_sha256'] for row in records]
    initialize(run, parent)
    shutil.copyfile(failed / 'pressure.json', run / 'pressure.json')
    artifact(ROOT, run, run / 'pressure.json', 'Unchanged pressure schedule for adaptive-proof retry')
    atomic(run / 'saved-cell-proof-check.json', proof_check)
    artifact(ROOT, run, run / 'saved-cell-proof-check.json', 'Adaptive interval proof on preserved cell frames')
    plan = load(run / 'plan.json')
    plan.update(certified_depth_retry=source['certified_depth_retry'],
                adaptive_interval_proof=dict(from_run=diagnostic.name, margin=1e-9,
                    max_nodes=20000, max_depth=64, start_checkpoint_sha256=records[-1]['checkpoint_sha256']))
    for path in (diagnostic / 'plan.json', diagnostic / 'cell-path.json', diagnostic / 'cell-frames/index.json',
                 failed / 'plan.json', run / 'pressure.json', run / 'saved-cell-proof-check.json'):
        plan['frozen'][str(path)] = sha(path)
    atomic(run / 'plan.json', plan)
    artifact(ROOT, run, run / 'plan.json', 'Adaptive interval positivity production retry')


def initialize_direction_retry(run, failed):
    for name in ('plan.json', 'pressure.json'):
        registered(ROOT, failed, failed / name)
    state, source = load(failed / 'state.json'), load(failed / 'plan.json')
    assert state['status'] == 'blocked' and source['adaptive_interval_proof']['margin'] == 1e-9
    assert not source.get('positivity_replay') and not source.get('cell_path_audit')
    assert source['source_checkpoint_sha256'] == sha(TARGET / 'restart.npz')
    assert any(item['kind'] == 'job_submission' and item['return_code'] == 0 and item['stdout'] == state['job']
               for item in read(failed))
    records = production_records(failed)
    assert len(records) == 20
    trace_path = failed / f'factor-4/substep-021/trace-slice-{state["slice"]:03d}.json'
    trace = load(trace_path)
    assert trace['substep'] == 21 and trace['positivity_fallback_algorithm'] == 'determinant_adaptive_interval_bernstein'
    reason = 'Direct determinant margin violation witness (stage=endpoint, element=813578)'
    assert any(trial.get('positivity_failure') == reason
               for row in trace['custom_newton_steps'] for trial in row['trials'])
    assert trace['custom_newton_steps'][-1]['initial_force_N'] > 1e-9
    parent = ROOT / 'experiments/hpc' / source['parent_refinement']
    qualified = refined.accepted_prefix(parent, 4, completed=True)
    assert [row['checkpoint_sha256'] for row in qualified] == [row['checkpoint_sha256'] for row in records]
    initialize(run, parent)
    shutil.copyfile(failed / 'pressure.json', run / 'pressure.json')
    assert sha(run / 'pressure.json') == sha(failed / 'pressure.json')
    artifact(ROOT, run, run / 'pressure.json', 'Preserved forcing for joint determinant/merit direction retry')
    plan = load(run / 'plan.json')
    plan.update(certified_depth_retry=source['certified_depth_retry'],
                adaptive_interval_proof=source['adaptive_interval_proof'],
                constrained_direction=dict(kind='joint_determinant_merit_qp', element=813578,
                    margin=1e-9, parent_run=failed.name, start_checkpoint_sha256=records[-1]['checkpoint_sha256']))
    for path in (failed / 'plan.json', trace_path, run / 'pressure.json'):
        plan['frozen'][str(path)] = sha(path)
    atomic(run / 'plan.json', plan)
    artifact(ROOT, run, run / 'plan.json', 'Joint determinant/force-merit production direction plan')


def submit_slice(run, number, after=None):
    command = ['qsub', '-N', 'lv_refined', '-q', 'iworkq',
               '-l', 'select=1:ncpus=28:mem=240gb:ngpus=4', '-l', 'place=excl',
               '-l', 'software=PBSJobApp', '-l', 'walltime=168:00:00', '-j', 'oe',
               '-o', str(run / f'pbs-{number:03d}.log')]
    if after:
        command += ['-W', 'depend=afterok:'+after]
    result = subprocess.run([*command, str(run / 'job.pbs')], cwd=ROOT, text=True, capture_output=True)
    event(run, 'job_submission', slice=number, dependency=after, argv=command,
          return_code=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip())
    result.check_returncode()
    job = result.stdout.strip()
    assert re.fullmatch(r'\d+(?:\.[\w.-]+)?', job)
    print('Refined production slice', number, 'job:', job, flush=True)
    return job


def native_environment():
    env = os.environ.copy()
    for key in ('LD_PRELOAD', 'PETSC_DIR', 'PETSC_ARCH', 'PETSC_OPTIONS', 'PKG_CONFIG_SYSROOT_DIR'):
        env.pop(key, None)
    pkg = f'{NATIVE}/lib/pkgconfig:{NATIVE}/share/pkgconfig'
    env.update(PATH=f'{NATIVE}/bin:{env["PATH"]}', LD_LIBRARY_PATH=f'{NATIVE}/lib:{CUDA}/lib64',
               PKG_CONFIG=str(NATIVE / 'bin/pkg-config'), PKG_CONFIG_PATH=pkg, PKG_CONFIG_LIBDIR=pkg,
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HYDRA_LAUNCHER='fork', FI_PROVIDER='tcp', DIJITSO_CACHE_DIR=str(ROOT / '.hpc/fenics-cache'),
               XDG_CACHE_HOME=str(ROOT / '.hpc/cache'))
    return env, pkg


def prepare_schedule(run):
    import numpy as np
    from lv_surrogate.physics.reference_fem import pressure_function, default_parameters_benchmark1
    config = load(TARGET / 'manifest.json')['config']
    old = load(run / 'prefix-pressure.json')
    dt, times, stress_times = production_schedule(config)
    parameters = default_parameters_benchmark1()
    for key in ('t_sys_pre', 't_dias_pre', 'gamma'):
        parameters[key] *= config['timing_scale']
    with np.load(TARGET / 'inputs.npz', allow_pickle=False) as inputs:
        original_times, original_pressures = inputs['time'], inputs['pressure']
    combined = np.unique(np.concatenate((stress_times, original_times-.4*config['dt_s'])))
    values = config['amplitude'] * pressure_function((0, config['duration_s']), parameters=parameters, t_eval=combined)
    pressures = values[np.searchsorted(combined, stress_times)]
    assert np.isfinite(pressures).all()
    prefix = old['factors']['4']
    np.testing.assert_array_equal(times[:20], prefix['endpoint_times'])
    np.testing.assert_array_equal(stress_times[:20], prefix['pressure_times'])
    np.testing.assert_allclose(pressures[:20], prefix['pressures'], rtol=1e-10, atol=1e-8)
    # Keep the already accepted prefix bit-for-bit; evaluate all future stress times directly.
    pressures[:20] = prefix['pressures']
    original = values[np.searchsorted(combined, original_times-.4*config['dt_s'])]
    reproduction = refined.pressure_reproduction(original, original_pressures)
    output = dict(old, pressure_reproduction=reproduction,
                  original_pressure_reproduction_max_abs=reproduction['full_cycle_max_abs_Pa'],
                  factors={'4': dict(dt_s=dt, endpoint_times=times.tolist(),
                                     pressure_times=stress_times.tolist(), pressures=pressures.tolist())})
    atomic(run / 'pressure.json', output)
    print('REFINED PRODUCTION PRESSURES READY remaining_substeps', len(times)-20, flush=True)


def execute(run):
    assert os.environ.get('PBS_JOBID')
    started = time.time()
    plan = load(run / 'plan.json')

    def check_deadline():
        if time.time() >= started+WALL_SECONDS-120:
            raise refined.SliceDeadline('PBS slice checkpoint pause during final verification')
    with (run / 'execution.lock').open('a') as execution, (TARGET / 'generation.lock').open('a') as generation:
        for lock in (execution, generation):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = load(run / 'state.json')
        state.update(status='running', job=os.environ['PBS_JOBID'], slice=state['slice']+1)
        atomic(run / 'state.json', state)
        try:
            registered(ROOT, run, run / 'plan.json')
            registered(ROOT, run, run / 'job.pbs')
            for path, digest in plan['frozen'].items():
                assert sha(Path(path)) == digest, path
            assert sha(GPU / 'lib/libpetsc.so') == plan['gpu_library_sha256']
            assert len(set(Path(os.environ['PBS_NODEFILE']).read_text().splitlines())) == 1
            env, pkg = native_environment()
            base = [str(NATIVE / 'bin/python'), str(Path(__file__).resolve())]
            if not (run / 'pressure.json').exists():
                with (run / 'prepare.log').open('x') as log:
                    subprocess.run([*base, 'prepare', '--run-id', run.name], cwd=ROOT,
                                   env=dict(env, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT, check=True)
                artifact(ROOT, run, run / 'pressure.json', 'Full quarter-step pressure schedule')
            registered(ROOT, run, run / 'pressure.json')
            devices = env.get('CUDA_VISIBLE_DEVICES') or subprocess.check_output(
                ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().replace('\n', ',')
            assert len(set(devices.split(','))) == 4
            gpu_env = dict(env, CUDA_VISIBLE_DEVICES=devices, PETSC_OPTIONS='-use_gpu_aware_mpi 0',
                           PKG_CONFIG_PATH=f'{GPU}/lib/pkgconfig:{pkg}', PKG_CONFIG_LIBDIR=f'{GPU}/lib/pkgconfig:{pkg}',
                           REFINED_SLICE=str(state['slice']),
                           REFINED_DEADLINE=str(started+WALL_SECONDS-120))
            log_path = run / f'factor-4/command-{state["slice"]:03d}.log'
            command = [str(NATIVE / 'bin/mpiexec'), '-n', '12', '/usr/bin/env',
                       f'LD_PRELOAD={GPU}/lib/libpetsc.so', *base, 'worker', '--run-id', run.name]
            with log_path.open('x') as log:
                result = subprocess.run(command, cwd=ROOT, env=gpu_env, stdout=log, stderr=subprocess.STDOUT)
            assert result.returncode == 0, f'Native worker exited {result.returncode}; inspect {log_path}'
            if plan.get('cell_path_audit'):
                report = run / 'cell-path.json'
                assert load(report)['complete'] and load(report)['reference_state_accepted'] is False
                artifact(ROOT, run, report, 'Independent limiting-cell path audit')
                artifact(ROOT, run, run / 'cell-frames/index.json', 'Captured cell displacement frames')
                assert sha(TARGET / 'restart.npz') == plan['source_checkpoint_sha256']
                assert sha(TARGET / 'manifest.json') == plan['frozen'][str(TARGET / 'manifest.json')]
                state.update(status='diagnostic_complete', result=str(report.relative_to(ROOT)))
                return
            if plan.get('positivity_replay'):
                report = run / 'positivity-replay.json'
                artifact(ROOT, run, report, 'Rejected refined trial positivity replay')
                if (run / 'rejected-cell.npz').exists():
                    artifact(ROOT, run, run / 'rejected-cell.npz', 'Captured rejected cell and P2 displacement')
                assert load(report)['reproduced'], 'Replay did not match the preserved rejected trial'
                assert sha(TARGET / 'restart.npz') == plan['source_checkpoint_sha256']
                assert sha(TARGET / 'manifest.json') == plan['frozen'][str(TARGET / 'manifest.json')]
                state.update(status='diagnostic_complete', result=str(report.relative_to(ROOT)))
                return
            summary = load(run / 'factor-4/summary.json')
            state.update(completed_substeps=summary['completed_substeps'])
            assert sha(TARGET / 'restart.npz') == plan['source_checkpoint_sha256']
            assert sha(TARGET / 'manifest.json') == plan['frozen'][str(TARGET / 'manifest.json')]
            if summary['complete']:
                records = production_records(run, verify_all_checkpoints=True, check_deadline=check_deadline)
                assert len(records) == plan['requested_substeps']
                result = dict(complete=True, final_time_s=records[-1]['time_s'], dt_s=plan['dt_s'],
                              completed_substeps=len(records), final_checkpoint_sha256=records[-1]['checkpoint_sha256'],
                              native_force_N=records[-1]['native_force_N'], original_reference_steps=554,
                              scope=plan['scope'])
                atomic(run / 'result.json', result)
                artifact(ROOT, run, run / 'result.json', 'Completed separate refined production trajectory')
                state['status'] = 'production_complete'
            else:
                assert summary.get('paused_for_walltime'), 'Worker stopped without completion or a planned pause'
                state['next_job'] = submit_slice(run, state['slice']+1, after=os.environ['PBS_JOBID'])
                state['status'] = 'awaiting_next_slice'
        except refined.SliceDeadline:
            state['next_job'] = submit_slice(run, state['slice']+1, after=os.environ['PBS_JOBID'])
            state['status'] = 'awaiting_next_slice'
        except Exception as exc:
            path = run / 'factor-4/summary.json'
            if path.exists():
                state['completed_substeps'] = load(path)['completed_substeps']
            state.update(status='blocked', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            atomic(run / 'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'execute', 'prepare', 'worker', 'resume'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--from-refinement', default='cmbbe-timestep-550-555-03')
    parser.add_argument('--certify-depth-from', help='Completed refined positivity replay authorizing depth-32 fallback')
    parser.add_argument('--adaptive-proof-from', help='Completed cell-path audit used for the saved-frame proof check')
    parser.add_argument('--constrained-direction-after', help='Blocked adaptive-proof production with a determinant margin witness')
    args = parser.parse_args()
    for value in (args.run_id, args.from_refinement, args.certify_depth_from, args.adaptive_proof_from,
                  args.constrained_direction_after):
        if value is None:
            continue
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            parser.error('Invalid run ID')
    run = ROOT / 'experiments/hpc' / args.run_id
    evidence_sources = (args.certify_depth_from, args.adaptive_proof_from, args.constrained_direction_after)
    if any(evidence_sources) and args.action != 'submit':
        parser.error('Proof evidence options require submit')
    if sum(bool(value) for value in evidence_sources) > 1:
        parser.error('Select one proof evidence source')
    if args.action in ('submit', 'resume'):
        with (ROOT / '.hpc/profile-submission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert_idle(ROOT)
            if args.action == 'submit':
                if args.constrained_direction_after:
                    initialize_direction_retry(run, ROOT / 'experiments/hpc' / args.constrained_direction_after)
                elif args.adaptive_proof_from:
                    initialize_adaptive_retry(run, ROOT / 'experiments/hpc' / args.adaptive_proof_from)
                elif args.certify_depth_from:
                    initialize_certified_retry(run, ROOT / 'experiments/hpc' / args.certify_depth_from)
                else:
                    initialize(run, ROOT / 'experiments/hpc' / args.from_refinement)
            else:
                registered(ROOT, run, run / 'plan.json')
                assert load(run / 'state.json')['status'] in ('blocked', 'running'), 'No interrupted production run'
            state = load(run / 'state.json')
            state.update(status='submitted', next_job=submit_slice(run, state['slice']+1))
            atomic(run / 'state.json', state)
    elif args.action == 'execute':
        execute(run)
    elif args.action == 'prepare':
        prepare_schedule(run)
    else:
        assert os.environ.get('PBS_JOBID')
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        assert len(set(devices)) == 4
        rank = int(os.environ['PMI_RANK'])
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[rank % 4]
        print('REFINED PRODUCTION GPU', rank, os.environ['CUDA_VISIBLE_DEVICES'], flush=True)
        try:
            refined.solve_factor(run, 4)
        except Exception:
            import traceback
            traceback.print_exc()
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
