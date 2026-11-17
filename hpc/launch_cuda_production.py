"""Submit and continue the full CUDA reference trajectory on one exclusive PBS node."""
import argparse
import fcntl
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from qualify_reference_restart import HPC, ROOT, check_checkpoint, load
from registry import artifact, atomic, event, read, sha
from submit_probe_profile import assert_idle, registered

PARENT = HPC / 'cmbbe-cofactor-ref-03'
ADMISSION = HPC / 'cmbbe-recovery-554-02'
WALL_SECONDS = 168 * 3600
GPU = Path.home() / '.local/petsc-3.22.3-cuda-sm52-m60'
CUDA = Path.home() / '.local/cuda-11.4.2'
GPU_LIB = GPU / 'lib/libpetsc.so'
NATIVE = ROOT / '.hpc/native-v2'
MODERN = ROOT / '.venv/bin/python'
SOURCES = ('hpc/launch_cuda_production.py', 'hpc/production_cuda_native.py',
           'src/lv_surrogate/physics/admissible_newton_gentle_shrink.py',
           'src/lv_surrogate/physics/regularization_recovery_policy.py',
           'hpc/worker8_native.py', 'hpc/platform_native_chunked.py',
           'hpc/cuda_determinant_direction.py', 'hpc/audit_reference_cell_555.py',
           'scripts/finalize_parallel_reference.py', 'scripts/compare_quadratic_references.py')


def submit(run, number, after=None):
    script = run / 'job.pbs'
    plan = load(run / 'plan.json')
    gpus = plan.get('gpus', 1)
    walltime = '01:00:00' if plan.get('positivity_diagnostic') else '168:00:00'
    cmd = ['qsub', '-N', 'lv_cuda_ref', '-q', 'iworkq',
           '-l', f'select=1:ncpus=28:mem=240gb:ngpus={gpus}', '-l', 'place=excl',
           '-l', 'software=PBSJobApp', '-l', f'walltime={walltime}',
           '-j', 'oe', '-o', str(run / f'pbs-{number:03d}.log')]
    if after:
        cmd += ['-W', 'depend=afterok:' + after]
    result = subprocess.run([*cmd, str(script)], cwd=ROOT, text=True, capture_output=True)
    job = result.stdout.strip()
    event(run, 'job_submission', slice=number, dependency=after, argv=cmd,
          return_code=result.returncode, stdout=job, stderr=result.stderr.strip())
    result.check_returncode()
    assert re.fullmatch(r'\d+(?:\.[\w.-]+)?', job), job
    print(f'Slice {number}: {job}', flush=True)


def initialize(run, retry=None, host_preconditioner=False, gpus=1, resume_cancelled=False, stagnation_shift=None, diagnose_positivity=False, certify_depth_from=None, adaptive_positivity=False, determinant_from=None, compare_direction=False, direction_comparison_from=None, pressure_after=None, residual_after=None, screen_blends=False):
    registered(ROOT, PARENT, PARENT / 'plan.json')
    inherited = load(PARENT / 'plan.json')
    state = load(PARENT / 'state.json')
    target = ROOT / 'data/processed' / inherited['target_name']
    manifest, digest = check_checkpoint(target, 554)
    assert manifest['steps'][553]['recovery_admission']['independent_audit_sha256'] == sha(
        ADMISSION / 'independent-audit.json')
    work = load(ADMISSION / 'independent-audit.json')
    registered(ROOT, ADMISSION, ADMISSION / 'independent-audit.json')
    assert work['passed'] and work['history_passed']
    markers = [target / name for name in ('failure.json', 'UNACCEPTED_iterate.npz')
               if (target / name).exists()]
    retry_log = None
    retry_iterate = None
    diagnostic = None
    diagnostic_seed = None
    positivity_depth = None
    direction_evidence = None
    comparison_report = None
    pressure_continuation = None
    residual_continuation = None
    if certify_depth_from is not None:
        replay_run = certify_depth_from
        replay_plan_path = replay_run / 'plan.json'
        registered(ROOT, replay_run, replay_plan_path)
        replay_plan = load(replay_plan_path)
        spec = replay_plan['positivity_diagnostic']
        replay_state = load(replay_run / 'state.json')
        assert replay_state['status'] == 'diagnostic_complete'
        assert replay_state['checkpoint_sha256'] == replay_plan['checkpoint_sha256'] == digest
        assert sha(GPU_LIB) == replay_plan['gpu_library_sha256']
        report = ROOT / replay_state['result']
        assert report == replay_run / 'slice-001/positivity-replay.json'
        registered(ROOT, replay_run, report)
        result = load(report)
        assert result['reproduced'] and result['state'] == 555
        assert result['stage'] == spec['stage'] == 'endpoint'
        assert result['element'] == spec['element'] == 813578
        iteration = spec.get('iteration', 0)
        original_depth = spec.get('original_depth', 12)
        certified_depth = {(0, 12): 16, (3, 16): 20}[iteration, original_depth]
        assert result.get('iteration', 0) == iteration
        if iteration == 3:
            assert result['prefix_matches'] and len(spec['prefix']) == 3
            assert result['initial_shift'] == spec['shift']
        assert result['alpha'] == 1.0 and result['shift'] == spec.get('trial_shift', spec['shift'])
        assert result['margin'] == 1e-9 and result['original_depth'] == original_depth
        proof, = result['results']
        assert proof['outcome'] == 'certified_positive' and proof['depth'] == certified_depth
        assert original_depth < proof['maximum_depth'] <= certified_depth
        assert math.isfinite(proof['lower_bound']) and proof['lower_bound'] > result['margin']
        retry_iterate = replay_run / 'diagnostic-seed.npz'
        registered(ROOT, replay_run, retry_iterate)
        assert sha(retry_iterate) == spec['seed_sha256']
        for marker in markers:
            if marker.name == 'failure.json':
                assert load(marker)['error'] in (
                    'Regularization ceiling without admissible descent',
                    'Regularized Newton iteration ceiling reached',
                )
        retry_log = report
        retry = replay_run
        stagnation_shift = spec['shift']
        positivity_depth = proof['depth']
        if adaptive_positivity:
            assert positivity_depth == 20, 'Adaptive refinement requires the depth-20 production replay'
        if determinant_from is not None:
            import numpy as np

            prior = determinant_from
            registered(ROOT, prior, prior / 'plan.json')
            previous = load(prior / 'plan.json')
            failed = load(prior / 'state.json')
            assert failed['status'] == 'blocked' and failed['completed_steps'] == 554
            assert failed['checkpoint_sha256'] == previous['checkpoint_sha256'] == digest
            assert previous['positivity_depth'] == 20 and previous['positivity_retry_depth'] == 24
            assert previous['host_preconditioner'] and previous['gpus'] == 4
            assert previous['gpu_library_sha256'] == replay_plan['gpu_library_sha256']
            assert any(row['kind'] == 'job_submission' and row['return_code'] == 0
                       and row['stdout'] == failed['job'] for row in read(prior))
            trace_path = target / 'custom_trace_000554.json'
            trace = load(trace_path)
            assert trace['step'] == 554 and trace['positivity_subdivision_retry_depth_limit'] == 24
            rows = trace['custom_newton_steps']
            assert not rows[-1]['accepted_custom_update'] and rows[-1]['next_regularization'] == 1e8
            assert any('stage=endpoint, element=813578' in (trial.get('positivity_failure') or '')
                       for row in rows for trial in row['trials'])
            last = next(row for row in reversed(rows) if row['accepted_custom_update'])
            retry_iterate = prior / f"slice-{failed['slice']:03d}" / 'latest-newton.npz'
            with np.load(retry_iterate, allow_pickle=False) as saved:
                assert int(saved['step']) == 554 and saved['checkpoint_sha256'].item() == digest
                assert float(saved['shift']) == last['next_regularization'] > stagnation_shift
                fields = np.load(target / 'displacement_coefficients.npy', mmap_mode='r')
                assert saved['u'].shape == fields.shape[1:] and np.isfinite(saved['u']).all()
            direction_evidence = dict(run=prior.name, job=failed['job'],
                                      trace_sha256=sha(trace_path), iterate_sha256=sha(retry_iterate))
        elif direction_comparison_from is not None:
            comparison = direction_comparison_from
            comparison_plan_path = comparison / 'plan.json'
            registered(ROOT, comparison, comparison_plan_path)
            comparison_plan = load(comparison_plan_path)
            comparison_state = load(comparison / 'state.json')
            assert comparison_state['status'] == 'diagnostic_complete'
            assert comparison_state['completed_steps'] == 554
            assert comparison_state['checkpoint_sha256'] == comparison_plan['checkpoint_sha256'] == digest
            assert comparison_plan['gpu_library_sha256'] == replay_plan['gpu_library_sha256']
            assert comparison_plan['host_preconditioner'] and comparison_plan['gpus'] == 4
            comparison_spec = comparison_plan['positivity_diagnostic']
            assert comparison_spec['kind'] == 'direction_comparison' and comparison_spec['element'] == 813578
            assert comparison_spec['shift'] == stagnation_shift == 1.0
            comparison_report = comparison / 'slice-001/direction-comparison.json'
            assert ROOT / comparison_state['result'] == comparison_report
            registered(ROOT, comparison, comparison_report)
            result = load(comparison_report)
            assert result['state'] == 555 and result['warm_start_matches']
            assert result['reference_state_accepted'] is False and result['force_target_N'] == 1e-9
            assert result['shift'] == stagnation_shift and result['stop_reason'] == 'direction_budget'
            assert result['requested_directions'] == comparison_spec['directions'] == 12
            corrected, baseline = result['corrected'], result['baseline_matched']
            assert corrected['directions'] == baseline['directions'] == 12
            assert corrected['accepted_updates'] > 0 and result['corrections_applied'] > 0
            assert 0 < corrected['force_N'] < min(baseline['force_N'], result['baseline_terminal']['force_N'])
            assert corrected['full_steps'] > baseline['full_steps']
            trace_path = comparison / 'baseline-trace.json'
            registered(ROOT, comparison, trace_path)
            assert sha(trace_path) == comparison_spec['baseline_trace_sha256']
            retry_iterate = comparison / 'diagnostic-seed.npz'
            registered(ROOT, comparison, retry_iterate)
            assert sha(retry_iterate) == comparison_spec['seed_sha256']
            direction_evidence = dict(run=comparison.name, job=comparison_state['job'],
                                      comparison_sha256=sha(comparison_report),
                                      trace_sha256=sha(trace_path), iterate_sha256=sha(retry_iterate))
            if pressure_after is not None:
                prior = pressure_after
                prior_plan_path = prior / 'plan.json'
                registered(ROOT, prior, prior_plan_path)
                prior_plan = load(prior_plan_path)
                prior_state = load(prior / 'state.json')
                assert prior_state['status'] == 'blocked' and prior_state['completed_steps'] == 554
                assert prior_state['checkpoint_sha256'] == prior_plan['checkpoint_sha256'] == digest
                assert prior_plan['determinant_direction_evidence'] == direction_evidence
                assert prior_plan['gpu_library_sha256'] == replay_plan['gpu_library_sha256']
                prior_trace = target / 'custom_trace_000554.json'
                trace = load(prior_trace)
                assert trace['step'] == 554 and trace['determinant_direction_element'] == 813578
                rows = trace['custom_newton_steps']
                limit = load(ROOT / state['stream_protocol'])['admissible_newton']['maximum_custom_newton_iterations']
                assert len(rows) == limit and rows[-1]['iteration'] == limit - 1
                prior_slice = prior / f"slice-{prior_state['slice']:03d}"
                ceiling_logs = [
                    path for path in sorted(prior_slice.glob('petsc-rank-*.log'))
                    if 'RuntimeError: Regularized Newton iteration ceiling reached' in path.read_text()
                ]
                assert ceiling_logs, 'Previous retry must have reached its Newton iteration ceiling'
                pressure_continuation = dict(run=prior.name, fractions=[0., .25, .5, .75, 1.],
                                             trace_sha256=sha(prior_trace),
                                             ceiling_log_sha256=sha(ceiling_logs[0]))
            elif residual_after is not None:
                prior = residual_after
                prior_plan_path = prior / 'plan.json'
                registered(ROOT, prior, prior_plan_path)
                prior_plan = load(prior_plan_path)
                prior_state = load(prior / 'state.json')
                assert prior_state['status'] == 'blocked' and prior_state['completed_steps'] == 554
                assert prior_state['checkpoint_sha256'] == prior_plan['checkpoint_sha256'] == digest
                assert prior_plan['determinant_direction_evidence'] == direction_evidence
                assert prior_plan['gpu_library_sha256'] == replay_plan['gpu_library_sha256']
                assert prior_plan['pressure_continuation']['fractions'][0] == 0.
                prior_slice = prior / f"slice-{prior_state['slice']:03d}"
                prior_trace = prior_slice / 'pressure-stage-0.json'
                trace = load(prior_trace)
                assert trace['step'] == 554 and trace['fraction'] == 0.
                assert trace['physical_state_accepted'] is False and not trace.get('complete')
                rows = trace['custom_newton_steps']
                limit = load(ROOT / state['stream_protocol'])['admissible_newton']['maximum_custom_newton_iterations']
                assert len(rows) == limit and rows[-1]['iteration'] == limit - 1
                residual_continuation = dict(run=prior.name, fractions=[i / 8 for i in range(1, 9)],
                                             anchor_sha256=sha(retry_iterate), trace_sha256=sha(prior_trace))
    elif screen_blends:
        audit_run = HPC / 'cmbbe-cell-audit-555-01'
        audit_report = audit_run / 'result.json'
        registered(ROOT, audit_run, audit_report)
        report = load(audit_report)
        assert report['accepted_states'] == 554 and report['element'] == 813578
        assert report['geometry']['nodal_coordinate_error'] == 0.
        previous_path = retry / 'plan.json'
        registered(ROOT, retry, previous_path)
        previous = load(previous_path)
        previous_state = load(retry / 'state.json')
        assert previous_state['status'] == 'blocked' and previous_state['completed_steps'] == 554
        assert previous['checkpoint_sha256'] == previous_state['checkpoint_sha256'] == digest
        assert previous['residual_continuation'] is not None
        assert sha(GPU_LIB) == previous['gpu_library_sha256']
        diagnostic_seed = retry / 'initial-newton.npz'
        seed_digest = sha(diagnostic_seed)
        assert seed_digest == previous['residual_continuation']['anchor_sha256']
        seed_events = [row for row in read(retry) if row['kind'] == 'artifact'
                       and row['path'] == str((retry / 'paused-iterate.npz').relative_to(ROOT))]
        assert seed_events and seed_events[-1]['sha256'] == seed_digest
        diagnostic = dict(kind='blend_screen', source_run=retry.name,
                          seed_sha256=seed_digest, audit_sha256=sha(audit_report),
                          element=813578, weights=[0., .25, .5, .75, .875, 1.],
                          whole_element_depth=20, retry_depth=24, margin=1e-9)
    elif compare_direction:
        failed_plan = load(retry / 'plan.json')
        registered(ROOT, retry, retry / 'plan.json')
        failed = load(retry / 'state.json')
        assert failed['status'] == 'blocked' and failed['completed_steps'] == 554
        assert failed['checkpoint_sha256'] == failed_plan['checkpoint_sha256'] == digest
        evidence = failed_plan['determinant_direction_evidence']
        baseline_run = HPC / evidence['run']
        baseline_plan_path = baseline_run / 'plan.json'
        registered(ROOT, baseline_run, baseline_plan_path)
        baseline_plan = load(baseline_plan_path)
        assert baseline_plan['checkpoint_sha256'] == digest
        assert baseline_plan['host_preconditioner'] and baseline_plan['gpus'] == 4
        assert baseline_plan['positivity_depth'] == 20 and baseline_plan['positivity_retry_depth'] == 24
        assert baseline_plan.get('determinant_direction_evidence') is None
        assert sha(GPU_LIB) == baseline_plan['gpu_library_sha256']
        baseline_trace = retry / 'prior-state-555-trace.json'
        registered(ROOT, retry, baseline_trace)
        assert sha(baseline_trace) == evidence['trace_sha256']
        baseline = load(baseline_trace)
        assert baseline['step'] == 554 and baseline['positivity_subdivision_retry_depth_limit'] == 24
        diagnostic_seed = baseline_run / 'initial-newton.npz'
        seed_events = [row for row in read(baseline_run) if row['kind'] == 'artifact'
                       and row['path'] == str((baseline_run / 'paused-iterate.npz').relative_to(ROOT))]
        seed_digest = sha(diagnostic_seed)
        assert seed_events and seed_digest == seed_events[-1]['sha256']
        rows = baseline['custom_newton_steps']
        assert rows[0]['regularization'] == baseline_plan['stagnation_shift']
        assert len(rows) >= 12 and not rows[-1]['accepted_custom_update']
        diagnostic = dict(kind='direction_comparison', source_run=baseline_run.name,
                          baseline_trace_sha256=sha(baseline_trace), seed_sha256=seed_digest,
                          shift=rows[0]['regularization'], directions=12, element=813578,
                          initial_force_N=rows[0]['initial_force_N'])
    elif diagnose_positivity:
        previous = load(retry / 'plan.json')
        registered(ROOT, retry, retry / 'plan.json')
        assert previous['gpus'] == 4 and previous['host_preconditioner']
        assert sha(GPU_LIB) == previous['gpu_library_sha256']
        assert previous['stagnation_shift'] is not None and previous['checkpoint_sha256'] == digest
        diagnostic_seed = retry / 'initial-newton.npz'
        evidence = [row for row in read(retry) if row['kind'] == 'artifact'
                    and row['path'] == str((retry / 'paused-iterate.npz').relative_to(ROOT))]
        assert evidence and sha(diagnostic_seed) == evidence[-1]['sha256']
        trace_path = target / 'custom_trace_000554.json'
        trace = load(trace_path)
        original_depth = previous.get('positivity_depth') or 12
        assert original_depth in (12, 16)
        iteration = 3 if original_depth == 16 else 0
        rows = trace['custom_newton_steps'][:iteration + 1]
        assert [row['iteration'] for row in rows] == list(range(iteration + 1))
        first, rejected = rows[0], rows[-1]
        full = rejected['trials'][0]
        assert trace['step'] == 554 and trace['line_search'] == 'directional_armijo_squared_force'
        assert trace.get('positivity_subdivision_depth_limit', 12) == original_depth
        assert first['regularization'] == previous['stagnation_shift']
        prefix = []
        for row in rows[:-1]:
            trial, = row['trials']
            assert row['accepted_custom_update'] and trial['alpha'] == 1
            prefix.append(dict(force_N=trial['native_force_N'], shift=row['next_regularization']))
        assert full['alpha'] == 1 and full['armijo_passed']
        assert full['positivity_failure'] == (
            'Whole-element positivity remains unresolved at maximum depth (stage=endpoint, element=813578)')
        diagnostic = dict(source_run=retry.name, source_trace_sha256=sha(trace_path),
                          seed_sha256=sha(diagnostic_seed), shift=first['regularization'],
                          iteration=iteration, original_depth=original_depth, prefix=prefix,
                          trial_shift=rejected['regularization'],
                          initial_force_N=rejected['initial_force_N'], trial_force_N=full['native_force_N'],
                          stage='endpoint', element=813578,
                          depths=[depth for depth in (16, 20, 24) if depth > original_depth])
    elif retry is not None:
        failed = load(retry / 'state.json')
        allowed = ('running', 'blocked') if resume_cancelled else ('blocked',)
        assert failed['status'] in allowed and failed['completed_steps'] == 554
        assert failed['checkpoint_sha256'] == digest
        assert any(row['kind'] == 'job_submission' and row['return_code'] == 0
                   and row['stdout'] == failed['job'] for row in read(retry))
        failed_slice = retry / f"slice-{failed['slice']:03d}"
        assert not (failed_slice / 'pause.json').exists()
        retry_log = failed_slice / 'generate.log'
        generator_log = retry_log.read_text()
        assert 'CUDA Newton operator' in generator_log
        rank_log = failed_slice / 'petsc-rank-00.log'
        if rank_log.is_file():
            retry_log = rank_log
        log = retry_log.read_text()
        error_code = 'error code 97'
        saved = failed_slice / 'latest-newton.npz'
        if stagnation_shift is not None:
            import numpy as np

            assert gpus == 4 and host_preconditioner
            previous = load(retry / 'plan.json')
            assert previous['gpus'] == 4 and previous['host_preconditioner']
            trace = target / 'custom_trace_000554.json'
            rows = load(trace)['custom_newton_steps']
            last = next(row for row in reversed(rows) if row['accepted_custom_update'])
            assert len(rows) >= 2 and not rows[-1]['accepted_custom_update']
            assert all(not t['armijo_passed'] for t in rows[-1]['trials'])
            assert any('unresolved at maximum depth' in (t['positivity_failure'] or '')
                       for t in last['trials'])
            assert rows[-1]['next_regularization'] == load(ROOT / state['stream_protocol'])[
                'admissible_newton']['maximum_regularization']
            with np.load(saved, allow_pickle=False) as checkpoint:
                assert int(checkpoint['step']) == 554
                assert checkpoint['checkpoint_sha256'].item() == digest
                assert checkpoint['u'].shape == np.load(
                    target / 'displacement_coefficients.npy', mmap_mode='r').shape[1:]
                assert np.isfinite(checkpoint['u']).all()
                assert float(checkpoint['shift']) == last['next_regularization'] > stagnation_shift
            for marker in markers:
                if marker.name == 'failure.json':
                    assert 'Regularization ceiling without admissible descent' in load(marker)['error']
            retry_log = trace
            retry_iterate = saved
        elif resume_cancelled or failed.get('blocker', '').startswith('RuntimeError: CUDA generator exited 9.'):
            assert host_preconditioner and load(retry / 'plan.json').get('host_preconditioner')
            assert 'REGULARIZED NEWTON 0 ' in generator_log
            if resume_cancelled:
                status = subprocess.check_output(['qstat', '-fx', failed['job']], text=True)
                assert re.search(r'^\s*job_state\s*=\s*F\s*$', status, re.MULTILINE), status
                assert saved.is_file(), 'Cancelled job has no saved iterate'
            else:
                assert 'Killed (signal 9)' in generator_log
            retry_log = failed_slice / 'generate.log'
            assert not any(marker.name == 'failure.json' for marker in markers)
            if saved.is_file():
                import numpy as np

                with np.load(saved, allow_pickle=False) as checkpoint:
                    assert int(checkpoint['step']) == 554
                    assert checkpoint['checkpoint_sha256'].item() == digest
                    assert np.isfinite(checkpoint['u']).all()
                    assert np.isfinite(checkpoint['shift']) and float(checkpoint['shift']) > 0
                retry_iterate = saved
        else:
            assert 'PETSc.KSP.setUp' in log
            if load(retry / 'plan.json').get('host_preconditioner'):
                assert host_preconditioner and 'REGULARIZED NEWTON 0 ' in generator_log
                error_code = 'error code 76'
            assert error_code in log
            for marker in markers:
                if marker.name == 'failure.json':
                    assert error_code in load(marker)['error']
    else:
        assert not markers, 'Existing failure artifacts require a reviewed retry'

    protocol = load(ROOT / state['stream_protocol'])
    assert protocol['parallel_ranks'] == 12
    # The frozen generator requires finite JSON numbers for these legacy fields.
    # Actual scheduling is controlled by the PBS slice and the collective deadline.
    protocol['resource_protocol']['native_hours'] = 1e99
    protocol['per_step_wall_seconds'] = 1e99
    protocol['stop_after_steps'] = 1600
    protocol_path = run / 'stream-protocol.json'
    assert GPU_LIB.is_file() and (NATIVE / 'bin/mpiexec').is_file()
    run.mkdir()
    if retry_iterate is not None:
        os.link(retry_iterate, run / 'paused-iterate.npz')
        if stagnation_shift is not None:
            os.link(retry_iterate, run / 'initial-newton.npz')
        artifact(ROOT, run, run / 'paused-iterate.npz', 'Recovered state-555 Newton iterate')
    if direction_evidence is not None:
        preserved = run / 'prior-state-555-trace.json'
        shutil.copy2(trace_path, preserved)
        assert sha(preserved) == direction_evidence['trace_sha256']
        artifact(ROOT, run, preserved, 'Previous determinant-limited state-555 solve')
        if comparison_report is not None:
            preserved_report = run / 'prior-direction-comparison.json'
            shutil.copy2(comparison_report, preserved_report)
            assert sha(preserved_report) == direction_evidence['comparison_sha256']
            artifact(ROOT, run, preserved_report, 'Completed original warm-start comparison')
    if pressure_continuation is not None:
        for source, name in ((prior_trace, 'prior-pressure-retry-trace.json'),
                             (ceiling_logs[0], 'prior-pressure-retry-ceiling.log')):
            destination = run / name
            shutil.copy2(source, destination)
            artifact(ROOT, run, destination, 'Previous exact-pressure iteration ceiling evidence')
    if residual_continuation is not None:
        preserved_trace = run / 'prior-pressure-stage-0.json'
        shutil.copy2(prior_trace, preserved_trace)
        assert sha(preserved_trace) == residual_continuation['trace_sha256']
        artifact(ROOT, run, preserved_trace, 'Previous pressure continuation attempt')
    if diagnostic is not None:
        os.link(diagnostic_seed, run / 'diagnostic-seed.npz')
        artifact(ROOT, run, run / 'diagnostic-seed.npz', 'Read-only replay input')
        if compare_direction:
            shutil.copy2(baseline_trace, run / 'baseline-trace.json')
            artifact(ROOT, run, run / 'baseline-trace.json', 'Matched direction-comparison baseline')
    atomic(protocol_path, protocol)
    plan = dict(inherited, run_id=run.name, queue='iworkq',
                stream_protocol=str(protocol_path.relative_to(ROOT)),
                checkpoint_sha256=digest, gpu_library_sha256=sha(GPU_LIB),
                host_preconditioner=host_preconditioner, gpus=gpus, stagnation_shift=stagnation_shift,
                positivity_diagnostic=diagnostic, positivity_depth=positivity_depth,
                positivity_retry_depth=24 if adaptive_positivity else None,
                determinant_direction_evidence=direction_evidence,
                pressure_continuation=pressure_continuation, residual_continuation=residual_continuation,
                sources={path: sha(ROOT / path) for path in SOURCES})
    if retry_log is not None:
        plan['retry_evidence'] = dict(run=retry.name, log_sha256=sha(retry_log), resume_cancelled=resume_cancelled)
        if stagnation_shift is not None and positivity_depth is None:
            preserved_trace = run / 'prior-state-555-trace.json'
            shutil.copy2(retry_log, preserved_trace)
            artifact(ROOT, run, preserved_trace, 'Armijo and positivity failure evidence')
        for marker in markers:
            archived = run / ('prior-' + marker.name)
            os.link(marker, archived)
            artifact(ROOT, run, archived, 'Archived state-555 failure artifact')
            marker.unlink()
    atomic(run / 'plan.json', plan)
    artifact(ROOT, run, run / 'plan.json', 'CUDA production plan')
    artifact(ROOT, run, protocol_path, 'Frozen generator protocol for PBS CUDA slices')
    command = shlex.join([str(MODERN), str(Path(__file__).resolve()), 'execute', '--run-id', run.name])
    script = run / 'job.pbs'
    script.write_text(f'#!/bin/bash\nset -eu\ncd {shlex.quote(str(ROOT))}\nexec {command}\n')
    artifact(ROOT, run, script, 'PBS slice command')
    atomic(run / 'state.json', dict(status='submitted', completed_steps=554, checkpoint_sha256=digest))


def execute(run):
    started = time.monotonic()
    deadline = started + WALL_SECONDS - 900
    job = os.environ['PBS_JOBID']
    with (PARENT / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = load(run / 'state.json')
        try:
            execute_locked(run, state, job, deadline)
        except Exception as exc:
            state.update(status='blocked', blocker=f'{type(exc).__name__}: {exc}')
            atomic(run / 'state.json', state)
            raise


def execute_locked(run, state, job, deadline):
    entry, = [row for row in read(run) if row['kind'] == 'job_submission'
              and row['return_code'] == 0 and row['stdout'] == job]
    number = entry['slice']
    assert state['status'] in ('submitted', 'paused')
    plan = load(run / 'plan.json')
    protocol = ROOT / plan['stream_protocol']
    for path in (run / 'plan.json', protocol, run / 'job.pbs'):
        registered(ROOT, run, path)
    for path, digest in plan['sources'].items():
        assert sha(ROOT / path) == digest, path
    assert sha(GPU_LIB) == plan['gpu_library_sha256']
    assert len(set(Path(os.environ['PBS_NODEFILE']).read_text().splitlines())) == 1
    target = ROOT / 'data/processed' / plan['target_name']
    assert sha(target / 'restart.npz') == state['checkpoint_sha256']
    output = run / f'slice-{number:03d}'
    output.mkdir()
    state.update(status='running', slice=number, job=job)
    atomic(run / 'state.json', state)
    env = os.environ.copy()
    for name in ('LD_PRELOAD', 'PETSC_DIR', 'PETSC_ARCH', 'PETSC_OPTIONS', 'PKG_CONFIG_SYSROOT_DIR'):
        env.pop(name, None)
    env.update(PATH=f'{NATIVE}/bin:{env["PATH"]}', LD_LIBRARY_PATH=f'{NATIVE}/lib:{CUDA}/lib64',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HYDRA_LAUNCHER='fork', FI_PROVIDER='tcp')
    pc_paths = f'{NATIVE}/lib/pkgconfig:{NATIVE}/share/pkgconfig'
    env.update(XDG_CACHE_HOME=str(ROOT / '.hpc/cache'),
               MPLCONFIGDIR=str(ROOT / '.hpc/cache/matplotlib'),
               DIJITSO_CACHE_DIR=str(ROOT / '.hpc/fenics-cache'),
               NUMEXPR_NUM_THREADS='1')
    env.update(PKG_CONFIG=str(NATIVE / 'bin/pkg-config'), PKG_CONFIG_PATH=pc_paths,
               PKG_CONFIG_LIBDIR=pc_paths)
    cpu = [str(NATIVE / 'bin/python')]

    def run_command(label, args, environment=env):
        with (output / f'{label}.log').open('a') as log:
            return subprocess.run(args, env=environment, cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT).returncode

    def pause():
        state.update(status='paused')
        atomic(run / 'state.json', state)
        submit(run, number + 1, after=job)

    if state['completed_steps'] < 1600:
        gpus = plan.get('gpus', 1)
        visible = env.get('CUDA_VISIBLE_DEVICES')
        if visible:
            devices = visible.split(',')
        elif os.environ.get('PBS_GPUFILE'):
            devices = []
            for gpu_entry in Path(os.environ['PBS_GPUFILE']).read_text().splitlines():
                match = re.search(r'(?:^|gpu)(\d+)$', gpu_entry)
                assert match, f'Unknown PBS GPU entry: {gpu_entry}'
                devices.append(match[1])
        elif gpus == 4:
            # All four GPUs are requested on the exclusive vnode.
            listing = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], env=env, text=True)
            devices = listing.splitlines()
            assert len(devices) == gpus
        else:
            devices = ['0']
        devices = list(dict.fromkeys(device.strip() for device in devices if device.strip()))
        assert len(devices) >= gpus, f'PBS exposed {len(devices)} GPUs for a {gpus}-GPU run'
        gpu_env = dict(env, CUDA_VISIBLE_DEVICES=','.join(devices[:gpus]),
                       LV_CUDA_GPU_COUNT=str(gpus), PETSC_OPTIONS='-use_gpu_aware_mpi 0',
                       PKG_CONFIG_PATH=f'{GPU}/lib/pkgconfig:{pc_paths}',
                       PKG_CONFIG_LIBDIR=f'{GPU}/lib/pkgconfig:{pc_paths}',
                       LV_CUDA_HOST_PRECONDITIONER=str(int(plan.get('host_preconditioner', False))),
                       LV_CUDA_SLICE=str(output), LV_CUDA_DEADLINE=str(deadline),
                       LV_REFERENCE_TARGET=str(target))
        if plan.get('stagnation_shift') is not None:
            gpu_env['LV_CUDA_STAGNATION_SHIFT'] = str(plan['stagnation_shift'])
        if plan.get('positivity_depth') is not None:
            gpu_env['LV_CUDA_POSITIVITY_DEPTH'] = str(plan['positivity_depth'])
        if plan.get('positivity_retry_depth') is not None:
            gpu_env['LV_CUDA_POSITIVITY_RETRY_DEPTH'] = str(plan['positivity_retry_depth'])
        if plan.get('determinant_direction_evidence') is not None:
            gpu_env['LV_CUDA_DETERMINANT_DIRECTION'] = '1'
        if plan.get('pressure_continuation') is not None:
            gpu_env['LV_CUDA_PRESSURE_FRACTIONS'] = ','.join(
                str(fraction) for fraction in plan['pressure_continuation']['fractions'])
        if plan.get('residual_continuation') is not None:
            gpu_env['LV_CUDA_RESIDUAL_FRACTIONS'] = ','.join(
                str(fraction) for fraction in plan['residual_continuation']['fractions'])
            gpu_env['LV_CUDA_RESIDUAL_ANCHOR_SHA256'] = plan['residual_continuation']['anchor_sha256']
        if plan.get('positivity_diagnostic'):
            gpu_env['LV_CUDA_DIAGNOSTIC_PLAN'] = str(run / 'plan.json')
        result = run_command('generate', [
            str(NATIVE / 'bin/mpiexec'), '-n', '12', '/usr/bin/env', f'LD_PRELOAD={GPU_LIB}',
            *cpu, 'hpc/production_cuda_native.py', 'generate', '--policy', str(protocol),
            '--protocol', str(protocol)], gpu_env)
        if plan.get('positivity_diagnostic'):
            if result:
                raise RuntimeError(f'Positivity replay exited {result}. See {output / "generate.log"}')
            assert sha(target / 'restart.npz') == state['checkpoint_sha256']
            assert load(target / 'manifest.json')['completed_steps'] == state['completed_steps']
            kind = plan['positivity_diagnostic'].get('kind')
            name = {'direction_comparison': 'direction-comparison.json',
                    'blend_screen': 'blend-screen.json'}.get(kind, 'positivity-replay.json')
            summary = output / name
            report = load(summary)
            key = {'direction_comparison': 'warm_start_matches',
                   'blend_screen': 'screen_complete'}.get(kind, 'reproduced')
            assert report[key]
            artifact(ROOT, run, summary, 'State-555 diagnostic result')
            state.update(status='diagnostic_complete', result=str(summary.relative_to(ROOT)))
            atomic(run / 'state.json', state)
            return
        updated = load(target / 'manifest.json')
        count = updated['completed_steps']
        digest = state['checkpoint_sha256']
        for index in range(state['completed_steps'], count):
            journal = load(target / 'step_journal' / f'{index:06d}.json')
            assert journal['input_checkpoint_sha256'] == digest
            assert journal['record'] == updated['steps'][index]
            digest = journal['accepted_checkpoint_sha256']
        assert sha(target / 'restart.npz') == digest
        state.update(completed_steps=count, checkpoint_sha256=digest)
        if result:
            receipt_path = output / 'pause.json'
            if not receipt_path.is_file():
                raise RuntimeError(
                    f'CUDA generator exited {result}. See {output / "generate.log"}'
                )
            receipt = load(receipt_path)
            failed = target / 'failure.json'
            if failed.exists():
                assert load(failed)['error'] == 'CUDA production slice deadline'
            assert receipt['step'] == count and receipt['checkpoint_sha256'] == digest
            shutil.copy2(target / f'custom_trace_{count:06d}.json', output / 'paused-trace.json')
            iterate = output / 'paused.npz'
            assert sha(iterate) == receipt['iterate_sha256']
            os.link(iterate, output / 'resume-link.npz')
            (output / 'resume-link.npz').replace(run / 'paused-iterate.npz')
            for name in ('failure.json', 'UNACCEPTED_iterate.npz'):
                marker = target / name
                if marker.exists():
                    marker.replace(output / name)
            pause()
            return
        assert count == 1600 and updated['complete']
        atomic(run / 'state.json', state)

    if deadline - time.monotonic() < 300:
        pause()
        return
    if not state.get('positivity_finalized'):
        assert run_command('positivity', [str(MODERN), 'scripts/finalize_parallel_reference.py',
                                          '--protocol', str(protocol)]) == 0
        state.update(positivity_finalized=True)
        atomic(run / 'state.json', state)
    while not state.get('audit_passed'):
        if deadline - time.monotonic() < 7200:
            pause()
            return
        checked = state.get('audit_checked_steps', 0)
        audit = output / f'audit-{checked:04d}.json'
        assert run_command('audit', [*cpu, 'hpc/platform_native_chunked.py',
                                     '--spec', str(run / 'plan.json'), '--mode', 'final',
                                     '--output', str(audit), '--checkpoint-root',
                                     str(run / 'audit-checkpoints'), '--max-steps', '64']) == 0
        result = load(audit)
        assert result['chunk_passed'] and (result['checked_steps'] > checked or result['passed'])
        state.update(audit_passed=result['audit_complete'] and result['passed'],
                     audit_checked_steps=result['checked_steps'])
        atomic(run / 'state.json', state)
    if deadline - time.monotonic() < 8 * 3600:
        pause()
        return
    mesh = run / 'mesh-comparison.json'
    assert run_command('mesh', [*cpu, 'scripts/compare_quadratic_references.py', '--config-a',
                                'configs/physics/dynamic_quadratic_reference_halfstep_ew_refined.json',
                                '--config-b', plan['config'], '--output', str(mesh)]) == 0
    assert load(mesh)['passes_declared_one_percent_sensitivity']
    state.update(status='reference_complete')
    atomic(run / 'state.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'execute'))
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--retry-after', metavar='BLOCKED_RUN')
    parser.add_argument('--host-preconditioner', action='store_true',
                        help='Build GAMG from the host AIJ matrix and solve with the CUDA matrix')
    parser.add_argument('--gpus', type=int, choices=(1, 4), default=1)
    parser.add_argument('--resume-cancelled', action='store_true')
    parser.add_argument('--stagnation-shift', type=float,
                        help='Retry a verified state-555 regularization ceiling with directional Armijo')
    parser.add_argument('--diagnose-positivity', action='store_true',
                        help='Replay one rejected trial and check the unresolved element at greater depth')
    parser.add_argument('--certify-depth-from', metavar='REPLAY_RUN',
                        help='Resume production with the depth certified by a registered replay')
    parser.add_argument('--adaptive-positivity', action='store_true',
                        help='Retry unresolved depth-20 cells at depth 24 for state 555')
    parser.add_argument('--determinant-direction-from', metavar='BLOCKED_RUN',
                        help='Warm-start a determinant-corrected state-555 solve from a blocked run')
    parser.add_argument('--compare-direction', action='store_true',
                        help='Compare 12 directions from the original warm start of the blocked retry')
    parser.add_argument('--direction-comparison-from', metavar='COMPARISON_RUN',
                        help='Start corrected production from a verified comparison warm start')
    parser.add_argument('--pressure-continuation-after', metavar='BLOCKED_RUN',
                        help='Continue pressure from state 554 using fixed state-554 histories')
    parser.add_argument('--residual-continuation-after', metavar='BLOCKED_RUN',
                        help='Continue the residual at the exact state-555 pressure')
    parser.add_argument('--screen-blends', action='store_true',
                        help='Screen blends of accepted state 554 and the original warm start')
    args = parser.parse_args()
    if args.screen_blends and (
            args.action != 'submit' or not args.retry_after or args.resume_cancelled
            or args.compare_direction or args.diagnose_positivity or args.stagnation_shift is not None
            or args.certify_depth_from or args.gpus != 4 or not args.host_preconditioner):
        parser.error('--screen-blends requires --retry-after, --gpus 4 and --host-preconditioner')
    if args.residual_continuation_after and (
            not args.direction_comparison_from or args.pressure_continuation_after):
        parser.error('--residual-continuation-after requires --direction-comparison-from')
    if args.pressure_continuation_after and not args.direction_comparison_from:
        parser.error('--pressure-continuation-after requires --direction-comparison-from')
    if args.direction_comparison_from and (
            not args.certify_depth_from or not args.adaptive_positivity or args.determinant_direction_from):
        parser.error('--direction-comparison-from requires --certify-depth-from and --adaptive-positivity')
    if args.compare_direction and (
            args.action != 'submit' or not args.retry_after or args.resume_cancelled
            or args.diagnose_positivity or args.stagnation_shift is not None or args.certify_depth_from
            or args.gpus != 4 or not args.host_preconditioner):
        parser.error('--compare-direction requires --retry-after, --gpus 4 and --host-preconditioner')
    if args.determinant_direction_from and (not args.certify_depth_from or not args.adaptive_positivity):
        parser.error('--determinant-direction-from requires --certify-depth-from and --adaptive-positivity')
    if args.adaptive_positivity and not args.certify_depth_from:
        parser.error('--adaptive-positivity requires --certify-depth-from')
    if args.certify_depth_from and (
            args.action != 'submit' or args.retry_after or args.resume_cancelled
            or args.diagnose_positivity or args.stagnation_shift is not None
            or args.gpus != 4 or not args.host_preconditioner):
        parser.error('--certify-depth-from requires submission with --gpus 4 and --host-preconditioner')
    if args.diagnose_positivity and (
            not args.retry_after or args.resume_cancelled or args.stagnation_shift is not None
            or args.gpus != 4 or not args.host_preconditioner):
        parser.error('--diagnose-positivity requires --retry-after, --gpus 4 and --host-preconditioner')
    if args.stagnation_shift is not None and (
            not math.isfinite(args.stagnation_shift) or args.stagnation_shift <= 0
            or not args.retry_after or args.resume_cancelled):
        parser.error('--stagnation-shift requires a positive shift and --retry-after')
    if args.resume_cancelled and not args.retry_after:
        parser.error('--resume-cancelled requires --retry-after')
    if any(not re.fullmatch(r'[\w-]+', value)
           for value in (args.run_id, args.retry_after, args.certify_depth_from,
                         args.determinant_direction_from, args.direction_comparison_from,
                         args.pressure_continuation_after, args.residual_continuation_after) if value is not None):
        parser.error('Invalid run ID')
    run = HPC / args.run_id
    if args.action == 'execute':
        if args.retry_after is not None or args.host_preconditioner or args.gpus != 1:
            parser.error('Retry and preconditioner options apply to submission only')
        execute(run)
    else:
        with (ROOT / '.hpc/profile-submission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert_idle(ROOT)
            with (PARENT / 'execution.lock').open('a') as execution:
                fcntl.flock(execution, fcntl.LOCK_EX | fcntl.LOCK_NB)
                initialize(run, HPC / args.retry_after if args.retry_after else None,
                           args.host_preconditioner, args.gpus, args.resume_cancelled, args.stagnation_shift,
                           args.diagnose_positivity,
                           HPC / args.certify_depth_from if args.certify_depth_from else None,
                           args.adaptive_positivity,
                           HPC / args.determinant_direction_from if args.determinant_direction_from else None,
                           args.compare_direction,
                           HPC / args.direction_comparison_from if args.direction_comparison_from else None,
                           HPC / args.pressure_continuation_after if args.pressure_continuation_after else None,
                           HPC / args.residual_continuation_after if args.residual_continuation_after else None,
                           args.screen_blends)
            submit(run, 1)


if __name__ == '__main__':
    main()
