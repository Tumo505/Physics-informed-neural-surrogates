"""One-state experiment from the preserved cofactor reference checkpoint."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

P = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P / 'src'))
from registry import artifact, atomic, environment_fingerprint, event, job_metadata, sha
from submit_probe_profile import assert_idle, registered


def load(path):
    return json.loads(path.read_text())


def check_trace(trace):
    steps = trace['custom_newton_steps'][-8:]
    assert trace['step'] == 550 and len(steps) == 8
    for low, high in zip(steps[::2], steps[1::2]):
        assert low['regularization'] == high['next_regularization'] == .00390625
        assert high['regularization'] == low['next_regularization'] == .015625
        assert low['accepted_custom_update'] and high['accepted_custom_update']
        assert low['trials'][0]['alpha'] == 1 and not low['trials'][0]['armijo_passed']
        assert low['trials'][-1]['whole_element_admissible'] and low['trials'][-1]['alpha'] < 1
        assert high['trials'][-1]['whole_element_admissible'] and high['trials'][-1]['alpha'] == 1
        assert high['trials'][-1]['actual_to_predicted_merit_ratio'] > .75


def check_followup(trace):
    rows = trace['custom_newton_steps'][-8:]
    assert trace['step'] == 550 and len(rows) == 8
    for high, low in zip(rows[::2], rows[1::2]):
        assert high['regularization'] == low['next_regularization'] == .03125
        assert low['regularization'] == high['next_regularization'] == .0078125
        assert high['accepted_custom_update'] and low['accepted_custom_update']
        high_trial, low_trial = high['trials'][-1], low['trials'][-1]
        assert high_trial['alpha'] == 1 and high_trial['whole_element_admissible']
        assert high_trial['actual_to_predicted_merit_ratio'] > .75
        assert low_trial['whole_element_admissible']
        if low_trial['alpha'] < 1:
            assert low['trials'][0]['alpha'] == 1 and not low['trials'][0]['armijo_passed']
        else:
            assert low_trial['actual_to_predicted_merit_ratio'] < .25


def submit(run, parent, previous=None, *, compare_shifts=False, queue=None, resume=None):
    assert not compare_shifts or previous is not None, "Supply the failed Jacobian reuse probe"
    assert resume is None or compare_shifts, "Use --compare-shifts with --resume-comparison"
    assert_idle(P)
    registered(P, parent, parent / 'plan.json')
    source, state = (load(parent / name) for name in ('plan.json', 'state.json'))
    assert state['status'] == 'blocked' and state['terminal_rejection']
    assert state['phase'] == 'reference_000547' and state['attempts'][state['phase']] == 1
    assert source['worker_policy']['implementation'] == 'cofactor'
    assert source['per_step_seconds'] == 1800
    log = parent / state['phase'] / 'attempt-001/command.log'
    registered(P, parent, log)
    assert 'RuntimeError: Regularized Newton wall budget exhausted' in log.read_text()
    data = P / 'data/processed' / source['target_name']
    trace = data / 'custom_trace_000550.json'
    check_trace(load(trace))
    manifest = load(data / 'manifest.json')
    assert manifest['completed_steps'] == 550 and manifest['n_steps'] == 1600
    assert manifest['prefix_independent_native_audit_passed']
    cumulative_seconds = state['cumulative_seconds']
    policy = 'band_half'
    reuse_jacobian = False
    scope = 'Half reduction only in the observed regularization band.'
    if previous is not None:
        for name in ('plan.json', 'summary.json'):
            registered(P, previous, previous / name)
        prior_plan, prior_result = (load(previous / name) for name in ('plan.json', 'summary.json'))
        assert prior_plan['parent_run'] == parent.name and prior_plan['target_name'] == source['target_name']
        prior_policy = prior_plan.get('regularization_policy', 'band_half')
        assert prior_policy in ('band_half', 'two_good_steps')
        assert bool(prior_plan.get('reuse_jacobian', False)) == compare_shifts
        for path in (parent / 'plan.json', data / 'manifest.json', data / 'restart.npz'):
            assert prior_plan['frozen'][str(path.relative_to(P))] == sha(path)
        assert prior_result['reference_continuation_allowed'] is False
        assert prior_result['complete'] and not prior_result['passed'] and prior_result['step'] == 550
        assert not prior_result['physical_new_state_generated']
        assert 'Regularized Newton wall budget exhausted' in prior_result['stop_reason']
        previous_trace = load(previous / 'trace.json')
        if compare_shifts:
            assert prior_policy == 'two_good_steps' and not prior_plan.get('compare_shifts')
            assert prior_result['reused_jacobian_updates'] > 0
            assert len(previous_trace['custom_newton_steps']) >= 28
            reuse_jacobian = True
        elif prior_policy == 'band_half':
            check_followup(previous_trace)
        else:
            timings = prior_result['phase_seconds_rank0']
            assembly = timings['jacobian_assembly']
            assert assembly > .5 * sum(timings.values()), 'Jacobian assembly must dominate measured time'
            assert previous_trace['step'] == 550 and previous_trace['regularization_policy'] == prior_policy
            reuse_jacobian = True
        assert len(previous_trace['custom_newton_steps']) == prior_result['accepted_custom_iterations']
        assert previous_trace['custom_newton_steps'][-1]['trials'][-1]['native_force_N'] == prior_result['last_accepted_custom_force_N']
        cumulative_seconds = prior_result['cumulative_seconds']
        assert cumulative_seconds >= state['cumulative_seconds']
        policy = 'two_good_steps'
        scope = 'Halve regularization after two consecutive good full steps at the same shift.'
    if reuse_jacobian:
        scope += ' Reuse the Jacobian for one update after a good full step, with a fresh rebuild on rejection.'
    if compare_shifts:
        scope = 'Replay 28 Newton updates, then compare three fresh directions from the same field.'
    resume_paths = []
    saved_iterate = None
    if resume is not None:
        iterate = resume / 'comparison-iterate.npy'
        registered(P, resume, iterate)
        registered(P, resume, resume / 'plan.json')
        resume_plan = load(resume / 'plan.json')
        assert resume_plan['compare_shifts'] and not resume_plan.get('saved_iterate')
        assert resume_plan['previous_recovery'] == previous.name
        for path in (parent / 'plan.json', data / 'manifest.json', data / 'restart.npz'):
            assert resume_plan['frozen'][str(path.relative_to(P))] == sha(path)
        replay = load(resume / 'trace.json')
        assert replay['step'] == 550 and len(replay['custom_newton_steps']) == 28
        expected_force = replay['custom_newton_steps'][-1]['trials'][-1]['native_force_N']
        log_path = resume / 'pbs.log'
        match = re.search(r'job killed: walltime (\d+) exceeded limit 3600', log_path.read_text())
        assert match, 'Expected the PBS walltime termination record'
        cumulative_seconds += int(match.group(1))
        resume_paths = [iterate, resume / 'plan.json', resume / 'trace.json', resume / 'events.jsonl', log_path]
        saved_iterate = str(iterate.relative_to(P))
        scope = 'Compare three fresh directions from the saved displacement field.'
    assert cumulative_seconds + 3600 < source['cumulative_pipeline_budget_hours'] * 3600
    frozen = {**source['scientific_sources'], **source['inherited_evidence_sha256']}
    frozen[source['config']] = source['reference_config_sha256']
    frozen[source['snapshot']] = source['snapshot_sha256']
    for name in ('probe_verification', 'stream_protocol'):
        path = P / state[name]
        frozen[str(path.relative_to(P))] = sha(path)
    seed_audit = P / manifest['prefix_native_audit']
    registered(P, parent, seed_audit)
    audit = load(seed_audit)
    assert audit['complete'] and audit['passed'] and audit['checked_steps'] == manifest['seed_steps']
    frozen[str(seed_audit.relative_to(P))] = sha(seed_audit)
    for path in (parent / 'plan.json', parent / 'state.json', log, trace, data / 'manifest.json',
                 data / 'restart.npz', data / 'step_journal/000548.json',
                 data / 'step_journal/000549.json', Path(__file__).resolve(),
                 P / 'src/lv_surrogate/physics/admissible_newton_gentle_shrink.py'):
        frozen[str(path.relative_to(P))] = sha(path)
    policy_path = P / 'src/lv_surrogate/physics/regularization_recovery_policy.py'
    frozen[str(policy_path.relative_to(P))] = sha(policy_path)
    if previous is not None:
        for name in ('plan.json', 'summary.json', 'trace.json'):
            path = previous / name
            frozen[str(path.relative_to(P))] = sha(path)
    for path in resume_paths:
        frozen[str(path.relative_to(P))] = sha(path)
    for path, digest in frozen.items():
        assert sha(P / path) == digest, 'Changed input or source: ' + path
    plan = dict(source, queue=queue or source['queue'], compare_shifts=compare_shifts, saved_iterate=saved_iterate,
                saved_iterate_force_N=expected_force if resume else None, run_id=run.name, step=550, frozen=frozen, parent_run=parent.name,
                cumulative_seconds_before=cumulative_seconds, generation_seconds_before=state['generation_seconds'],
                regularization_policy=policy, reuse_jacobian=reuse_jacobian, previous_recovery=previous.name if previous else None,
                scope=f'One equation from checkpoint 550. {scope} 1800 s solver cap. Candidate fields only.')
    run.mkdir()
    atomic(run / 'plan.json', plan)
    atomic(run / 'state.json', dict(status='submitted', phase='recovery_probe', blockers=[],
                                   generation_seconds=state['generation_seconds'],
                                   cumulative_seconds=cumulative_seconds))
    artifact(P, run, run / 'plan.json', 'State-551 recovery probe protocol')
    prefix = P / plan['native_prefix']
    pkgconfig = f'{prefix}/lib/pkgconfig:{prefix}/share/pkgconfig'
    command = [str(P / '.tools/bin/micromamba'), '--no-rc', 'run', '-r', str(P / '.hpc/mamba'),
               '-p', str(prefix), 'mpiexec', '-n', str(plan['ranks']), 'python',
               'hpc/probe_regularization_recovery.py', 'execute', '--run-id', run.name]
    script = run / 'job.pbs'
    script.write_text('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(P)) + '\n'
        'unset CONDARC MAMBARC CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL BASH_ENV ENV FI_TCP_IFACE PKG_CONFIG_SYSROOT_DIR\n'
        + 'export PATH=' + shlex.quote(f'{prefix}/bin:{P}/.hpc/tools/bin:/usr/bin:/bin') + '\n'
        + 'export PKG_CONFIG=' + shlex.quote(str(prefix / 'bin/pkg-config')) + '\n'
        + 'export PKG_CONFIG_PATH=' + shlex.quote(pkgconfig) + '\nexport PKG_CONFIG_LIBDIR="$PKG_CONFIG_PATH"\n'
        + 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 '
          'HYDRA_LAUNCHER=fork FI_PROVIDER=tcp CUDA_VISIBLE_DEVICES=""\n'
        + 'export XDG_CACHE_HOME=' + shlex.quote(str(P / '.hpc/cache')) + '\n'
        + 'export MPLCONFIGDIR=' + shlex.quote(str(P / '.hpc/cache/matplotlib')) + '\n'
        + 'export DIJITSO_CACHE_DIR=' + shlex.quote(str(P / '.hpc/fenics-cache')) + '\n'
        + 'exec ' + shlex.join(command) + '\n')
    args = ['qsub', '-N', 'lv_recovery551', '-q', plan['queue'], '-l',
            f"select=1:ncpus={plan['cpus']}:mem={plan['memory_gb']}gb",
            '-l', 'walltime=01:00:00', '-l', 'place=excl', '-l', 'software=' + plan['software'],
            '-j', 'oe', '-o', str(run / 'pbs.log')]
    if plan.get('account'):
        args += ['-A', plan['account']]
    args.append(str(script))
    reply = subprocess.run(args, cwd=P, capture_output=True, text=True)
    event(run, 'job_submission', argv=args, return_code=reply.returncode,
          stdout=reply.stdout.strip(), stderr=reply.stderr.strip())
    if reply.returncode:
        status = load(run / 'state.json')
        status.update(status='submission_failed', blockers=[reply.stderr.strip()])
        atomic(run / 'state.json', status)
        raise SystemExit(reply.stderr)
    print('State-551 probe job:', reply.stdout.strip())
    print('Result:', (run / 'summary.json').relative_to(P))


def execute(run):
    assert os.environ.get('PBS_JOBID'), 'Run the full-size probe via PBS'
    from mpi4py import MPI
    import numpy as np
    comm = MPI.COMM_WORLD
    rank = comm.rank
    started = time.monotonic()
    plan = load(run / 'plan.json')
    status = {}
    result = dict(complete=False, passed=False, step=550, physical_new_state_generated=False,
                  reference_continuation_allowed=False, stop_reason=None)

    def on_root(function):
        value, error = None, None
        if rank == 0:
            try:
                value = function()
            except Exception as exc:
                traceback.print_exc()
                error = f'{type(exc).__name__}: {exc}'
        error = comm.bcast(error, root=0)
        if error:
            raise RuntimeError(error)
        return value

    def claim_run():
        lock = (run / 'execution.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registered(P, run, run / 'plan.json')
        status.update(load(run / 'state.json'))
        assert status['status'] == 'submitted' and not (run / 'summary.json').exists()
        status.update(status='running')
        atomic(run / 'state.json', status)
        return lock

    def preflight():
        metadata = job_metadata()
        event(run, 'job_started', **metadata)
        assert len(metadata.get('allocated_hosts', [])) == 1
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
        assert available >= plan['host_free_reserve_gib'] * 2**30
        assert shutil.disk_usage(P).free >= 100 * 2**30
        for path, digest in plan['frozen'].items():
            assert sha(P / path) == digest, 'Changed input or source: ' + path
        for path, digest in plan['environment_sha256'].items():
            assert environment_fingerprint(P / path) == digest, 'Changed environment: ' + path

    lock = on_root(claim_run)
    data = P / 'data/processed' / plan['target_name']
    step = plan['step']
    trace = dict(step=step, custom_newton_steps=[])
    try:
        on_root(preflight)
        manifest = load(data / 'manifest.json')
        assert comm.size == plan['ranks'] == 12 and step == 550
        from lv_surrogate.physics.parallel_fem import ParallelP2
        from lv_surrogate.physics.reference_fem import residual
        from lv_surrogate.physics.admissible_newton_gentle_shrink import solve
        from lv_surrogate.physics.threaded_bernstein import ThreadedBernsteinAudit
        from cofactor_proof import CofactorBernsteinAudit
        from proof_benchmark import same_proof
        stream = load(P / load(P / 'experiments/hpc' / plan['parent_run'] / 'state.json')['stream_protocol'])
        cfg = manifest['config']
        assert cfg == load(P / plan['config'])
        assert manifest['config_sha256'] == plan['reference_config_sha256']
        assert manifest['completed_steps'] == step and stream['per_step_wall_seconds'] == 1800

        def load_checkpoint():
            assert sha(data / 'inputs.npz') == manifest['source_inputs_sha256']
            fields = np.load(data / 'displacement_coefficients.npy', mmap_mode='r')
            journals = [load(data / f'step_journal/{i:06d}.json') for i in (step-2, step-1)]
            for i, row in zip((step-2, step-1), journals):
                assert row['record'] == manifest['steps'][i]
                assert hashlib.sha256(fields[i].tobytes()).hexdigest() == row['coefficient_sha256']
            assert journals[0]['accepted_checkpoint_sha256'] == journals[1]['input_checkpoint_sha256']
            assert journals[1]['accepted_checkpoint_sha256'] == sha(data / 'restart.npz')
            with np.load(data / 'restart.npz') as cp:
                assert int(cp['completed_steps']) == step
                for name in ('u', 'u_old'):
                    np.testing.assert_array_equal(cp[name], fields[step-1])
                history = {name: cp[name] for name in ('u_old', 'v_old', 'a_old')}
            with np.load(data / 'inputs.npz') as arrays:
                pressure, time_s = float(arrays['pressure'][step]), float(arrays['time'][step])
            return history, fields[step-2].copy(), pressure, time_s

        history, previous, pressure, time_s = comm.bcast(on_root(load_checkpoint), root=0)
        adapter = ParallelP2(P, stream['source_reference'], cfg)
        problem = adapter.problem
        for name, values in history.items():
            adapter.assign(name, values)
        problem.parameters['p'].assign(pressure)
        problem.solver._solver.parameters.update(dict(method='newtonls', maximum_iterations=100,
                                                       absolute_tolerance=1e-8, relative_tolerance=0.))
        problem.solver._snes.setUseEW(False)
        arrays = on_root(lambda: np.load(data / 'inputs.npz'))
        audit = on_root(lambda: CofactorBernsteinAudit(arrays, workers=8))
        old, dt = history['u_old'], cfg['dt_s']
        alpha = float(problem.parameters['alpha_f'])
        assert alpha == .4
        if plan.get('saved_iterate'):
            def load_iterate():
                field = np.load(P / plan['saved_iterate'], allow_pickle=False)
                assert field.shape == old.shape and np.isfinite(field).all()
                return field

            saved = comm.bcast(on_root(load_iterate), root=0)
            adapter.assign('u', saved)
            saved_force = float(residual(problem).norm('l2'))
            np.testing.assert_allclose(saved_force, plan['saved_iterate_force_N'], rtol=1e-5, atol=1e-12)
            trace.update(time_s=time_s, saved_iterate=plan['saved_iterate'])
        else:
            guesses = [old, old + dt*history['v_old'], old + dt*history['v_old'] + .5*dt**2*history['a_old'],
                       (alpha*previous + (1-alpha)*old - alpha*old)/(1-alpha)]
            forces = []
            for guess in guesses:
                adapter.assign('u', guess)
                forces.append(float(residual(problem).norm('l2')))
            for choice in sorted((i for i in range(4) if np.isfinite(forces[i])), key=forces.__getitem__):
                def predictor_error():
                    try:
                        audit.pair(guesses[choice], old)
                    except Exception as exc:
                        return str(exc)
                    return None
                error = comm.bcast(on_root(predictor_error), root=0)
                if error is None:
                    adapter.assign('u', guesses[choice])
                    break
            else:
                raise RuntimeError('No whole-element-admissible initializer')
            trace.update(time_s=time_s, selected_predictor=choice)

        timings = {}
        trace['regularization_policy'] = plan.get('regularization_policy', 'band_half')

        def persist():
            if rank == 0:
                trace['phase_seconds_rank0'] = timings.copy()
                atomic(run / 'trace.json', trace)

        persist()
        solve_started = time.monotonic()
        local_error = None
        comparison = plan.get('compare_shifts', False)
        try:
            if not plan.get('saved_iterate'):
                solve(problem, adapter, history, audit,
                      dict(stream['admissible_newton'], wall_budget_seconds=1800), trace, persist,
                      policy=trace['regularization_policy'], timings=timings,
                      reuse_jacobian=plan.get('reuse_jacobian', False), stop_after=28 if comparison else None)
        except Exception as exc:
            cause = exc.__cause__
            local_error = f'{type(exc).__name__}: {exc}' + (f' caused by {cause}' if cause else '')
        errors = comm.allgather(local_error)
        on_root(persist)
        if any(errors):
            raise RuntimeError(next(error for error in errors if error))
        if comparison:
            common = comm.bcast(adapter.gather('u'), root=0)
            common_force = float(residual(problem).norm('l2'))
            def check_common():
                assert plan.get('saved_iterate') or len(trace['custom_newton_steps']) == 28

            on_root(check_common)
            assert common_force > 1e-9
            assert np.isfinite(common_force)

            def save_common():
                path = run / 'comparison-iterate.npy'
                np.save(path, common, allow_pickle=False)
                artifact(P, run, path, 'Shared Newton iterate for shift comparison')
            on_root(save_common)
            result['common_force_N'] = common_force
            result['comparison_complete'] = False
            result['shift_comparison'] = []
            for shift in (.0078125, .015625, .03125):
                adapter.assign('u', common)
                trial_started = time.monotonic()
                remaining = comm.allreduce(1800 - (trial_started - solve_started), op=MPI.MIN)
                if remaining <= 0:
                    raise RuntimeError('Shared 1800 s solver budget exhausted before all shifts were tested')
                trial = {}
                solve(problem, adapter, history, audit,
                      dict(stream['admissible_newton'], initial_regularization=shift,
                           wall_budget_seconds=remaining), trial, lambda: None,
                      policy=trace['regularization_policy'], stop_after=1)
                seconds = comm.allreduce(time.monotonic() - trial_started, op=MPI.MAX)
                force_after = float(residual(problem).norm('l2'))
                if rank == 0:
                    row = trial['custom_newton_steps'][0]
                    accepted = row['accepted_custom_update'] if 'accepted_custom_update' in row else False
                    result['shift_comparison'].append(dict(
                        shift=shift, seconds=seconds, accepted=accepted, force_N=force_after,
                        reduction_N_per_second=(common_force-force_after)/seconds if accepted else None,
                        direction=row))
                on_root(lambda: atomic(run / 'comparison-progress.json', result))
            result['comparison_complete'] = True

            def register_comparison():
                path = run / 'comparison-progress.json'
                atomic(path, result)
                artifact(P, run, path, 'Measured shift comparison')

            on_root(register_comparison)
        else:
            before = adapter.gather('u')
            problem.solver.solve()
            field = adapter.gather('u')
            force = float(residual(problem).norm('l2'))
            snes = problem.solver._snes
            assert 0 <= force < 1e-8 and snes.getConvergedReason() > 0 and snes.getIterationNumber() == 0
            assert not snes.getUseEW() and snes.getKSP().getType() == 'gmres'
            assert snes.getKSP().getPC().getType() == 'gamg'
            solve_seconds = comm.allreduce(time.monotonic() - solve_started, op=MPI.MAX)

            def verify_candidate():
                np.testing.assert_array_equal(before, field)
                cofactor = audit.pair(field, old)
                audit.close()
                legacy = ThreadedBernsteinAudit(arrays, workers=8)
                try:
                    proof = legacy.pair(field, old)
                finally:
                    legacy.close()
                same_proof(proof, cofactor)
                atomic(run / 'proof.json', proof)
                path = run / 'candidate.npz'
                np.savez_compressed(path, accepted=False, step=step, time_s=time_s, u=field, **history)
                for output in (path, run / 'proof.json', run / 'trace.json'):
                    artifact(P, run, output, 'Recovery probe output, no reference admission')

            on_root(verify_candidate)
            result.update(passed=True, native_force_N=force, SNES_reason=int(snes.getConvergedReason()),
                          native_updates=0, solve_seconds=solve_seconds, proof_agreement=True)
    except Exception as exc:
        result['stop_reason'] = f'{type(exc).__name__}: {exc}'
        if rank == 0:
            traceback.print_exc()
    if rank == 0:
        accepted = [row['trials'][-1] for row in trace['custom_newton_steps']
                    if row.get('accepted_custom_update')]
        result.update(regularization_policy=trace.get('regularization_policy'),
                      phase_seconds_rank0=trace.get('phase_seconds_rank0', {}),
                      reused_jacobian_updates=sum(row.get('jacobian_reused', False)
                                                  for row in trace['custom_newton_steps']
                                                  if row.get('accepted_custom_update')),
                      jacobian_reuse_rejections=len(trace.get('jacobian_reuse_rejections', [])),
                      accepted_custom_iterations=len(accepted),
                      last_accepted_custom_force_N=accepted[-1]['native_force_N'] if accepted else None)
        result.update(complete=True, wall_seconds=time.monotonic()-started,
                      cumulative_seconds=plan['cumulative_seconds_before']+time.monotonic()-started)
        atomic(run / 'summary.json', result)
        status.update(status='diagnostic_complete', cumulative_seconds=result['cumulative_seconds'],
                      blockers=[] if result['passed'] else [result['stop_reason']])
        atomic(run / 'state.json', status)
        artifact(P, run, run / 'summary.json', 'State-551 recovery probe result')
        event(run, 'job_finished', status=status['status'], pbs_job_id=os.environ['PBS_JOBID'])
    comm.Barrier()
    if lock is not None:
        lock.close()
    return 0 if result['passed'] or result.get('comparison_complete') else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['submit', 'execute'])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--parent-run', default='cmbbe-cofactor-ref-03')
    parser.add_argument('--previous-recovery')
    parser.add_argument('--compare-shifts', action='store_true')
    parser.add_argument('--queue')
    parser.add_argument('--resume-comparison')
    args = parser.parse_args()
    for name in filter(None, (args.run_id, args.parent_run, args.previous_recovery, args.resume_comparison)):
        assert re.fullmatch('[A-Za-z0-9_-]+', name), 'Invalid run ID'
    directory = P / 'experiments/hpc'
    if args.action == 'submit':
        with (P / '.hpc/profile-submission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            submit(directory / args.run_id, directory / args.parent_run,
                   directory / args.previous_recovery if args.previous_recovery else None,
                   compare_shifts=args.compare_shifts, queue=args.queue,
                   resume=directory / args.resume_comparison if args.resume_comparison else None)
    else:
        raise SystemExit(execute(directory / args.run_id))
