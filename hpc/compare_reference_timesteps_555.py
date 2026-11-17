"""Compare half and quarter time steps from the independently replayed state-550 histories."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'hpc'), str(ROOT / 'src')]
from registry import artifact, atomic, event, sha
from submit_probe_profile import assert_idle, registered

TARGET = ROOT / 'data/processed/cardiac_hpc_cmbbe-cofactor-ref-03'
AUDIT = ROOT / 'experiments/hpc/cmbbe-neighborhood-audit-555-01'
SOURCE = ROOT / 'experiments/hpc/cmbbe-residual-audit-555-01'
NATIVE = ROOT / '.hpc/native-v2'
GPU = Path.home() / '.local/petsc-3.22.3-cuda-sm52-m60'
CUDA = Path.home() / '.local/cuda-11.4.2'


def load(path):
    return json.loads(path.read_text())


def save_npz(path, **values):
    import numpy as np
    pending = path.with_suffix('.pending.npz')
    with pending.open('wb') as output:
        np.savez(output, **values)
        output.flush()
        os.fsync(output.fileno())
    pending.replace(path)


def sampling_schedule(base_dt, factor):
    import numpy as np
    assert factor in (2, 4)
    dt = base_dt / factor
    end_times = 550 * base_dt + np.arange(1, 5 * factor + 1) * dt
    return dt, end_times, end_times - .4 * dt


def submit(run):
    with (ROOT / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(ROOT)
        for path, parent in ((AUDIT / 'result.json', AUDIT), (SOURCE / 'plan.json', SOURCE)):
            registered(ROOT, parent, path)
        evidence = load(AUDIT / 'result.json')
        assert evidence['complete'] and evidence['accepted_states'] == 554
        source = load(SOURCE / 'plan.json')
        manifest = load(TARGET / 'manifest.json')
        assert manifest['completed_steps'] == 554
        assert sha(TARGET / 'restart.npz') == source['checkpoint_sha256']
        frozen = dict(source['frozen'])
        files = [Path(__file__), AUDIT / 'result.json',
                 ROOT / 'hpc/production_cuda_native.py', ROOT / 'hpc/cofactor_proof.py',
                 ROOT / 'hpc/audit_reference_neighborhood_555.py',
                 ROOT / 'hpc/audit_reference_cell_555.py',
                 ROOT / 'src/lv_surrogate/physics/admissible_newton_gentle_shrink.py',
                 ROOT / 'src/lv_surrogate/physics/regularization_recovery_policy.py',
                 ROOT / 'external/cardiac_benchmark/src/cardiac_benchmark/pressure_model.py',
                 ROOT / 'external/cardiac_benchmark/src/cardiac_benchmark/material.py']
        # The solver has advanced since the earlier assembly audit. Pin its current version.
        for path in files:
            frozen[str(path)] = sha(path)
        for path, digest in frozen.items():
            assert sha(Path(path)) == digest, path
        run.mkdir()
        atomic(run / 'plan.json', dict(source_checkpoint_sha256=source['checkpoint_sha256'],
                                      protocol=source['protocol'], start_state=550, stop_state=555,
                                      factors=[2, 4], gpu_library_sha256=sha(GPU / 'lib/libpetsc.so'),
                                      frozen=frozen))
        artifact(ROOT, run, run / 'plan.json', 'Separate time-refinement comparison plan')
        command = [str(ROOT / '.venv/bin/python'), str(Path(__file__).resolve()),
                   'execute', '--run-id', run.name]
        script = run / 'job.pbs'
        script.write_text('#!/bin/bash\nset -eu\ncd ' + shlex.quote(str(ROOT)) +
                          '\nexec ' + shlex.join(command) + '\n')
        artifact(ROOT, run, script, 'PBS time-refinement comparison')
        atomic(run / 'state.json', dict(status='submitted', accepted_reference_steps=554))
        result = subprocess.run(
            ['qsub', '-N', 'lv_dt_compare', '-q', 'iworkq', '-l',
             'select=1:ncpus=28:mem=240gb:ngpus=4', '-l', 'place=excl',
             '-l', 'software=PBSJobApp', '-l', 'walltime=168:00:00',
             '-j', 'oe', '-o', str(run / 'pbs.log'), str(script)], text=True, capture_output=True)
        event(run, 'job_submission', return_code=result.returncode,
              stdout=result.stdout.strip(), stderr=result.stderr.strip())
        result.check_returncode()
        print('Time-refinement comparison job:', result.stdout.strip())


def execute(run):
    assert os.environ.get('PBS_JOBID')
    plan = load(run / 'plan.json')
    state = dict(status='running', job=os.environ['PBS_JOBID'], accepted_reference_steps=554)
    with (run / 'execution.lock').open('a') as execution, (TARGET / 'generation.lock').open('a') as generation:
        for lock in (execution, generation):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            registered(ROOT, run, run / 'plan.json')
            for path, digest in plan['frozen'].items():
                assert sha(Path(path)) == digest, path
            assert sha(GPU / 'lib/libpetsc.so') == plan['gpu_library_sha256']
            assert len(set(Path(os.environ['PBS_NODEFILE']).read_text().splitlines())) == 1
            atomic(run / 'state.json', state)
            env = os.environ.copy()
            for key in ('LD_PRELOAD', 'PETSC_DIR', 'PETSC_ARCH', 'PETSC_OPTIONS', 'PKG_CONFIG_SYSROOT_DIR'):
                env.pop(key, None)
            pkg = f'{NATIVE}/lib/pkgconfig:{NATIVE}/share/pkgconfig'
            env.update(PATH=f'{NATIVE}/bin:{env["PATH"]}', LD_LIBRARY_PATH=f'{NATIVE}/lib:{CUDA}/lib64',
                       PKG_CONFIG=str(NATIVE / 'bin/pkg-config'), PKG_CONFIG_PATH=pkg, PKG_CONFIG_LIBDIR=pkg,
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       HYDRA_LAUNCHER='fork', FI_PROVIDER='tcp',
                       DIJITSO_CACHE_DIR=str(ROOT / '.hpc/fenics-cache'),
                       XDG_CACHE_HOME=str(ROOT / '.hpc/cache'))
            base = [str(NATIVE / 'bin/python'), str(Path(__file__).resolve())]
            with (run / 'prepare.log').open('x') as log:
                subprocess.run([*base, 'prepare', '--run-id', run.name], env=dict(env, CUDA_VISIBLE_DEVICES=''),
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            for filename in ('seed-550.npz', 'patch.npz', 'pressure.json'):
                artifact(ROOT, run, run / filename, 'Time-refinement initial data')
            visible = env.get('CUDA_VISIBLE_DEVICES')
            if not visible:
                visible = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                    text=True).strip().replace('\n', ',')
            assert len(set(visible.split(','))) == 4
            gpu_env = dict(env, CUDA_VISIBLE_DEVICES=visible, PETSC_OPTIONS='-use_gpu_aware_mpi 0',
                           PKG_CONFIG_PATH=f'{GPU}/lib/pkgconfig:{pkg}',
                           PKG_CONFIG_LIBDIR=f'{GPU}/lib/pkgconfig:{pkg}')
            outcomes = {}
            for factor in plan['factors']:
                folder = run / f'factor-{factor}'
                folder.mkdir()
                state['phase'] = f'factor_{factor}'
                atomic(run / 'state.json', state)
                command = [str(NATIVE / 'bin/mpiexec'), '-n', '12', '/usr/bin/env',
                           f'LD_PRELOAD={GPU}/lib/libpetsc.so', *base, 'solve',
                           '--run-id', run.name, '--factor', str(factor)]
                with (folder / 'command.log').open('x') as log:
                    reply = subprocess.run(command, env=gpu_env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                outcomes[str(factor)] = dict(exit_code=reply.returncode,
                                            completed=load(folder / 'summary.json') if (folder / 'summary.json').exists() else None)
            compare(run, outcomes)
            assert sha(TARGET / 'restart.npz') == plan['source_checkpoint_sha256']
            assert sha(TARGET / 'manifest.json') == plan['frozen'][str(TARGET / 'manifest.json')]
            artifact(ROOT, run, run / 'result.json', 'Time-refinement comparison result')
            state.update(status='comparison_complete', phase='finished', outcomes=outcomes)
        except Exception as exc:
            state.update(status='blocked', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            atomic(run / 'state.json', state)


def prepare(run):
    import numpy as np
    from scipy.spatial import cKDTree
    from lv_surrogate.physics.parallel_fem import ParallelP2
    from lv_surrogate.physics.reference_fem import residual, pressure_function, default_parameters_benchmark1

    plan = load(run / 'plan.json')
    protocol = load(Path(plan['protocol']))
    manifest = load(TARGET / 'manifest.json')
    cfg = manifest['config']
    adapter = ParallelP2(ROOT, protocol['source_reference'], cfg)
    assert adapter.comm.size == 1
    problem = adapter.problem
    fields = np.load(TARGET / 'displacement_coefficients.npy', mmap_mode='r')
    for name in ('u', 'u_old', 'v_old', 'a_old'):
        getattr(problem, name).vector().zero()
    with np.load(TARGET / 'inputs.npz', allow_pickle=False) as inputs:
        original_times, original_pressures = inputs['time'], inputs['pressure']
        pressures = original_pressures[:550]
        selected = np.array(load(AUDIT / 'result.json')['cells'])
        vertices = inputs['coords'][inputs['cells'][selected]]
        maps = inputs['canonical_cell_dofs'][selected]
        quadrature = inputs['qpoints']
    previous_digest = None
    for index in range(550):
        if index >= 548:
            journal = load(TARGET / f'step_journal/{index:06d}.json')
            assert journal['record'] == manifest['steps'][index]
            assert hashlib.sha256(np.asarray(fields[index]).tobytes()).hexdigest() == journal['coefficient_sha256']
            if previous_digest is not None:
                assert journal['input_checkpoint_sha256'] == previous_digest
            previous_digest = journal['accepted_checkpoint_sha256']
        adapter.assign('u', fields[index])
        if index == 549:
            journal = load(TARGET / 'step_journal/000549.json')
            assert hashlib.sha256(np.asarray(fields[index]).tobytes()).hexdigest() == journal['coefficient_sha256']
            problem.parameters['p'].assign(float(pressures[index]))
            force = float(residual(problem).norm('l2'))
            assert force < 1e-8, force
        problem._update_fields()
        if (index+1) % 64 == 0:
            print('SEED HISTORY REPLAY', index+1, '/', 550, flush=True)
    values = {name: adapter.gather(name) for name in ('u', 'u_old', 'v_old', 'a_old')}
    np.testing.assert_array_equal(values['u_old'], fields[549])
    save_npz(run / 'seed-550.npz', **values, previous=fields[548], completed_steps=550)
    native_vertices = adapter.geo.mesh.coordinates()
    distance, indices = cKDTree(native_vertices).query(vertices.reshape(-1,3))
    assert distance.max() < 1e-12
    count = len(native_vertices)
    fibers = {
        name: getattr(adapter.geo, name).compute_vertex_values(adapter.geo.mesh).reshape(3,count).T[
            indices].reshape(-1,4,3) for name in ('f0','s0')}
    save_npz(run / 'patch.npz', vertices=vertices, maps=maps, cells=selected,
             quadrature=quadrature, **fibers)
    parameters = default_parameters_benchmark1()
    for key in ('t_sys_pre', 't_dias_pre', 'gamma'):
        parameters[key] *= cfg['timing_scale']
    schedules = {str(factor): sampling_schedule(cfg['dt_s'], factor) for factor in plan['factors']}
    combined = np.unique(np.concatenate([original_times - .4*cfg['dt_s'],
                                         *[item[2] for item in schedules.values()]]))
    pressure = cfg['amplitude'] * pressure_function((0, cfg['duration_s']), parameters=parameters, t_eval=combined)
    reconstructed = pressure[np.searchsorted(combined, original_times - .4*cfg['dt_s'])]
    np.testing.assert_allclose(reconstructed, original_pressures, rtol=1e-10, atol=1e-8)
    output = dict(original_pressure_reproduction_max_abs=float(np.max(abs(reconstructed-original_pressures))),
                  source_force_at_state_550_N=force, material_parameters={
                      key: float(value) for key,value in problem.material.parameters.items()}, factors={})
    for factor, (dt, times, stage_times) in schedules.items():
        output['factors'][factor] = dict(dt_s=dt, endpoint_times=times.tolist(),
                                         pressure_times=stage_times.tolist(),
                                         pressures=pressure[np.searchsorted(combined, stage_times)].tolist())
    atomic(run / 'pressure.json', output)
    print('STATE 550 SEED AND PRESSURE SCHEDULES READY', flush=True)


def patch_metrics(patch, current, old, parameters):
    import numpy as np
    from audit_reference_cell_555 import cartesian_deformation
    from audit_reference_neighborhood_555 import point_metrics

    xyz = patch['quadrature']
    barycentric = np.vstack((np.column_stack((1-xyz.sum(axis=1), xyz)), np.eye(4),
                             np.array([[.5,.5,0,0],[.5,0,.5,0],[.5,0,0,.5],
                                       [0,.5,.5,0],[0,.5,0,.5],[0,0,.5,.5]])))
    output = {}
    for stage, values in (('endpoint', current), ('stress', .4*old+.6*current)):
        rows = []
        for index, cell in enumerate(patch['cells']):
            vertices = patch['vertices'][index]
            deformation = cartesian_deformation(vertices, values[patch['maps'][index]], barycentric @ vertices)
            metrics = point_metrics(deformation, barycentric @ patch['f0'][index],
                                    barycentric @ patch['s0'][index], parameters)
            row = dict(cell=int(cell), sampled_min_determinant=float(metrics['determinant'].min()))
            for key in ('isotropic_exponent','fiber_exponent','sheet_exponent','coupling_exponent'):
                finite = metrics[key][np.isfinite(metrics[key])]
                row[key] = float(finite.max()) if len(finite) else None
            rows.append(row)
        output[stage] = rows
    return output


def solve_factor(run, factor):
    import numpy as np
    from mpi4py import MPI
    from lv_surrogate.physics.parallel_fem import ParallelP2
    from lv_surrogate.physics.reference_fem import residual
    from lv_surrogate.physics.admissible_newton_gentle_shrink import solve
    from cofactor_proof import CofactorBernsteinAudit
    from production_cuda_native import use_positivity_depth, add_positivity_diagnostics

    plan = load(run / 'plan.json')
    cfg = dict(load(TARGET / 'manifest.json')['config'])
    protocol = load(Path(plan['protocol']))
    forcing = load(run / 'pressure.json')
    schedule = forcing['factors'][str(factor)]
    cfg['dt_s'] = schedule['dt_s']
    adapter = ParallelP2(ROOT, protocol['source_reference'], cfg)
    problem, comm, rank = adapter.problem, adapter.comm, adapter.rank
    loaded = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
              if '/libpetsc.so' in line}
    assert loaded == {(GPU / 'lib/libpetsc.so').resolve()}
    assert comm.size == 12 and float(problem.parameters['dt']) == cfg['dt_s']
    assert float(problem.parameters['alpha_f']) == .4
    problem.solver._snes.setUseEW(False)
    problem.solver._solver.parameters.update(dict(method='newtonls', maximum_iterations=100,
                                                  absolute_tolerance=1e-8, relative_tolerance=0.))
    folder = run / f'factor-{factor}'
    for name in ('seed-550.npz', 'patch.npz', 'pressure.json'):
        if rank == 0:
            registered(ROOT, run, run / name)
    with np.load(run / 'seed-550.npz', allow_pickle=False) as seed:
        assert int(seed['completed_steps']) == 550
        histories = {name: seed[name] for name in ('u_old','v_old','a_old')}
        previous = seed['previous']
        for name in ('u', *histories):
            adapter.assign(name, seed[name])
    audit = patch = None
    if rank == 0:
        with np.load(TARGET / 'inputs.npz', allow_pickle=False) as inputs:
            audit = CofactorBernsteinAudit(inputs, workers=8)
        patch = np.load(run / 'patch.npz', allow_pickle=False)
        use_positivity_depth(audit, 20, 24)
        add_positivity_diagnostics(audit)
    records = []
    summary = dict(factor=factor, start_state=550, stop_state=555, dt_s=cfg['dt_s'],
                   completed_substeps=0, requested_substeps=5*factor, complete=False, records=records,
                   reference_state_accepted=False)

    def on_root(function):
        value = error = None
        if rank == 0:
            try:
                value = function()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        error = comm.bcast(error, root=0)
        if error:
            raise RuntimeError(error)
        return value

    def proof_pair(candidate):
        return on_root(lambda: audit.pair(candidate, histories['u_old']))

    try:
        for index, (time_s, pressure) in enumerate(zip(schedule['endpoint_times'],schedule['pressures'],strict=True)):
            step = folder / f'substep-{index+1:03d}'
            on_root(lambda: step.mkdir())
            problem.parameters['p'].assign(pressure)
            old, velocity, acceleration = (histories[name] for name in ('u_old','v_old','a_old'))
            dt = cfg['dt_s']
            candidates = [old, old+dt*velocity, old+dt*velocity+.5*dt**2*acceleration,
                          old + (.4/.6)*(previous-old)]
            norms = []
            for candidate in candidates:
                adapter.assign('u', candidate)
                force = float(residual(problem).norm('l2'))
                norms.append(force)
            selected = None
            for choice in sorted(range(len(norms)), key=lambda i: norms[i] if np.isfinite(norms[i]) else float('inf')):
                if not np.isfinite(norms[choice]):
                    continue
                adapter.assign('u', candidates[choice])
                error = None
                if rank == 0:
                    try:
                        audit.pair(candidates[choice], old)
                    except ValueError as exc:
                        error = str(exc)
                error = comm.bcast(error, root=0)
                if error is None:
                    selected = choice
                    break
            assert selected is not None, 'No admissible initializer for refined substep'
            trace = dict(factor=factor, substep=index+1, time_s=time_s, pressure_Pa=pressure,
                         selected_predictor=selected,
                         initial_forces_N=[value if np.isfinite(value) else None for value in norms])

            def persist():
                if rank == 0:
                    atomic(step / 'trace.json', trace)

            persist()
            options = dict(protocol['admissible_newton'], wall_budget_seconds=float('inf'))
            solve(problem, adapter, histories, audit, options, trace, persist,
                  policy='two_good_steps', cuda_linear=True, host_preconditioner=True,
                  directional_armijo=True, late_forcing_floor=1e-5)
            assert 0 <= trace['custom_force_N'] < 1e-9
            persist()
            before = adapter.gather('u')
            problem.solver.solve()
            candidate = adapter.gather('u')
            assert problem.solver._snes.getIterationNumber() == 0
            assert problem.solver._snes.getConvergedReason() > 0
            force = float(residual(problem).norm('l2'))
            assert np.isfinite(force) and force < 1e-8
            def verify_candidate():
                np.testing.assert_array_equal(before, candidate)
            on_root(verify_candidate)
            proof = proof_pair(candidate if rank == 0 else None)
            problem._update_fields()
            checkpoint = {name: adapter.gather(name) for name in ('u','u_old','v_old','a_old')}

            def accept():
                beta = float(problem._beta)
                gamma = float(problem._gamma)
                expected_a = (candidate-old-dt*velocity-(.5-beta)*dt**2*acceleration)/(beta*dt**2)
                expected_v = velocity+(1-gamma)*dt*acceleration+gamma*dt*expected_a
                np.testing.assert_allclose(checkpoint['a_old'], expected_a, rtol=1e-7, atol=1e-7)
                np.testing.assert_allclose(checkpoint['v_old'], expected_v, rtol=1e-7, atol=1e-10)
                np.testing.assert_array_equal(checkpoint['u_old'], candidate)
                save_npz(step / 'checkpoint.npz', **checkpoint, previous=old, time_s=time_s,
                         completed_substeps=index+1, factor=factor)
                row = dict(substep=index+1, time_s=time_s, pressure_time_s=schedule['pressure_times'][index],
                           pressure_Pa=pressure, native_force_N=force, custom_force_N=trace['custom_force_N'],
                           native_updates=0, custom_iterations=trace['custom_iterations'],
                           checkpoint_sha256=sha(step / 'checkpoint.npz'),
                           trace_sha256=sha(step / 'trace.json'),
                           input_checkpoint_sha256=sha(run / 'seed-550.npz') if index==0 else records[-1]['checkpoint_sha256'],
                           velocity_max_abs=float(np.max(abs(checkpoint['v_old']))),
                           acceleration_max_abs=float(np.max(abs(checkpoint['a_old']))),
                           patch=patch_metrics(patch, candidate, old, forcing['material_parameters']),
                           whole_element_proof=proof)
                atomic(step / 'accepted.json', row)
                records.append(row)
                summary['completed_substeps'] = index+1
                atomic(folder / 'summary.json', summary)
                print('REFINED ACCEPTED', 'factor',factor,'substep',index+1,'/',5*factor,
                      'time',time_s,'force_N',force,flush=True)
            on_root(accept)
            previous = old
            histories = comm.bcast({name: checkpoint[name] for name in histories} if rank==0 else None,root=0)
        summary['complete'] = True
        on_root(lambda: atomic(folder / 'summary.json', summary))
    except Exception as exc:
        if rank == 0:
            summary['failure'] = f'{type(exc).__name__}: {exc}'
            atomic(folder / 'summary.json', summary)
        raise
    finally:
        if rank == 0:
            audit.close()
            patch.close()


def compare(run, outcomes):
    import numpy as np
    plan = load(run / 'plan.json')
    fields = np.load(TARGET / 'displacement_coefficients.npy', mmap_mode='r')
    pressure = load(run / 'pressure.json')
    base_dt = load(TARGET / 'manifest.json')['config']['dt_s']
    rows = []
    for state in range(551, 556):
        samples = {}
        row = dict(original_sample_state=state, time_s=state*base_dt, factors={})
        for factor in plan['factors']:
            substep = (state-550)*factor
            folder = run / f'factor-{factor}/substep-{substep:03d}'
            if not (folder / 'accepted.json').exists():
                continue
            accepted = load(folder / 'accepted.json')
            path = folder / 'checkpoint.npz'
            assert sha(path) == accepted['checkpoint_sha256']
            assert abs(accepted['time_s']-state*base_dt) < 1e-12
            with np.load(path,allow_pickle=False) as saved:
                samples[factor] = {name:saved[name] for name in ('u','v_old','a_old')}
            comparison = dict(native_force_N=accepted['native_force_N'], patch=accepted['patch'])
            if state <= 554:
                difference = samples[factor]['u'] - fields[state-1]
                comparison['displacement_relative_l2_vs_original'] = float(
                    np.linalg.norm(difference)/max(np.linalg.norm(fields[state-1]),1e-30))
                comparison['displacement_max_abs_vs_original'] = float(np.max(abs(difference)))
            row['factors'][str(factor)] = comparison
        if set(samples) == {2,4}:
            row['half_vs_quarter'] = {}
            for name in ('u','v_old','a_old'):
                difference = samples[2][name]-samples[4][name]
                row['half_vs_quarter'][name] = dict(
                    relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(samples[4][name]),1e-30)),
                    max_abs=float(np.max(abs(difference))))
        rows.append(row)
    result = dict(complete=True, both_factors_completed=all(
                      outcome['completed'] and outcome['completed']['complete'] for outcome in outcomes.values()),
                  start_state=550, stop_state=555, source_accepted_states=554, outcomes=outcomes,
                  original_pressure_reproduction_max_abs=pressure['original_pressure_reproduction_max_abs'],
                  samples=rows, scope='Local time-refinement comparison inheriting the state-550 histories. '
                  'This interval does not establish full-cycle time convergence. Source reference remains separate.')
    atomic(run / 'result.json', result)
    print('TIME REFINEMENT COMPARISON COMPLETE', result['both_factors_completed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('submit','execute','prepare','solve'))
    parser.add_argument('--run-id',default='cmbbe-timestep-550-555-01')
    parser.add_argument('--factor',type=int,choices=(2,4))
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):
        parser.error('Invalid run ID')
    run = ROOT / 'experiments/hpc' / args.run_id
    if args.action == 'submit':
        submit(run)
    elif args.action == 'execute':
        execute(run)
    elif args.action == 'prepare':
        prepare(run)
    else:
        if args.factor is None:
            parser.error('--factor is required for solve')
        devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        assert len(set(devices))==4
        rank=int(os.environ['PMI_RANK'])
        os.environ['CUDA_VISIBLE_DEVICES']=devices[rank%4]
        print('TIME REFINEMENT GPU',rank,os.environ['CUDA_VISIBLE_DEVICES'],flush=True)
        try:
            solve_factor(run,args.factor)
        except Exception:
            import traceback
            traceback.print_exc()
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
