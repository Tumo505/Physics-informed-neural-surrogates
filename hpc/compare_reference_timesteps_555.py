"""Compare half and quarter time steps from the independently replayed state-550 histories."""
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


def pressure_reproduction(reconstructed, stored):
    """Validate the local experiment interval and retain full-cycle diagnostics."""
    import numpy as np
    reconstructed, stored = np.asarray(reconstructed), np.asarray(stored)
    assert reconstructed.shape == stored.shape and len(stored) >= 555
    assert np.isfinite(reconstructed).all() and np.isfinite(stored).all()
    local = slice(549, 555)
    np.testing.assert_allclose(reconstructed[local], stored[local], rtol=1e-10, atol=1e-8)
    difference = abs(reconstructed-stored)
    mismatches = np.flatnonzero(~np.isclose(reconstructed, stored, rtol=1e-10, atol=1e-8))
    return dict(checked_states=list(range(550, 556)), rtol=1e-10, atol_Pa=1e-8,
                local_max_abs_Pa=float(difference[local].max()),
                full_cycle_max_abs_Pa=float(difference.max()),
                full_cycle_mismatch_count=int(len(mismatches)),
                first_full_cycle_mismatch_state=int(mismatches[0]+1) if len(mismatches) else None)


def accepted_prefix(parent, factor, *, completed=False):
    """Recover only the hash-verified, committed refined checkpoint chain."""
    result = load(parent / 'result.json')
    assert result['complete']
    summary = result['outcomes'][str(factor)]['completed']
    assert summary and summary['complete'] == completed
    assert summary['factor'] == factor and summary['start_state'] == 550 and summary['stop_state'] == 555
    rows = summary['records']
    assert 0 < len(rows) == summary['completed_substeps'] <= 5*factor
    assert (len(rows) == 5*factor) == completed
    schedule = load(parent / 'pressure.json')['factors'][str(factor)]
    digest = sha(parent / 'seed-550.npz')
    for index, row in enumerate(rows, 1):
        step = parent / f'factor-{factor}/substep-{index:03d}'
        assert row['substep'] == index and row['input_checkpoint_sha256'] == digest
        assert row == load(step / 'accepted.json')
        assert sha(step / 'checkpoint.npz') == row['checkpoint_sha256']
        assert sha(step / 'trace.json') == row['trace_sha256']
        assert row['time_s'] == schedule['endpoint_times'][index-1]
        assert row['pressure_Pa'] == schedule['pressures'][index-1]
        assert 0 <= row['custom_force_N'] < 1e-9 and 0 <= row['native_force_N'] < 1e-8
        assert row['native_updates'] == 0
        digest = row['checkpoint_sha256']
    return rows


def restored_checkpoint(path, row, factor):
    import numpy as np
    assert sha(path) == row['checkpoint_sha256']
    with np.load(path, allow_pickle=False) as saved:
        assert int(saved['completed_substeps']) == row['substep'] and int(saved['factor']) == factor
        assert float(saved['time_s']) == row['time_s']
        values = {name: saved[name] for name in ('u', 'u_old', 'v_old', 'a_old', 'previous')}
    np.testing.assert_array_equal(values['u'], values['u_old'])
    for value in values.values():
        assert value.ndim == 1 and value.shape == values['u'].shape and np.isfinite(value).all()
    return values


def submit(run, resume_after=None, factor=None):
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
        parent = None
        rows = []
        factors = [2, 4]
        if resume_after is not None:
            assert factor in (2, 4), '--factor is required with --resume-after'
            parent = ROOT / 'experiments/hpc' / resume_after
            for filename in ('plan.json', 'result.json', 'seed-550.npz', 'patch.npz', 'pressure.json'):
                registered(ROOT, parent, parent / filename)
            parent_plan = load(parent / 'plan.json')
            assert parent_plan['source_checkpoint_sha256'] == source['checkpoint_sha256']
            assert parent_plan['protocol'] == source['protocol']
            assert parent_plan['gpu_library_sha256'] == sha(GPU / 'lib/libpetsc.so')
            rows = accepted_prefix(parent, factor)
            factors = [factor]
        else:
            assert factor is None, '--factor for submit requires --resume-after'
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
        resume = None
        if parent is not None:
            for filename in ('seed-550.npz', 'patch.npz', 'pressure.json'):
                shutil.copyfile(parent / filename, run / filename)
                assert sha(run / filename) == sha(parent / filename)
                artifact(ROOT, run, run / filename, 'Verified refined resume input')
                frozen[str(run / filename)] = sha(run / filename)
            folder = run / f'factor-{factor}'
            folder.mkdir()
            for row in rows:
                relative = Path(f'factor-{factor}/substep-{row["substep"]:03d}')
                (run / relative).mkdir()
                for filename in ('accepted.json', 'trace.json', 'checkpoint.npz'):
                    path = run / relative / filename
                    shutil.copyfile(parent / relative / filename, path)
                    expected = {'checkpoint.npz': row['checkpoint_sha256'],
                                'trace.json': row['trace_sha256'],
                                'accepted.json': sha(parent / relative / filename)}[filename]
                    assert sha(path) == expected
                    artifact(ROOT, run, path, 'Verified accepted refined prefix')
                    frozen[str(path)] = sha(path)
            atomic(run / 'resume.json', dict(parent_run=parent.name, factor=factor, records=rows))
            artifact(ROOT, run, run / 'resume.json', 'Refined resume provenance')
            frozen[str(run / 'resume.json')] = sha(run / 'resume.json')
            frozen[str(parent / 'result.json')] = sha(parent / 'result.json')
            resume = dict(parent_run=parent.name, factor=factor, completed_substeps=len(rows))
        atomic(run / 'plan.json', dict(source_checkpoint_sha256=source['checkpoint_sha256'],
                                      protocol=source['protocol'], start_state=550, stop_state=555,
                                      factors=factors, resume=resume,
                                      gpu_library_sha256=sha(GPU / 'lib/libpetsc.so'),
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
            if not plan.get('resume'):
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
                folder.mkdir(exist_ok=bool(plan.get('resume')))
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
    reproduction = pressure_reproduction(reconstructed, original_pressures)
    output = dict(original_pressure_reproduction_max_abs=reproduction['full_cycle_max_abs_Pa'],
                  pressure_reproduction=reproduction,
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


class SliceDeadline(TimeoutError):
    pass


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
    production = plan.get('refined_production', False)
    requested = len(schedule['endpoint_times'])
    assert requested == (plan['stop_state']-550)*factor
    deadline = float(os.environ.get('REFINED_DEADLINE', 'inf'))

    def check_deadline():
        if production:
            import time
            if comm.allreduce(time.time(), op=MPI.MAX) >= deadline:
                raise SliceDeadline('PBS slice checkpoint pause')
    for name in ('seed-550.npz', 'patch.npz', 'pressure.json'):
        if rank == 0:
            registered(ROOT, run, run / name)
    with np.load(run / 'seed-550.npz', allow_pickle=False) as seed:
        assert int(seed['completed_steps']) == 550
        histories = {name: seed[name] for name in ('u_old','v_old','a_old')}
        previous = seed['previous']
        for name in ('u', *histories):
            adapter.assign(name, seed[name])
    audit = patch = capture = path_audit = None
    replay = plan.get('positivity_replay')
    path_spec = plan.get('cell_path_audit')
    assert not (replay and path_spec)
    trace = {}
    if rank == 0:
        with np.load(TARGET / 'inputs.npz', allow_pickle=False) as inputs:
            audit = CofactorBernsteinAudit(inputs, workers=8)
        patch = np.load(run / 'patch.npz', allow_pickle=False)
        use_positivity_depth(audit, 20, 24)
        if plan.get('certified_depth_retry'):
            from refined_positivity_policy import extend_unresolved_depth
            extend_unresolved_depth(audit, plan['certified_depth_retry']['depth'])
            print('REFINED POSITIVITY maximum_depth=20 unresolved_retry_depths=24,32 margin=1e-9', flush=True)
        if plan.get('adaptive_interval_proof'):
            from refined_positivity_policy import install_adaptive_proof
            assert plan['certified_depth_retry']['depth'] == 32
            install_adaptive_proof(audit)
            print('REFINED POSITIVITY adaptive_interval_bernstein enabled margin=1e-9', flush=True)
        if replay:
            from replay_refined_positivity import TrialCapture
            capture = TrialCapture(audit, replay, lambda: trace)
        if path_spec:
            from audit_refined_cell_path import CellPath
            path_audit = CellPath(run, audit, path_spec, lambda: trace)
        observer = capture or path_audit
        add_positivity_diagnostics(audit, observer.on_failure if observer else None)
        if capture:
            capture.track_field()
        if path_audit:
            path_audit.track_field()
    records = []
    summary = dict(factor=factor, start_state=550, stop_state=plan['stop_state'], dt_s=cfg['dt_s'],
                   completed_substeps=0, requested_substeps=requested, complete=False, records=records,
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

    coupled_pool = None
    try:
        start = 0
        if production:
            from launch_refined_production import production_records
            # Each rank restores the same accepted prefix, including velocity and acceleration.
            records.extend(comm.bcast(on_root(lambda: production_records(run)), root=0))
            start = len(records)
        elif plan.get('resume'):
            resume = load(run / 'resume.json')
            assert resume['factor'] == factor == plan['resume']['factor']
            records.extend(resume['records'])
            start = len(records)
            assert start == plan['resume']['completed_substeps']
        if start:
            last = folder / f'substep-{start:03d}'
            assert records[-1]['time_s'] == schedule['endpoint_times'][start-1]
            saved = restored_checkpoint(last / 'checkpoint.npz', records[-1], factor)
            previous = saved['previous']
            histories = {name: saved[name] for name in histories}
            for name in ('u', *histories):
                adapter.assign(name, saved[name])
            on_root(lambda: audit.pair(histories['u_old'], previous))
            summary['completed_substeps'] = start
            on_root(lambda: atomic(folder / 'summary.json', summary))
            if rank == 0:
                print('REFINED RESUME factor', factor, 'accepted_substeps', start,
                      'next_substep', start+1, flush=True)
        if path_spec:
            on_root(lambda: path_audit.start(histories['u_old'], previous, records[-1]['native_force_N']))
        for index, (time_s, pressure) in enumerate(zip(schedule['endpoint_times'],schedule['pressures'],strict=True)):
            if index < start:
                continue
            check_deadline()
            step = folder / f'substep-{index+1:03d}'
            if replay:
                assert index+1 == replay['substep'], 'Replay must begin at the failed substep'
            if path_spec:
                assert index+1 == path_spec['substep'], 'Cell-path replay must begin at the failed substep'
            on_root(lambda: step.mkdir(exist_ok=production))
            trace_path = step / 'trace.json'
            if production:
                trace_path = step / f'trace-slice-{int(os.environ["REFINED_SLICE"]):03d}.json'
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
            if plan.get('certified_depth_retry'):
                trace.update(positivity_subdivision_depth_limit=20,
                             positivity_subdivision_retry_depth_limit=32, positivity_margin=1e-9)
            if plan.get('adaptive_interval_proof'):
                trace['positivity_fallback_algorithm'] = 'determinant_adaptive_interval_bernstein'
            if plan.get('constrained_direction'):
                trace['direction_policy'] = 'joint_determinant_merit_qp'
            if plan.get('coupled_newton'):
                assert not plan.get('constrained_direction')
                trace['direction_policy'] = 'assembled_jacobian_active_set'

            def persist():
                if rank == 0:
                    atomic(trace_path, trace)

            persist()
            if path_spec:
                on_root(lambda: path_audit.initializer(candidates[selected], norms[selected]))
            options = dict(protocol['admissible_newton'], wall_budget_seconds=float('inf'))
            guard = None
            coupled = None
            if plan.get('coupled_newton'):
                from coupled_newton_direction import build_constraints, GrowingConstraints
                spec = plan['coupled_newton']
                if spec.get('dynamic_elements'):
                    if coupled_pool is None:
                        seeds = set(spec['initial_elements'])
                        if records:
                            seeds.update(records[-1].get('coupled_elements', []))
                        coupled_pool = GrowingConstraints(audit, seeds, lambda: histories['u_old'],
                                                         spec['maximum_elements'])
                    coupled = coupled_pool
                else:
                    def coupled(current, raw):
                        return build_constraints(audit, spec['element'],
                                                 current, histories['u_old'], raw)
            if plan.get('constrained_direction'):
                from refined_constrained_direction import correct_direction

                def guard(current, direction, negative_merit):
                    corrected, detail = correct_direction(audit, plan['constrained_direction']['element'],
                        current, histories['u_old'], direction, negative_merit)
                    if detail['applied']:
                        print('CONSTRAINED DIRECTION iteration', len(trace.get('custom_newton_steps', [])),
                              'determinant', detail['minimum_point_determinant'],
                              'merit_slope', detail['corrected_merit_slope'],
                              'correction_norm', detail['correction_norm'], flush=True)
                    return corrected, detail

            def save_cell(iteration, candidate, force, shift):
                if rank == 0:
                    path_audit.accepted(iteration, candidate, force, shift)

            stop = replay['iteration']+1 if replay else path_spec['directions'] if path_spec else None
            diagnostic_stop = None
            try:
                solve(problem, adapter, histories, audit, options, trace, persist,
                      policy='two_good_steps', cuda_linear=True, host_preconditioner=True,
                      directional_armijo=True, late_forcing_floor=1e-5, stop_after=stop,
                      direction_guard=guard,
                      coupled_constraints=coupled,
                      save_iterate=save_cell if path_spec else None,
                      check_deadline=check_deadline if production else None)
            except RuntimeError as exc:
                if not path_spec or str(exc) not in (
                        'Regularization ceiling without admissible descent',
                        'Regularized Newton iteration ceiling reached'):
                    raise
                diagnostic_stop = str(exc)
            if path_spec:
                persist()
                if diagnostic_stop is None and trace.get('custom_force_N', float('inf')) < 1e-9:
                    diagnostic_stop = 'custom_force_target_reached_during_replay'
                on_root(lambda: path_audit.finish(trace, diagnostic_stop))
                return
            if replay:
                persist()
                from replay_refined_positivity import finish_replay
                on_root(lambda: finish_replay(run, replay, trace, capture))
                return
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
                           trace_sha256=sha(trace_path), trace_file=trace_path.name,
                           input_checkpoint_sha256=sha(run / 'seed-550.npz') if index==0 else records[-1]['checkpoint_sha256'],
                           velocity_max_abs=float(np.max(abs(checkpoint['v_old']))),
                           acceleration_max_abs=float(np.max(abs(checkpoint['a_old']))),
                           patch=patch_metrics(patch, candidate, old, forcing['material_parameters']),
                           whole_element_proof=proof)
                if trace.get('coupled_elements'):
                    row['coupled_elements'] = trace['coupled_elements']
                atomic(step / 'accepted.json', row)
                if production:
                    artifact(ROOT, run, step / 'accepted.json', 'Accepted refined production substep')
                    from launch_refined_production import compact_record
                    records.append(compact_record(row))
                else:
                    records.append(row)
                summary['completed_substeps'] = index+1
                atomic(folder / 'summary.json', summary)
                print('REFINED ACCEPTED', 'factor',factor,'substep',index+1,'/',requested,
                      'time',time_s,'force_N',force,flush=True)
            on_root(accept)
            previous = old
            histories = comm.bcast({name: checkpoint[name] for name in histories} if rank==0 else None,root=0)
        summary['complete'] = True
        on_root(lambda: atomic(folder / 'summary.json', summary))
    except SliceDeadline:
        summary['paused_for_walltime'] = True
        on_root(lambda: atomic(folder / 'summary.json', summary))
        if rank == 0:
            print('REFINED SLICE PAUSED accepted_substeps', summary['completed_substeps'], flush=True)
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
    requested_complete = all(
        outcome['completed'] and outcome['completed']['complete'] for outcome in outcomes.values())
    result = dict(complete=True, both_factors_completed=set(outcomes) == {'2', '4'} and requested_complete,
                  requested_factors_completed=requested_complete,
                  start_state=550, stop_state=555, source_accepted_states=554, outcomes=outcomes,
                  original_pressure_reproduction_max_abs=pressure['original_pressure_reproduction_max_abs'],
                  pressure_reproduction=pressure['pressure_reproduction'],
                  samples=rows, scope='Local time-refinement comparison inheriting the state-550 histories. '
                  'This interval does not establish full-cycle time convergence. Source reference remains separate.')
    atomic(run / 'result.json', result)
    print('TIME REFINEMENT REQUESTED FACTORS COMPLETE', requested_complete, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('submit','execute','prepare','solve'))
    parser.add_argument('--run-id',default='cmbbe-timestep-550-555-01')
    parser.add_argument('--factor',type=int,choices=(2,4))
    parser.add_argument('--resume-after', help='Completed comparison containing accepted refined substeps')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):
        parser.error('Invalid run ID')
    if args.resume_after and not re.fullmatch(r'[A-Za-z0-9_-]+',args.resume_after):
        parser.error('Invalid resume run ID')
    if args.resume_after and args.action != 'submit':
        parser.error('--resume-after is only supported for submit')
    run = ROOT / 'experiments/hpc' / args.run_id
    if args.action == 'submit':
        submit(run, args.resume_after, args.factor)
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
