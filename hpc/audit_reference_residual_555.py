"""Compare state-555 MPI CUDA and independent serial CPU native residual assembly."""
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
BLENDS = ROOT / 'experiments/hpc/cmbbe-blends-555-01'
NATIVE = ROOT / '.hpc/native-v2'
GPU = Path.home() / '.local/petsc-3.22.3-cuda-sm52-m60'
CUDA = Path.home() / '.local/cuda-11.4.2'


def load(path):
    return json.loads(path.read_text())


def array_hash(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def submit(run):
    with (ROOT / '.hpc/profile-submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_idle(ROOT)
        for path in (BLENDS / 'plan.json', BLENDS / 'slice-001/blend-screen.json'):
            registered(ROOT, BLENDS, path)
        source = load(BLENDS / 'plan.json')
        report = load(BLENDS / 'slice-001/blend-screen.json')
        assert report['screen_complete'] and report['state'] == 555
        manifest = load(TARGET / 'manifest.json')
        assert manifest['completed_steps'] == 554
        restart = TARGET / 'restart.npz'
        assert sha(restart) == source['checkpoint_sha256']
        seed = BLENDS / 'diagnostic-seed.npz'
        registered(ROOT, BLENDS, seed)
        assert sha(seed) == source['positivity_diagnostic']['seed_sha256']
        protocol = ROOT / source['stream_protocol']
        registered(ROOT, BLENDS, protocol)
        pinned = [Path(__file__), restart, TARGET / 'manifest.json', TARGET / 'inputs.npz',
                  BLENDS / 'plan.json', BLENDS / 'slice-001/blend-screen.json', seed, protocol,
                  ROOT / 'src/lv_surrogate/physics/parallel_fem.py',
                  ROOT / 'src/lv_surrogate/physics/reference_fem.py',
                  ROOT / 'src/lv_surrogate/physics/quadratic_solver.py',
                  ROOT / 'external/cardiac_benchmark/src/cardiac_benchmark/problem.py',
                  ROOT / 'external/cardiac_benchmark/src/cardiac_benchmark/solver.py']
        run.mkdir()
        atomic(run / 'plan.json', dict(
            seed=str(seed), protocol=str(protocol), checkpoint_sha256=sha(restart),
            gpu_library_sha256=sha(GPU / 'lib/libpetsc.so'),
            frozen={str(path): sha(path) for path in pinned}))
        artifact(ROOT, run, run / 'plan.json', 'Independent residual assembly audit')
        script = run / 'job.pbs'
        command = [str(ROOT / '.venv/bin/python'), str(Path(__file__).resolve()),
                   'execute', '--run-id', run.name]
        script.write_text('#!/bin/bash\nset -eu\ncd ' + shlex.quote(str(ROOT))
                          + '\nexec ' + shlex.join(command) + '\n')
        artifact(ROOT, run, script, 'PBS residual audit command')
        atomic(run / 'state.json', dict(status='submitted'))
        reply = subprocess.run(
            ['qsub', '-N', 'lv_residual_audit', '-q', 'iworkq', '-l',
             'select=1:ncpus=28:mem=240gb:ngpus=4', '-l', 'place=excl',
             '-l', 'software=PBSJobApp', '-l', 'walltime=168:00:00',
             '-j', 'oe', '-o', str(run / 'pbs.log'), str(script)],
            text=True, capture_output=True)
        event(run, 'job_submission', return_code=reply.returncode,
              stdout=reply.stdout.strip(), stderr=reply.stderr.strip())
        reply.check_returncode()
        print('Residual assembly audit job:', reply.stdout.strip())


def execute(run):
    assert os.environ.get('PBS_JOBID'), 'Use a PBS compute job'
    plan = load(run / 'plan.json')
    state = dict(status='running', job=os.environ['PBS_JOBID'])
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
            gpu_env = dict(env, PETSC_OPTIONS='-use_gpu_aware_mpi 0',
                           PKG_CONFIG_PATH=f'{GPU}/lib/pkgconfig:{pkg}',
                           PKG_CONFIG_LIBDIR=f'{GPU}/lib/pkgconfig:{pkg}')
            visible = env.get('CUDA_VISIBLE_DEVICES')
            if not visible:
                visible = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip().replace('\n', ',')
            assert len(set(visible.split(','))) == 4
            gpu_env['CUDA_VISIBLE_DEVICES'] = visible
            base = [str(NATIVE / 'bin/python'), str(Path(__file__).resolve()), 'assemble', '--run-id', run.name]
            for mode, command, environment in (
                ('mpi', [str(NATIVE / 'bin/mpiexec'), '-n', '12', '/usr/bin/env',
                         f'LD_PRELOAD={GPU}/lib/libpetsc.so', *base, '--mode', 'mpi'], gpu_env),
                ('serial', [*base, '--mode', 'serial'], dict(env, CUDA_VISIBLE_DEVICES='')),
            ):
                with (run / f'{mode}.log').open('x') as log:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
            compare(run)
            assert sha(TARGET / 'restart.npz') == plan['checkpoint_sha256']
            assert sha(TARGET / 'manifest.json') == plan['frozen'][str(TARGET / 'manifest.json')]
            state['status'] = 'audit_complete'
        except Exception as exc:
            state.update(status='blocked', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            atomic(run / 'state.json', state)


def assemble(run, mode):
    if mode == 'mpi':
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[int(os.environ['PMI_RANK']) % len(devices)]
    import numpy as np
    from lv_surrogate.physics.parallel_fem import ParallelP2
    from lv_surrogate.physics.reference_fem import df, residual

    plan = load(run / 'plan.json')
    manifest = load(TARGET / 'manifest.json')
    protocol = load(Path(plan['protocol']))
    adapter = ParallelP2(ROOT, protocol['source_reference'], manifest['config'])
    assert adapter.comm.size == (12 if mode == 'mpi' else 1)
    comm, rank = adapter.comm, adapter.rank
    problem = adapter.problem
    loaded = sorted({str(Path(line.split()[-1]).resolve()) for line in Path('/proc/self/maps').read_text().splitlines()
                     if '/libpetsc.so' in line})
    expected = (GPU if mode == 'mpi' else NATIVE) / 'lib/libpetsc.so'
    assert loaded == [str(expected.resolve())], loaded

    def canonical(vector):
        parts = comm.gather(vector.get_local(), root=0)
        if rank:
            return None
        result = np.empty(adapter.ndof)
        for ids, values in zip(adapter.parts, parts):
            result[ids] = values
        return result

    with np.load(TARGET / 'restart.npz', allow_pickle=False) as saved:
        assert int(saved['completed_steps']) == 554
        history = {name: saved[name] for name in ('u_old', 'v_old', 'a_old')}
    for name, values in history.items():
        adapter.assign(name, values)
        roundtrip = adapter.gather(name)
        if rank == 0:
            np.testing.assert_array_equal(roundtrip, values)
    with np.load(plan['seed'], allow_pickle=False) as seed:
        assert int(seed['step']) == 554 and seed['checkpoint_sha256'].item() == plan['checkpoint_sha256']
        warm = seed['u']
    with np.load(TARGET / 'inputs.npz', allow_pickle=False) as inputs:
        pressure = float(inputs['pressure'][554])
        time_s = float(inputs['time'][554])
        previous_time = float(inputs['time'][553])
    problem.parameters['p'].assign(pressure)
    parameters = {}
    for name, value in problem.parameters.items():
        try:
            parameters[name] = float(value)
        except (TypeError, ValueError):
            continue
    assert parameters['dt'] == manifest['config']['dt_s']
    np.testing.assert_allclose(time_s-previous_time, parameters['dt'], rtol=1e-10, atol=1e-14)
    metadata = dict(mode=mode, ranks=comm.size, loaded_petsc=loaded,
                    time_s=time_s, previous_time_s=previous_time, pressure_Pa=pressure,
                    parameters=parameters, canonical_map_error=adapter.map_error,
                    histories={name: dict(sha256=array_hash(value), max_abs=float(np.max(abs(value))),
                                          l2=float(np.linalg.norm(value))) for name,value in history.items()},
                    candidates=[])
    vectors = {}
    for label, candidate in (('theta0', history['u_old']), ('theta1', warm)):
        adapter.assign('u', candidate)
        roundtrip = adapter.gather('u')
        if rank == 0:
            np.testing.assert_array_equal(roundtrip, candidate)
        raw = residual(problem)
        callback = df.PETScVector()
        problem._problem.F(callback, problem.u.vector())
        row = dict(label=label, candidate_sha256=array_hash(candidate),
                   raw_force_N=float(raw.norm('l2')), callback_force_N=float(callback.norm('l2')))
        for kind, vector in (('raw', raw), ('callback', callback)):
            values = canonical(vector)
            if rank == 0:
                vectors[label + '_' + kind] = values
        # Record the acceleration/velocity used by this generalized-alpha equation.
        acceleration = problem.a(u=problem.u.vector(), u_old=problem.u_old.vector(),
                                 v_old=problem.v_old.vector(), a_old=problem.a_old.vector())
        velocity = problem.v(a=acceleration, v_old=problem.v_old.vector(), a_old=problem.a_old.vector())
        local_a = np.asarray(acceleration.get_local())
        local_v = np.asarray(velocity.get_local())
        from mpi4py import MPI
        row['derived_acceleration_max_abs'] = comm.allreduce(float(np.max(abs(local_a))), op=MPI.MAX)
        row['derived_velocity_max_abs'] = comm.allreduce(float(np.max(abs(local_v))), op=MPI.MAX)
        metadata['candidates'].append(row)
        if rank == 0:
            print(label, row, flush=True)
    for name, values in history.items():
        roundtrip = adapter.gather(name)
        if rank == 0:
            np.testing.assert_array_equal(roundtrip, values)
    if rank == 0:
        np.savez(run / f'{mode}-vectors.npz', **vectors)
        atomic(run / f'{mode}.json', metadata)


def compare(run):
    import numpy as np

    mpi, serial = (load(run / f'{mode}.json') for mode in ('mpi', 'serial'))
    assert mpi['parameters'] == serial['parameters']
    assert mpi['histories'] == serial['histories']
    for name in ('time_s', 'previous_time_s', 'pressure_Pa'):
        assert mpi[name] == serial[name]
    rows = []
    with np.load(run / 'mpi-vectors.npz') as parallel, np.load(run / 'serial-vectors.npz') as independent:
        for candidate, native in zip(mpi['candidates'], serial['candidates'], strict=True):
            assert candidate['candidate_sha256'] == native['candidate_sha256']
            label = candidate['label']
            row = dict(label=label, mpi_force_N=candidate['raw_force_N'], serial_force_N=native['raw_force_N'],
                       mpi_callback_force_N=candidate['callback_force_N'],
                       serial_callback_force_N=native['callback_force_N'],
                       derived_acceleration_max_abs=native['derived_acceleration_max_abs'],
                       derived_velocity_max_abs=native['derived_velocity_max_abs'])
            for kind in ('raw', 'callback'):
                name = label + '_' + kind
                difference = parallel[name] - independent[name]
                relative = float(np.linalg.norm(difference) / max(np.linalg.norm(independent[name]), 1e-30))
                row[kind + '_relative_vector_difference'] = relative
                row[kind + '_max_abs_vector_difference'] = float(np.max(abs(difference)))
                assert relative <= 1e-8, (name, relative)
            row['serial_boundary_treatment_difference_N'] = float(
                np.linalg.norm(independent[label+'_raw'] - independent[label+'_callback']))
            rows.append(row)
    screen = load(BLENDS / 'slice-001/blend-screen.json')
    for row, theta in zip(rows, (0., 1.), strict=True):
        previous = next(candidate for candidate in screen['candidates'] if candidate['theta'] == theta)
        np.testing.assert_allclose(row['mpi_force_N'], previous['native_force_N'], rtol=1e-8, atol=1e-9)
    result = dict(complete=True, passed=True, state=555, accepted_states=554, pressure_Pa=serial['pressure_Pa'],
                  time_s=serial['time_s'], parameters=serial['parameters'], histories=serial['histories'],
                  candidates=rows, reference_state_accepted=False,
                  scope='Independent serial CPU reassembly compared with MPI CUDA runtime on identical cached mesh, '
                        'fixed histories and canonical candidates. Both raw and native callback residuals are retained.')
    atomic(run / 'result.json', result)
    for mode in ('mpi', 'serial'):
        artifact(ROOT, run, run / f'{mode}.json', 'Residual assembly metadata')
        artifact(ROOT, run, run / f'{mode}-vectors.npz', 'Canonical native residual vectors')
    artifact(ROOT, run, run / 'result.json', 'Serial versus MPI residual comparison')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('submit', 'execute', 'assemble'))
    parser.add_argument('--run-id', default='cmbbe-residual-audit-555-01')
    parser.add_argument('--mode', choices=('mpi', 'serial'))
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('Invalid run ID')
    run = ROOT / 'experiments/hpc' / args.run_id
    if args.action == 'assemble':
        if not args.mode:
            parser.error('--mode is required for assembly')
        try:
            assemble(run, args.mode)
        except Exception:
            import traceback
            traceback.print_exc()
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
    elif args.action == 'execute':
        execute(run)
    else:
        submit(run)
