"""Submit and continue the full CUDA reference trajectory on one exclusive PBS node."""
import argparse
import fcntl
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
           'scripts/finalize_parallel_reference.py', 'scripts/compare_quadratic_references.py')


def submit(run, number, after=None):
    script = run / 'job.pbs'
    cmd = ['qsub', '-N', 'lv_cuda_ref', '-q', 'iworkq',
           '-l', 'select=1:ncpus=28:mem=240gb:ngpus=1', '-l', 'place=excl',
           '-l', 'software=PBSJobApp', '-l', 'walltime=168:00:00',
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


def initialize(run):
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
    assert not any((target / name).exists() for name in ('failure.json', 'UNACCEPTED_iterate.npz'))
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
    atomic(protocol_path, protocol)
    plan = dict(inherited, run_id=run.name, queue='iworkq',
                stream_protocol=str(protocol_path.relative_to(ROOT)),
                checkpoint_sha256=digest, gpu_library_sha256=sha(GPU_LIB),
                sources={path: sha(ROOT / path) for path in SOURCES})
    atomic(run / 'plan.json', plan)
    artifact(ROOT, run, run / 'plan.json', 'CUDA production plan')
    artifact(ROOT, run, protocol_path, 'Frozen generator protocol for PBS CUDA slices')
    command = shlex.join([str(MODERN), str(Path(__file__).resolve()), 'execute', '--run-id', run.name])
    script = run / 'job.pbs'
    script.write_text(f'#!/bin/bash\nset -eu\ncd {shlex.quote(str(ROOT))}\nexec {command}\n')
    artifact(ROOT, run, script, 'PBS slice command')
    atomic(run / 'state.json', dict(status='submitted', completed_steps=554, checkpoint_sha256=digest))
    submit(run, 1)


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
        visible = env.get('CUDA_VISIBLE_DEVICES')
        if visible:
            device = visible.split(',')[0]
        elif os.environ.get('PBS_GPUFILE'):
            gpu_entry = Path(os.environ['PBS_GPUFILE']).read_text().splitlines()[0]
            match = re.search(r'(?:^|gpu)(\d+)$', gpu_entry)
            assert match, f'Unknown PBS GPU entry: {gpu_entry}'
            device = match[1]
        else:
            # The request uses exclusive placement on the selected single vnode.
            device = '0'
        gpu_env = dict(env, CUDA_VISIBLE_DEVICES=device, PETSC_OPTIONS='-use_gpu_aware_mpi 0',
                       PKG_CONFIG_PATH=f'{GPU}/lib/pkgconfig:{pc_paths}',
                       PKG_CONFIG_LIBDIR=f'{GPU}/lib/pkgconfig:{pc_paths}',
                       LV_CUDA_SLICE=str(output), LV_CUDA_DEADLINE=str(deadline),
                       LV_REFERENCE_TARGET=str(target))
        result = run_command('generate', [
            str(NATIVE / 'bin/mpiexec'), '-n', '12', '/usr/bin/env', f'LD_PRELOAD={GPU_LIB}',
            *cpu, 'hpc/production_cuda_native.py', 'generate', '--policy', str(protocol),
            '--protocol', str(protocol)], gpu_env)
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
            receipt = load(output / 'pause.json')
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
    args = parser.parse_args()
    if not re.fullmatch(r'[\w-]+', args.run_id):
        parser.error('Invalid run ID')
    run = HPC / args.run_id
    if args.action == 'execute':
        execute(run)
    else:
        with (ROOT / '.hpc/profile-submission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert_idle(ROOT)
            initialize(run)


if __name__ == '__main__':
    main()
