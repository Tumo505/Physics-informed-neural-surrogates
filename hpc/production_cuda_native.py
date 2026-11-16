"""Run the frozen reference generator with CUDA PETSc Newton linear algebra."""
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import traceback
import time

import worker8_native
from registry import atomic, sha


def add_positivity_diagnostics(audit, on_failure=None):
    """Annotate failures without changing proof arithmetic or its depth limit."""
    original_field = audit.field
    stage = None
    restores = []
    for start, child in audit.shards:
        resolve = child.resolve
        coords = child.X
        base = coords.__array_interface__['data'][0]

        def diagnose(deformation, vertices, resolve=resolve, start=start, base=base,
                     stride=coords.strides[0]):
            try:
                return resolve(deformation, vertices)
            except ValueError as exc:
                offset = vertices.__array_interface__['data'][0] - base
                element = start + offset // stride
                if on_failure is not None:
                    on_failure(stage, element, deformation, vertices, str(exc))
                raise ValueError(f'{exc} (stage={stage}, element={element})') from exc

        child.resolve = diagnose
        restores.append((child, resolve))
    field_index = 0

    def field(values):
        nonlocal field_index, stage
        stage = ('endpoint', 'stress')[field_index]
        field_index = (field_index + 1) % 2
        try:
            return original_field(values)
        except Exception:
            field_index = 0
            raise

    audit.field = field

    def restore():
        audit.field = original_field
        for child, resolve in restores:
            child.resolve = resolve

    return restore


def solve_cuda(problem, adapter, history, audit, options, trace, persist):
    import numpy as np
    from petsc4py import PETSc
    from lv_surrogate.physics.admissible_newton_gentle_shrink import solve

    run = Path(os.environ['LV_CUDA_SLICE'])
    checkpoint = Path(os.environ['LV_REFERENCE_TARGET']) / 'restart.npz'
    step = trace['step']
    deadline = float(os.environ['LV_CUDA_DEADLINE'])
    rank, comm = adapter.rank, adapter.comm
    options = dict(options, wall_budget_seconds=float('inf'))
    latest = None
    resume = run.parent / 'paused-iterate.npz'
    if resume.is_file():
        with np.load(resume, allow_pickle=False) as saved:
            if int(saved['step']) == step:
                digest = comm.bcast(sha(checkpoint) if rank == 0 else None, root=0)
                assert saved['checkpoint_sha256'].item() == digest
                adapter.assign('u', saved['u'])
                options['initial_regularization'] = float(saved['shift'])
                retry_shift = os.environ.get('LV_CUDA_STAGNATION_SHIFT')
                if retry_shift and resume.samefile(run.parent / 'initial-newton.npz'):
                    options['initial_regularization'] = float(retry_shift)
                if rank == 0:
                    print(f'RESUMED NEWTON state={step + 1} shift={options["initial_regularization"]:g}',
                          flush=True)
    host_preconditioner = os.environ.get('LV_CUDA_HOST_PRECONDITIONER') == '1'
    trace['solver_policy'] = 'cuda_fresh_jacobian_two_good_steps'
    directional_armijo = step == 554 and bool(os.environ.get('LV_CUDA_STAGNATION_SHIFT'))
    if directional_armijo:
        trace['line_search'] = 'directional_armijo_squared_force'
    trace['gamg_matrix_backend'] = 'host' if host_preconditioner else 'cuda'

    def save_checkpoint(label, candidate, shift):
        error = None
        if rank == 0:
            try:
                pending = run / f'{label}.pending.npz'
                digest = sha(checkpoint)
                with pending.open('wb') as output:
                    np.savez(output, u=candidate, step=step, shift=shift, checkpoint_sha256=digest)
                    output.flush()
                    os.fsync(output.fileno())
                saved = run / f'{label}.npz'
                pending.replace(saved)
                if label == 'paused':
                    atomic(run / 'pause.json', dict(step=step, checkpoint_sha256=digest,
                                                    iterate_sha256=sha(saved)))
            except Exception as exc:
                error = str(exc)
        error = comm.bcast(error, root=0)
        if error:
            raise RuntimeError(error)

    def log_memory(iteration):
        with Path('/proc/self/status').open() as status:
            rss_kib = next(int(line.split()[1]) for line in status if line.startswith('VmRSS:'))
        ranks = comm.gather(rss_kib, root=0)
        if rank != 0:
            return
        try:
            gpu = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip().replace('\n', ' | ')
        except (OSError, subprocess.SubprocessError) as exc:
            gpu = f'unavailable ({exc})'
        print(f'MEMORY state={step + 1} iteration={iteration} '
              f'rank_rss_sum_KiB={sum(ranks)} rank_rss_max_KiB={max(ranks)} '
              f'GPU index,used_MiB,total_MiB: {gpu}', flush=True)

    def remember(iteration, candidate, force, shift):
        nonlocal latest
        latest = candidate, shift
        if step == 554:
            save_checkpoint('latest-newton', candidate, shift)
            log_memory(iteration)

    def check_deadline():
        if not comm.bcast(time.monotonic() >= deadline if rank == 0 else None, root=0):
            return
        candidate, shift = latest if latest is not None else (None, options['initial_regularization'])
        if candidate is None:
            candidate = adapter.gather('u')
        save_checkpoint('paused', candidate, shift)

        raise RuntimeError('CUDA production slice deadline')

    if step == 554:
        log_memory("start")
    restore = add_positivity_diagnostics(audit) if directional_armijo and rank == 0 else None
    try:
        solve(problem, adapter, history, audit, options, trace, persist,
              policy='two_good_steps', late_forcing_floor=1e-5,
              cuda_linear=True, check_deadline=check_deadline, save_iterate=remember,
              host_preconditioner=host_preconditioner, directional_armijo=directional_armijo)
    except PETSc.Error:
        print(f'PETSc failure on rank {rank}, state {step + 1}', file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        os.fsync(sys.stderr.fileno())
        raise
    finally:
        if restore is not None:
            restore()
    assert 0 <= trace['custom_force_N'] < 1e-9
    persist()


def diagnose_trial(plan_path):
    """Reconstruct one direction and replay its unresolved cell without the generator."""
    import numpy as np
    from mpi4py import MPI
    from cofactor_proof import CofactorBernsteinAudit
    from lv_surrogate.physics.admissible_newton_gentle_shrink import solve
    from lv_surrogate.physics.bernstein import BernsteinAudit
    from lv_surrogate.physics.parallel_fem import ParallelP2

    comm = MPI.COMM_WORLD
    rank = comm.rank
    plan = json.loads(plan_path.read_text())
    spec = plan['positivity_diagnostic']
    target = Path(os.environ['LV_REFERENCE_TARGET'])
    output = Path(os.environ['LV_CUDA_SLICE'])
    seed = plan_path.parent / 'diagnostic-seed.npz'
    restart = target / 'restart.npz'
    protocol = json.loads((worker8_native.P / plan['stream_protocol']).read_text())
    assert comm.size == protocol['parallel_ranks'] == 12
    if rank == 0:
        worker8_native.check_policy(worker8_native.P, protocol, generation=True)
        assert sha(seed) == spec['seed_sha256']
        assert sha(restart) == plan['checkpoint_sha256']
        assert sha(target / 'custom_trace_000554.json') == spec['source_trace_sha256']
        cfg_path = worker8_native.P / protocol['config']
        manifest = json.loads((target / 'manifest.json').read_text())
        assert sha(cfg_path) == manifest['config_sha256']
        assert sha(target / 'inputs.npz') == manifest['source_inputs_sha256']
    cfg = json.loads((worker8_native.P / protocol['config']).read_text())
    adapter = ParallelP2(worker8_native.P, protocol['source_reference'], cfg)
    problem = adapter.problem
    with np.load(restart, allow_pickle=False) as saved:
        assert int(saved['completed_steps']) == 554
        history = {name: saved[name] for name in ('u_old', 'v_old', 'a_old')}
    for name, values in history.items():
        adapter.assign(name, values)
    with np.load(seed, allow_pickle=False) as warm:
        assert int(warm['step']) == 554 and warm['checkpoint_sha256'].item() == plan['checkpoint_sha256']
        adapter.assign('u', warm['u'])
    with np.load(target / 'inputs.npz', allow_pickle=False) as arrays:
        problem.parameters['p'].assign(float(arrays['pressure'][554]))
        audit = CofactorBernsteinAudit(arrays, workers=8) if rank == 0 else None

    captured = []

    def capture(stage, element, deformation, vertices, reason):
        if stage == spec['stage'] and element == spec['element'] and not captured:
            captured.append((deformation.copy(), vertices.copy(), reason))

    trace = dict(step=554)
    restore = add_positivity_diagnostics(audit, capture) if rank == 0 else None
    options = dict(protocol['admissible_newton'], initial_regularization=spec['shift'],
                   maximum_backtracks=0, wall_budget_seconds=1800)
    try:
        solve(problem, adapter, history, audit, options, trace, lambda: None,
              policy='two_good_steps', late_forcing_floor=1e-5, stop_after=1,
              cuda_linear=True, host_preconditioner=True, directional_armijo=True)
    finally:
        if rank == 0:
            restore()
            audit.close()
    if rank != 0:
        return

    atomic(output / 'replay-trace.json', trace)
    row, = trace['custom_newton_steps']
    trial, = row['trials']
    expected_reason = (
        f"Whole-element positivity remains unresolved at maximum depth "
        f"(stage={spec['stage']}, element={spec['element']})"
    )
    reproduced = (trial['alpha'] == 1 and trial['armijo_passed']
                  and trial['positivity_failure'] == expected_reason and len(captured) == 1
                  and np.isclose(row['initial_force_N'], spec['initial_force_N'], rtol=1e-6, atol=1e-10)
                  and np.isclose(trial['native_force_N'], spec['trial_force_N'], rtol=1e-6, atol=1e-10))
    summary = dict(reproduced=bool(reproduced), state=555, element=spec['element'],
                   stage=spec['stage'], alpha=trial['alpha'], shift=spec['shift'],
                   trial_force_N=trial['native_force_N'], margin=audit.margin,
                   original_depth=audit.max_depth, results=[])
    if reproduced:
        deformation, vertices, _ = captured[0]
        np.savez(output / 'rejected-cell.npz', deformation=deformation, vertices=vertices)
        deeper = copy.copy(audit)
        for depth in spec['depths']:
            deeper.max_depth = depth
            try:
                proof = BernsteinAudit.resolve(deeper, deformation, vertices)
            except ValueError as exc:
                reason = str(exc)
                outcome = 'nonpositive_witness' if 'nonpositive' in reason else 'unresolved'
                summary['results'].append(dict(depth=depth, outcome=outcome, reason=reason))
                if outcome == 'unresolved':
                    continue
            else:
                summary['results'].append(dict(depth=depth, outcome='certified_positive', **proof))
            break
    atomic(output / 'positivity-replay.json', summary)
    print('POSITIVITY REPLAY', summary, flush=True)


def main():
    # Hydra exports PMI_RANK before Python starts on the exclusive PBS vnode.
    # Select the rank's device before DOLFIN or petsc4py initializes CUDA.
    gpu_count = int(os.environ.get('LV_CUDA_GPU_COUNT', '1'))
    if gpu_count > 1:
        rank = int(os.environ['PMI_RANK'])
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        assert len(set(devices)) == gpu_count
        device = devices[rank % gpu_count]
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        print(f'CUDA rank {rank} assigned device {device}', flush=True)
    from lv_surrogate.physics import admissible_newton_regularized_forcing
    from mpi4py import MPI
    from petsc4py import PETSc

    expected = Path(os.environ['LD_PRELOAD']).resolve()
    loaded = {Path(line.split()[-1]).resolve()
              for line in Path('/proc/self/maps').read_text().splitlines()
              if '/libpetsc.so' in line}
    assert loaded == {expected}, loaded
    assert PETSc.Sys.hasExternalPackage('cuda')
    if MPI.COMM_WORLD.rank == 0:
        print(f'CUDA PETSc on {MPI.COMM_WORLD.size} MPI ranks: {expected}', flush=True)
    assert os.environ['PBS_JOBID']
    rank = MPI.COMM_WORLD.rank
    path = Path(os.environ['LV_CUDA_SLICE']) / f'petsc-rank-{rank:02d}.log'
    with path.open('ab', buffering=0) as rank_log:
        os.dup2(rank_log.fileno(), sys.stderr.fileno())
    diagnostic = os.environ.get('LV_CUDA_DIAGNOSTIC_PLAN')
    if diagnostic:
        try:
            diagnose_trial(Path(diagnostic))
        except Exception:
            traceback.print_exc()
            sys.stderr.flush()
            MPI.COMM_WORLD.Abort(1)
        return
    admissible_newton_regularized_forcing.solve = solve_cuda
    worker8_native.main()


if __name__ == '__main__':
    main()
