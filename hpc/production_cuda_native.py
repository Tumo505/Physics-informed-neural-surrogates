"""Run the frozen reference generator with CUDA PETSc Newton linear algebra."""
import os
from pathlib import Path
import sys
import traceback
import time

import worker8_native
from registry import atomic, sha


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
    host_preconditioner = os.environ.get('LV_CUDA_HOST_PRECONDITIONER') == '1'
    trace['solver_policy'] = 'cuda_fresh_jacobian_two_good_steps'
    trace['gamg_matrix_backend'] = 'host' if host_preconditioner else 'cuda'

    def remember(iteration, candidate, force, shift):
        nonlocal latest
        latest = candidate, shift

    def check_deadline():
        if not comm.bcast(time.monotonic() >= deadline if rank == 0 else None, root=0):
            return
        candidate, shift = latest if latest is not None else (None, options['initial_regularization'])
        if candidate is None:
            candidate = adapter.gather('u')
        error = None
        if rank == 0:
            try:
                pending = run / 'paused.pending.npz'
                digest = sha(checkpoint)
                with pending.open('wb') as output:
                    np.savez(output, u=candidate, step=step, shift=shift, checkpoint_sha256=digest)
                    output.flush()
                    os.fsync(output.fileno())
                saved = run / 'paused.npz'
                pending.replace(saved)
                atomic(run / 'pause.json', dict(step=step, checkpoint_sha256=digest,
                                                iterate_sha256=sha(saved)))
            except Exception as exc:
                error = str(exc)
        error = comm.bcast(error, root=0)
        if error:
            raise RuntimeError(error)
        raise RuntimeError('CUDA production slice deadline')

    try:
        solve(problem, adapter, history, audit, options, trace, persist,
              policy='two_good_steps', late_forcing_floor=1e-5,
              cuda_linear=True, check_deadline=check_deadline, save_iterate=remember,
              host_preconditioner=host_preconditioner)
    except PETSc.Error:
        print(f'PETSc failure on rank {rank}, state {step + 1}', file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        os.fsync(sys.stderr.fileno())
        raise
    assert 0 <= trace['custom_force_N'] < 1e-9
    persist()


def main():
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
    admissible_newton_regularized_forcing.solve = solve_cuda
    worker8_native.main()


if __name__ == '__main__':
    main()
