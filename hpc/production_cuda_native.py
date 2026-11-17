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


def use_positivity_depth(audit, depth, retry_depth=None):
    """Extend subdivision for the current state through its final proof."""
    assert audit.max_depth == 12 and audit.margin == 1e-9 and depth in (16, 20)
    children = [child for _, child in audit.shards]
    assert all(child.max_depth == audit.max_depth for child in children)
    assert retry_depth is None or (depth == 20 and retry_depth == 24)
    original_pair = audit.pair
    original_resolvers = [child.resolve for child in children] if retry_depth is not None else []
    for child in children:
        child.max_depth = depth

    for child, resolve in zip(children, original_resolvers):
        deeper = copy.copy(child)
        deeper.max_depth = retry_depth

        def resolve_with_retry(deformation, vertices, resolve=resolve, deeper=deeper):
            try:
                return resolve(deformation, vertices)
            except ValueError as exc:
                if str(exc) != 'Whole-element positivity remains unresolved at maximum depth':
                    raise
            proof = deeper.resolve(deformation, vertices)
            return dict(proof, subdivision_retry_depth=retry_depth)

        child.resolve = resolve_with_retry

    def restore():
        for child in children:
            child.max_depth = audit.max_depth
        for child, resolve in zip(children, original_resolvers):
            child.resolve = resolve
        audit.pair = original_pair

    def pair(*args, **kwargs):
        # The generator supplies alpha_f explicitly for the final proof.
        if 'alpha_f' not in kwargs:
            return original_pair(*args, **kwargs)
        try:
            proof = original_pair(*args, **kwargs)
            proof['subdivision_depth_limit'] = depth
            if retry_depth is not None:
                proof['subdivision_retry_depth_limit'] = retry_depth
            return proof
        finally:
            restore()

    audit.pair = pair
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
    depth = int(os.environ.get('LV_CUDA_POSITIVITY_DEPTH', '12')) if step == 554 else 12
    retry_depth = os.environ.get('LV_CUDA_POSITIVITY_RETRY_DEPTH') if step == 554 else None
    retry_depth = int(retry_depth) if retry_depth else None
    if retry_depth is not None:
        trace['positivity_subdivision_retry_depth_limit'] = retry_depth
    if depth != 12:
        trace['positivity_subdivision_depth_limit'] = depth
        if rank == 0:
            retry_note = f' unresolved_retry_depth={retry_depth}' if retry_depth is not None else ''
            print(f'POSITIVITY state={step + 1} maximum_depth={depth}{retry_note}', flush=True)
    guarded = step == 554 and os.environ.get('LV_CUDA_DETERMINANT_DIRECTION') == '1'
    direction_guard = None
    if guarded:
        assert directional_armijo and retry_depth == 24
        from cuda_determinant_direction import correct_direction

        def direction_guard(current, direction, merit_gradient):
            return correct_direction(audit, 813578, current, direction, merit_gradient)

        trace['determinant_direction_element'] = 813578
        if rank == 0:
            print('DETERMINANT DIRECTION state=555 endpoint_element=813578', flush=True)
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
    restore_depth = use_positivity_depth(audit, depth, retry_depth) if depth != 12 and rank == 0 else None
    restore = add_positivity_diagnostics(audit) if directional_armijo and rank == 0 else None
    solved = False
    try:
        solve(problem, adapter, history, audit, options, trace, persist,
              policy='two_good_steps', late_forcing_floor=1e-5,
              cuda_linear=True, check_deadline=check_deadline, save_iterate=remember,
              host_preconditioner=host_preconditioner, directional_armijo=directional_armijo,
              direction_guard=direction_guard)
        assert 0 <= trace['custom_force_N'] < 1e-9
        persist()
        solved = True
    except PETSc.Error:
        print(f'PETSc failure on rank {rank}, state {step + 1}', file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        os.fsync(sys.stderr.fileno())
        raise
    finally:
        if restore is not None:
            restore()
        if restore_depth is not None and not solved:
            restore_depth()


def diagnose_trial(plan_path):
    """Reconstruct a bounded Newton prefix and check the selected rejected trial."""
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
    iteration = spec.get('iteration', 0)
    original_depth = spec.get('original_depth', 12)
    trace = dict(step=554)

    def capture(stage, element, deformation, vertices, reason):
        if (len(trace.get('custom_newton_steps', [])) == iteration
                and stage == spec['stage'] and element == spec['element'] and not captured):
            captured.append((deformation.copy(), vertices.copy(), reason))

    restore = add_positivity_diagnostics(audit, capture) if rank == 0 else None
    restore_depth = use_positivity_depth(audit, original_depth) if rank == 0 and original_depth != 12 else None
    options = dict(protocol['admissible_newton'], initial_regularization=spec['shift'],
                   maximum_backtracks=0, wall_budget_seconds=1800)
    try:
        solve(problem, adapter, history, audit, options, trace, lambda: None,
              policy='two_good_steps', late_forcing_floor=1e-5, stop_after=iteration + 1,
              cuda_linear=True, host_preconditioner=True, directional_armijo=True)
    finally:
        if rank == 0:
            restore()
            if restore_depth is not None:
                restore_depth()
            audit.close()
    if rank != 0:
        return

    atomic(output / 'replay-trace.json', trace)
    rows = trace['custom_newton_steps']
    assert len(rows) == iteration + 1
    row = rows[-1]
    trial, = row['trials']
    prefix_matches = all(
        prior['accepted_custom_update']
        and prior['next_regularization'] == expected['shift']
        and prior['trials'][0]['alpha'] == 1
        and np.isclose(prior['trials'][0]['native_force_N'], expected['force_N'], rtol=1e-6, atol=1e-10)
        for prior, expected in zip(rows[:-1], spec.get('prefix', []), strict=True)
    )
    expected_reason = (
        f"Whole-element positivity remains unresolved at maximum depth "
        f"(stage={spec['stage']}, element={spec['element']})"
    )
    trial_shift = spec.get('trial_shift', spec['shift'])
    reproduced = (prefix_matches and row['iteration'] == iteration and row['regularization'] == trial_shift
                  and trial['alpha'] == 1 and trial['armijo_passed']
                  and trial['positivity_failure'] == expected_reason and len(captured) == 1
                  and np.isclose(row['initial_force_N'], spec['initial_force_N'], rtol=1e-6, atol=1e-10)
                  and np.isclose(trial['native_force_N'], spec['trial_force_N'], rtol=1e-6, atol=1e-10))
    summary = dict(reproduced=bool(reproduced), state=555, element=spec['element'],
                   stage=spec['stage'], iteration=iteration, prefix_matches=prefix_matches,
                   alpha=trial['alpha'], shift=trial_shift, initial_shift=spec['shift'],
                   trial_force_N=trial['native_force_N'], margin=audit.margin,
                   original_depth=original_depth, results=[])
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
