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
    pressure_fractions = os.environ.get('LV_CUDA_PRESSURE_FRACTIONS') if step == 554 else None
    residual_fractions = os.environ.get('LV_CUDA_RESIDUAL_FRACTIONS') if step == 554 else None
    assert not (pressure_fractions and residual_fractions)
    continuation = 'pressure' if pressure_fractions else 'residual' if residual_fractions else None
    stage = 0
    exact_pressure = None
    if pressure_fractions:
        fractions = tuple(map(float, pressure_fractions.split(',')))
        assert fractions == (0., .25, .5, .75, 1.)
        with np.load(checkpoint.parent / 'inputs.npz', allow_pickle=False) as inputs:
            old_pressure, exact_pressure = map(float, inputs['pressure'][step - 1:step + 1])
        assert np.isfinite(old_pressure) and np.isfinite(exact_pressure) and old_pressure != exact_pressure
        pressure_schedule = [old_pressure + fraction * (exact_pressure - old_pressure)
                             for fraction in fractions[:-1]] + [exact_pressure]
        assert float(problem.parameters['p']) == exact_pressure
        trace['pressure_continuation'] = dict(fractions=list(fractions), pressures=pressure_schedule)
    if residual_fractions:
        from lv_surrogate.physics.reference_fem import residual

        fractions = tuple(map(float, residual_fractions.split(',')))
        assert fractions == tuple(i / 8 for i in range(1, 9))
        exact_pressure = float(problem.parameters['p'])
        assert np.isfinite(exact_pressure)
        anchor_path = run.parent / 'initial-newton.npz'
        digest = comm.bcast(sha(anchor_path) if rank == 0 else None, root=0)
        assert digest == os.environ['LV_CUDA_RESIDUAL_ANCHOR_SHA256']
        with np.load(anchor_path, allow_pickle=False) as seed:
            assert int(seed['step']) == step
            checkpoint_digest = comm.bcast(sha(checkpoint) if rank == 0 else None, root=0)
            assert seed['checkpoint_sha256'].item() == checkpoint_digest
            adapter.assign('u', seed['u'])
        anchor = residual(problem)
        anchor_force = float(anchor.norm('l2'))
        assert np.isfinite(anchor_force)
        trace['residual_continuation'] = dict(fractions=list(fractions), pressure=exact_pressure,
                                              anchor_force_N=anchor_force, anchor_sha256=digest)
        if rank == 0:
            print(f'RESIDUAL ANCHOR state=555 pressure={exact_pressure:.17g} '
                  f'force_N={anchor_force:.9g}', flush=True)
    resume = run.parent / 'paused-iterate.npz'
    if resume.is_file():
        with np.load(resume, allow_pickle=False) as saved:
            if int(saved['step']) == step:
                digest = comm.bcast(sha(checkpoint) if rank == 0 else None, root=0)
                assert saved['checkpoint_sha256'].item() == digest
                adapter.assign('u', saved['u'])
                options['initial_regularization'] = float(saved['shift'])
                if continuation and f'{continuation}_stage' in saved:
                    stage = int(saved[f'{continuation}_stage'])
                    assert 0 <= stage < len(fractions)
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
                    metadata = {f'{continuation}_stage': stage} if continuation else {}
                    np.savez(output, u=candidate, step=step, shift=shift, checkpoint_sha256=digest,
                             **metadata)
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
            label = 'latest-newton'
            if continuation and stage < len(fractions) - 1:
                label = f'{continuation}-stage-{stage}-latest-newton'
            save_checkpoint(label, candidate, shift)
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
    def solve_stage(record, save, offset=None):
        solve(problem, adapter, history, audit, options, record, save,
              policy='two_good_steps', late_forcing_floor=1e-5,
              cuda_linear=True, check_deadline=check_deadline, save_iterate=remember,
              host_preconditioner=host_preconditioner, directional_armijo=directional_armijo,
              direction_guard=direction_guard, residual_offset=offset)
        assert 0 <= record['custom_force_N'] < 1e-9

    solved = False
    try:
        if continuation:
            label = continuation.upper()
            for stage in range(stage, len(fractions)):
                fraction = fractions[stage]
                pressure = pressure_schedule[stage] if pressure_fractions else exact_pressure
                problem.parameters['p'].assign(pressure)
                latest = None
                exact_stage = stage == len(fractions) - 1
                if rank == 0:
                    print(f'{label} STAGE state=555 index={stage} fraction={fraction:g} '
                          f'pressure={pressure:.17g} exact={exact_stage}', flush=True)
                if exact_stage:
                    solve_stage(trace, persist)
                    break
                offset = None
                if residual_fractions:
                    offset = anchor.copy()
                    offset *= 1. - fraction
                record = dict(step=step, fraction=fraction, pressure=pressure,
                              physical_state_accepted=False)
                path = run / f'{continuation}-stage-{stage}.json'

                def save_stage():
                    if rank == 0:
                        atomic(path, record)

                solve_stage(record, save_stage, offset)
                record['complete'] = True
                save_stage()
                if rank == 0:
                    print(f'{label} STAGE COMPLETE index={stage} '
                          f'force_N={record["custom_force_N"]:.9g}', flush=True)
                options['initial_regularization'] = float(os.environ['LV_CUDA_STAGNATION_SHIFT'])
        else:
            solve_stage(trace, persist)
        persist()
        solved = True
    except PETSc.Error:
        print(f'PETSc failure on rank {rank}, state {step + 1}', file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        os.fsync(sys.stderr.fileno())
        raise
    finally:
        if exact_pressure is not None:
            problem.parameters['p'].assign(exact_pressure)
        if restore is not None:
            restore()
        if restore_depth is not None and not solved:
            restore_depth()


def compare_directions(problem, adapter, history, audit, protocol, spec, baseline_path, output):
    import numpy as np
    from cuda_determinant_direction import correct_direction
    from lv_surrogate.physics.admissible_newton_gentle_shrink import solve
    from lv_surrogate.physics.reference_fem import residual

    force = float(residual(problem).norm('l2'))
    assert np.isclose(force, spec['initial_force_N'], rtol=1e-6, atol=1e-10)
    rank = adapter.rank
    trace = dict(step=554, initial_force_N=force)
    trace_path = output / 'direction-trace.json'
    depth = use_positivity_depth(audit, 20, 24) if rank == 0 else None
    diagnostics = add_positivity_diagnostics(audit) if rank == 0 else None

    def guard(current, direction, merit_gradient):
        return correct_direction(audit, spec['element'], current, direction, merit_gradient)

    options = dict(protocol['admissible_newton'], initial_regularization=spec['shift'],
                   wall_budget_seconds=2700)
    failure = None
    try:
        solve(problem, adapter, history, audit, options, trace, lambda: atomic(trace_path, trace),
              policy='two_good_steps', late_forcing_floor=1e-5, stop_after=spec['directions'],
              cuda_linear=True, host_preconditioner=True, directional_armijo=True,
              direction_guard=guard)
    except RuntimeError as exc:
        failure = str(exc)
        if failure not in ('Regularization ceiling without admissible descent',
                           'Regularized Newton wall budget exhausted',
                           'Regularized Newton iteration ceiling reached',
                           'Shifted linear direction failed true residual qualification'):
            raise
    finally:
        if rank == 0:
            diagnostics()
            depth()
            audit.close()
    if rank != 0:
        return

    def metrics(rows):
        accepted = [row['trials'][-1] for row in rows if row['accepted_custom_update']]
        return dict(directions=len(rows), accepted_updates=len(accepted),
                    full_steps=sum(t['alpha'] == 1 for t in accepted),
                    median_accepted_alpha=float(np.median([t['alpha'] for t in accepted])) if accepted else None,
                    force_N=accepted[-1]['native_force_N'] if accepted else force)

    baseline_rows = json.loads(baseline_path.read_text())['custom_newton_steps']
    rows = trace.get('custom_newton_steps', [])
    corrected = metrics(rows)
    baseline = metrics(baseline_rows[:len(rows)])
    report = dict(state=555, warm_start_matches=True, initial_force_N=force,
                  shift=spec['shift'], requested_directions=spec['directions'],
                  corrected=corrected, baseline_matched=baseline, baseline_terminal=metrics(baseline_rows),
                  corrections_applied=sum(bool(row.get('determinant_direction', {}).get('applied')) for row in rows),
                  corrected_force_advantage_N=baseline['force_N'] - corrected['force_N'],
                  force_target_N=options['custom_absolute_tolerance_N'],
                  stop_reason=failure or ('force_target_reached' if 'custom_force_N' in trace else 'direction_budget'),
                  reference_state_accepted=False)
    atomic(output / 'direction-comparison.json', report)
    print('DIRECTION COMPARISON', report, flush=True)


def screen_blends(problem, adapter, history, audit, spec, seed_path, output):
    import numpy as np
    from audit_reference_cell_555 import cartesian_deformation, inspect_field
    from lv_surrogate.physics.bernstein import BernsteinAudit
    from lv_surrogate.physics.reference_fem import residual

    with np.load(seed_path, allow_pickle=False) as seed:
        warm = seed['u']
    old = history['u_old']
    assert warm.shape == old.shape and np.isfinite(warm).all()
    rank, comm = adapter.rank, adapter.comm
    report = dict(state=555, pressure=float(problem.parameters['p']), element=spec['element'],
                  margin=spec['margin'], weights=spec['weights'], candidates=[],
                  screen_complete=False, reference_state_accepted=False,
                  minimum_scope='Multistart numerical estimates. Whole-element acceptance uses Bernstein proofs.')
    restore_depth = use_positivity_depth(audit, spec['whole_element_depth'], spec['retry_depth']) if rank == 0 else None
    diagnostics = add_positivity_diagnostics(audit) if rank == 0 else None

    def on_root(function):
        result = error = None
        if rank == 0:
            try:
                result = function()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        error = comm.bcast(error, root=0)
        if error:
            raise RuntimeError(error)
        return result

    def evaluate(candidate, theta, force):
        element = spec['element']
        vertices = audit.X[element]
        local = BernsteinAudit(dict(coords=vertices, cells=np.array([[0, 1, 2, 3]]),
                                    canonical_cell_dofs=audit.maps[element:element + 1]))
        nodal = candidate[audit.maps[element]]
        deformation = cartesian_deformation(vertices, nodal, vertices)
        estimate = inspect_field(local, deformation)
        row = dict(theta=theta, native_force_N=force if np.isfinite(force) else None,
                   local_minimum_estimate=estimate['optimized_minimum_estimate'],
                   local_minimum_barycentric=estimate['minimum_barycentric'],
                   local_bernstein=estimate['bernstein'],
                   local_singular_values=estimate['singular_values_at_minimum'],
                   whole_element_admissible=False, positivity_failure=None)
        try:
            proof = audit.pair(candidate, history['u_old'])
        except ValueError as exc:
            row['positivity_failure'] = str(exc)
        else:
            row['whole_element_admissible'] = True
            row['whole_element_proof'] = proof
        report['candidates'].append(row)
        atomic(output / 'blend-screen.json', report)
        print(f'BLEND theta={theta:g} force_N={force:.9g} '
              f'local_min_estimate={row["local_minimum_estimate"]:.9g} '
              f'admissible={row["whole_element_admissible"]}', flush=True)

    try:
        for theta in spec['weights']:
            if theta == 0:
                candidate = old.copy()
            elif theta == 1:
                candidate = warm.copy()
            else:
                candidate = old + theta * (warm - old)
            adapter.assign('u', candidate)
            force = float(residual(problem).norm('l2'))
            on_root(lambda: evaluate(candidate, theta, force))
        def finish():
            report['screen_complete'] = True
            atomic(output / 'blend-screen.json', report)
        on_root(finish)
    finally:
        if rank == 0:
            diagnostics()
            restore_depth()
            audit.close()


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
        if spec.get('kind') == 'direction_comparison':
            assert sha(plan_path.parent / 'baseline-trace.json') == spec['baseline_trace_sha256']
        elif spec.get('kind') == 'blend_screen':
            audit_report = worker8_native.P / 'experiments/hpc/cmbbe-cell-audit-555-01/result.json'
            assert sha(audit_report) == spec['audit_sha256']
        else:
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

    if spec.get('kind') == 'blend_screen':
        screen_blends(problem, adapter, history, audit, spec, seed, output)
        return

    if spec.get('kind') == 'direction_comparison':
        compare_directions(problem, adapter, history, audit, protocol, spec,
                           plan_path.parent / 'baseline-trace.json', output)
        return

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
