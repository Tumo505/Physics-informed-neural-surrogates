"""Instrument the frozen known probe, stopping after its first custom solve.

No numerical kernel is copied or edited. An observer wraps the original solve
entry point and stops all ranks before the original probe can advance histories.
"""
import argparse
import cProfile
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

P = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P / 'hpc'), str(P / 'src')]
from registry import atomic, sha
from profile_report import measurements


class DiagnosticFinished(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--directory', required=True)
    args = ap.parse_args()
    # Imports happen in the locked native environment, on a PBS compute node.
    import numpy as np
    import probe_native as probe
    comm = probe.MPI.COMM_WORLD
    rank = comm.rank
    directory = P / args.directory
    prefix = directory / f'rank-{rank:03d}'
    assert not prefix.with_suffix('.json').exists(), 'Never overwrite a diagnostic'
    spec = json.loads((P / args.spec).read_text())
    snap = json.loads((P / spec['snapshot']).read_text())
    error = None
    if rank == 0:
        try:
            for path, digest in snap['files'].items():
                assert sha(P / path) == digest, 'Source input changed: ' + path
            fields = np.load(P / 'data/processed' / snap['reference'] /
                             'displacement_coefficients.npy', mmap_mode='r')
            digest = hashlib.sha256()
            for i in range(snap['accepted_steps']):
                digest.update(np.asarray(fields[i]).tobytes())
            assert digest.hexdigest() == snap['prefix_fields_sha256'], 'Inherited fields changed'
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
    error = comm.bcast(error, root=0)
    assert error is None, error
    start = int(np.load(P / spec['known_start_checkpoint'])['completed_steps'])
    original_solve = probe.solve
    context = {}
    outcome = dict(first_known_equation_converged=False, independent_audit_passed=False,
                   physical_new_state_generated=False, histories_advanced=False,
                   known_pair_qualified=False, diagnostic_only=True, start_step=start)

    def observed_solve(problem, adapter, current, audit, options, trace, persist):
        context.update(problem=problem, adapter=adapter)
        assert options['wall_budget_seconds'] == 1800
        original_solve(problem, adapter, current, audit, options, trace, persist)
        # Exactly the fresh native check used by the frozen probe.
        problem.solver.solve()
        force = float(probe.residual(problem).norm('l2'))
        reason = int(problem.solver._snes.getConvergedReason())
        updates = int(problem.solver._snes.getIterationNumber())
        assert reason > 0 and updates == 0 and force < 1e-8
        field = adapter.gather('u')
        difference = None
        if rank == 0:
            archived = np.load(P / 'data/processed' / snap['reference'] /
                               'displacement_coefficients.npy', mmap_mode='r')
            difference = float(np.max(abs(field - archived[start])))
        difference = comm.bcast(difference, root=0)
        assert difference < 1e-9
        outcome.update(first_known_equation_converged=True, native_force_N=force,
                       SNES_reason=reason, native_updates=updates,
                       known_field_max_difference_m=difference)
        raise DiagnosticFinished('First known equation checked; diagnostic does not advance histories')

    probe.solve = observed_solve
    profile = cProfile.Profile()
    started = time.monotonic()
    status, error = 'error', None
    sys.argv = ['probe_native.py', '--spec', args.spec,
                '--output', str(directory / 'trace.json'),
                '--fields', str(directory / 'unused-fields.npy'),
                '--checkpoint', str(directory / 'unused-checkpoint.npz')]
    profile.enable()
    try:
        probe.main()
        raise AssertionError('Observer did not stop the original probe')
    except DiagnosticFinished:
        status = 'first_known_equation_checked'
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        status = 'solver_timeout' if 'wall budget exhausted' in str(exc) else 'error'
        traceback.print_exc()
    finally:
        profile.disable()
        elapsed = time.monotonic() - started
        profile.dump_stats(str(prefix.with_suffix('.prof')))
        report = measurements(profile)
        report.update(outcome, rank=rank, ranks=comm.size, status=status,
                      error=error, wall_seconds=elapsed)
        # Local owned entries + their exact canonical IDs need no MPI collective
        # on failure. These preserve terminal native functions, never regenerated
        # histories. A missing rank file after a hard kill is explicitly incomplete.
        if context:
            adapter, problem = context['adapter'], context['problem']
            terminal = prefix.with_suffix('.npz')
            np.savez_compressed(terminal, canonical_ids=adapter.mapping,
                                **{k: getattr(problem, k).vector().get_local()
                                   for k in ['u', 'u_old', 'v_old', 'a_old']})
            report['terminal_owned_fields_sha256'] = sha(terminal)
        atomic(prefix.with_suffix('.json'), report)
    # Give other ranks a bounded opportunity to finish their local evidence before
    # MPI_Abort. No unbounded final gather or barrier is used after a failure.
    request = comm.Ibarrier()
    flush_deadline = time.monotonic() + 30
    while not request.Test() and time.monotonic() < flush_deadline:
        time.sleep(.1)
    # Do not wait in a final collective when another rank might have failed.
    if status != 'first_known_equation_checked':
        comm.Abort(2)


if __name__ == '__main__':
    main()
