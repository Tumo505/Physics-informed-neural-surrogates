"""Use the separately qualified eight-worker policy with unchanged native kernels."""
import argparse
import hashlib
import json
import runpy
import sys
import time
import traceback
from pathlib import Path

P = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P / 'src'), str(P / 'hpc')]
from registry import atomic, sha
from worker8_gates import cofactor_benchmark_gate, known_gate


def install(module, implementation='legacy'):
    assert implementation in ('legacy', 'cofactor')
    audit_class = module.ThreadedBernsteinAudit
    if implementation == 'cofactor':
        from cofactor_proof import CofactorBernsteinAudit
        audit_class = CofactorBernsteinAudit

    class EightWorkerAudit(audit_class):
        def __init__(self, *args, **kwargs):
            assert kwargs.get('workers', 8) == 8, 'Only the fixed eight-worker policy is admitted'
            kwargs['workers'] = 8
            super().__init__(*args, **kwargs)

        def pair(self, *args, **kwargs):
            proof = super().pair(*args, **kwargs)
            if implementation == 'cofactor':
                proof['proof_implementation'] = implementation
            return proof
    module.ThreadedBernsteinAudit = EightWorkerAudit


def check_policy(root, spec, generation=False):
    policy = spec['worker_policy']
    assert policy['workers'] == 8 and policy['per_step_seconds'] == 1800
    assert sha(root / policy['benchmark_summary']) == policy['benchmark_summary_sha256']
    benchmark = json.loads((root / policy['benchmark_summary']).read_text())
    assert benchmark['complete'] and benchmark['proof_benchmark_passed']
    implementation = policy.get('implementation', 'legacy')
    assert implementation in ('legacy', 'cofactor')
    if implementation == 'cofactor':
        cofactor_benchmark_gate(benchmark, policy['benchmark_steps'])
    if generation:
        path = root / policy['known_verification']
        assert sha(path) == policy['known_verification_sha256']
        verification = json.loads(path.read_text())
        assert verification['complete'] and verification['passed']
        assert len(verification['records']) == 2
        assert all(0 <= row['native_residual_N'] < 1e-8 for row in verification['records'])
        known_gate(root, policy['known_probe'], policy['known_verification'],
                   policy['known_fields'], policy['known_checkpoint'], implementation=implementation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['probe', 'generate'])
    ap.add_argument('--policy', required=True)
    args, forwarded = ap.parse_known_args()
    expected_flag = '--protocol' if args.mode == 'generate' else '--spec'
    assert forwarded[forwarded.index(expected_flag) + 1] == args.policy, 'Policy/spec must agree'
    spec = json.loads((P / args.policy).read_text())
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    error = None
    if comm.rank == 0:
        try:
            check_policy(P, spec, generation=args.mode == 'generate')
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
    error = comm.bcast(error, root=0)
    assert error is None, error
    import lv_surrogate.physics.threaded_bernstein as threaded
    install(threaded, spec['worker_policy'].get('implementation', 'legacy'))
    if args.mode == 'generate':
        sys.argv = ['generate_regularized_reference.py', *forwarded]
        # Original durable journal/checkpoint kernel and its failure handling.
        runpy.run_path(str(P / 'scripts/generate_regularized_reference.py'), run_name='__main__')
        return
    import numpy as np
    import probe_native as probe
    comm = probe.MPI.COMM_WORLD
    assert comm.size == spec['ranks'] == 12
    snap = json.loads((P / spec['snapshot']).read_text())
    assert sha(P / spec['snapshot']) == spec['snapshot_sha256']
    error = None
    if comm.rank == 0:
        try:
            for path, digest in snap['files'].items():
                assert sha(P / path) == digest, 'Source input changed: ' + path
            fields = np.load(P / 'data/processed' / snap['reference'] /
                             'displacement_coefficients.npy', mmap_mode='r')
            digest = hashlib.sha256()
            for field in fields[:snap['accepted_steps']]:
                digest.update(np.asarray(field).tobytes())
            assert digest.hexdigest() == snap['prefix_fields_sha256'], 'Source prefix changed'
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
    error = comm.bcast(error, root=0)
    assert error is None, error
    context = {}
    original_solve = probe.solve
    def capture(problem, adapter, current, audit, options, trace, persist):
        assert options['wall_budget_seconds'] == 1800
        context.update(problem=problem, adapter=adapter, audit=audit)
        return original_solve(problem, adapter, current, audit, options, trace, persist)
    probe.solve = capture
    sys.argv = ['probe_native.py', *forwarded]
    failed = None
    try:
        probe.main()
    except BaseException as exc:
        traceback.print_exc()
        failed = f'{type(exc).__name__}: {exc}'
    finally:
        # Exact terminal owned functions, with no new collective or reconstructed
        # histories, even if the first or second known equation times out.
        output = P / forwarded[forwarded.index('--output') + 1]
        path = output.parent / f'terminal-rank-{comm.rank:03d}.npz'
        record = dict(rank=comm.rank, failed=failed is not None, reason=failed, workers=8)
        if context:
            adapter, problem = context['adapter'], context['problem']
            np.savez_compressed(path, canonical_ids=adapter.mapping,
                                **{k: getattr(problem, k).vector().get_local()
                                   for k in ['u', 'u_old', 'v_old', 'a_old']})
            record['owned_native_fields_sha256'] = sha(path)
            if context['audit'] is not None:
                context['audit'].close()
        atomic(path.with_suffix('.json'), record)
    request = comm.Ibarrier()
    deadline = time.monotonic() + 30
    while not request.Test() and time.monotonic() < deadline:
        time.sleep(.1)
    if failed:
        comm.Abort(2)


if __name__ == '__main__':
    main()
