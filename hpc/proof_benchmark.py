"""Fixed-field worker benchmark; imports no FE solver and advances no histories."""
import argparse
import cProfile
from contextlib import nullcontext
import hashlib
import json
import statistics
import sys
import threading
import time
from pathlib import Path

P = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(P / 'src'), str(P / 'hpc')]
import numpy as np
from lv_surrogate.physics.threaded_bernstein import ThreadedBernsteinAudit
from registry import atomic, sha
from profile_report import measurements

WORKERS = [4, 1, 2, 8, 16, 16, 8, 2, 1, 4]
MARGIN = 1e-9


def same_proof(expected, observed):
    """Same sufficient positivity criteria, elements and subdivision topology.

    Worker-dependent chunk boundaries change reconstruction sample locations.
    Their errors must each meet the original 1e-11 check, rather than be equal.
    Bound comparisons permit only 1e-12 roundoff; the positive margin never changes.
    """
    for name in ['endpoint', 'stress']:
        a, b = expected[name], observed[name]
        assert a['passed'] is True and b['passed'] is True
        assert a['minimum_Bernstein_bound'] > MARGIN and b['minimum_Bernstein_bound'] > MARGIN
        assert a['original_unresolved_elements'] == b['original_unresolved_elements']
        assert a['disjoint_cells_total'] == b['disjoint_cells_total']
        for key in ['minimum_Bernstein_bound', 'minimum_parent_Bernstein_bound']:
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=1e-12)
        for proof in [a, b]:
            assert proof['polynomial_reconstruction_max_abs_error'] <= 1e-11
            assert len(proof['refined_elements']) == proof['original_unresolved_elements']
            for leaf in proof['refined_elements']:
                assert leaf['classification'] == 'positive_bound' and leaf['lower_bound'] > MARGIN
                assert 0 <= leaf['maximum_depth'] <= 12 and leaf['leaf_tetrahedra'] > 0
        assert len(a['refined_elements']) == len(b['refined_elements'])
        for x, y in zip(a['refined_elements'], b['refined_elements']):
            for key in ['element', 'classification', 'leaf_tetrahedra', 'maximum_depth']:
                assert x[key] == y[key], 'Subdivision topology changed'
            np.testing.assert_allclose(x['lower_bound'], y['lower_bound'], rtol=1e-12, atol=1e-12)


def timed_pair(arrays, u, old, workers, profile_dir=None):
    started = time.perf_counter()
    audit = ThreadedBernsteinAudit(arrays, workers=workers, chunk=8192,
                                   margin=MARGIN, max_depth=12)
    initialization = time.perf_counter() - started
    threads = []
    mutex = threading.Lock()
    profile_mutex = threading.Lock()
    for shard_id, (_, child) in enumerate(audit.shards):
        original = child.field
        def observed(field, original=original, shard_id=shard_id):
            wall, cpu = time.perf_counter(), time.thread_time()
            # Concurrent cProfile instances can conflict over monitoring IDs on
            # Python 3.12. Only the separate attribution run serializes shards;
            # the actual timing comparisons always retain concurrent execution.
            profiler = cProfile.Profile() if profile_dir else None
            try:
                with profile_mutex if profiler else nullcontext():
                    if profiler:
                        profiler.enable()
                    try:
                        return original(field)
                    finally:
                        if profiler:
                            profiler.disable()
            finally:
                row = dict(shard=shard_id, thread=threading.current_thread().name,
                           wall_seconds=time.perf_counter() - wall,
                           thread_cpu_seconds=time.thread_time() - cpu)
                with mutex:
                    index = len(threads)
                    threads.append(row)
                if profiler:
                    path = Path(profile_dir) / f'worker-call-{index:03d}'
                    profiler.dump_stats(str(path.with_suffix('.prof')))
                    stats = measurements(profiler)
                    stats['timing_scope'] = ('Single worker shard internals; shards serialized for attribution only. '
                                             'Inclusive rows overlap; not concurrent throughput or contention. '
                                             'The enclosing wall measurement also includes waiting for the profiler lock.')
                    atomic(path.with_suffix('.json'), dict(row, **stats))
        child.field = observed
    started, cpu = time.perf_counter(), time.process_time()
    try:
        proof = audit.pair(u, old, alpha_f=.4)
        wall = time.perf_counter() - started
        process_cpu = time.process_time() - cpu
    finally:
        audit.close()
    return proof, dict(workers=workers, pair_seconds=wall,
                       process_cpu_seconds=process_cpu, effective_cpu_cores=process_cpu / wall,
                       initialization_seconds=initialization, worker_calls=threads,
                       worker_profile_scope='Attribution only: shards serialized to support portable cProfile; '
                                            'not a measure of concurrent contention or throughput' if profile_dir else None,
                       profiled=profile_dir is not None)


def worker_summary(records):
    result = []
    for workers in sorted({r['workers'] for r in records if not r['profiled']}):
        rows = [r for r in records if r['workers'] == workers and not r['profiled']]
        values = [r['pair_seconds'] for r in rows]
        result.append(dict(workers=workers, repetitions=len(values),
                           median_pair_seconds=statistics.median(values),
                           min_pair_seconds=min(values), max_pair_seconds=max(values),
                           median_effective_cpu_cores=statistics.median(r['effective_cpu_cores'] for r in rows)))
    return result


def run(spec, directory):
    started = time.perf_counter()
    s = json.loads(Path(spec).read_text())
    out = Path(directory) / 'result.json'
    assert not out.exists(), 'Never overwrite or retry a benchmark'
    report = dict(complete=False, passed=False, diagnostic_only=True,
                  physical_new_state_generated=False, reference_continuation_allowed=False,
                  records=[], worker_order=WORKERS, scope=s['scope'])
    atomic(out, report)
    try:
        print('VERIFY FIXED ARCHIVED INPUTS', flush=True)
        snap = json.loads((P / s['snapshot']).read_text())
        assert sha(P / s['snapshot']) == s['snapshot_sha256']
        for path, digest in snap['files'].items():
            assert sha(P / path) == digest, 'Changed source input: ' + path
        root = P / 'data/processed' / snap['reference']
        with np.load(root / 'inputs.npz') as data:
            arrays = {name: data[name] for name in ['coords', 'cells', 'canonical_cell_dofs']}
        fields = np.load(root / 'displacement_coefficients.npy', mmap_mode='r')
        case = s['fixed_case']
        u, old = np.array(fields[case['step']]), np.array(fields[case['previous_step']])
        for name, value in [('field_sha256', u), ('previous_field_sha256', old)]:
            assert hashlib.sha256(value.tobytes()).hexdigest() == case[name], 'Changed archived field'
        with np.load(P / s['known_start_checkpoint']) as cp:
            assert int(cp['completed_steps']) == case['step']
            np.testing.assert_array_equal(old, cp['u_old'])
        archived = json.loads((root / 'positivity.json').read_text())[case['step']]
        assert archived['step'] == case['step']
        baseline = None
        for index, workers in enumerate(WORKERS):
            report['active_case'] = dict(order=index, workers=workers, started_monotonic=time.monotonic())
            atomic(out, report)
            print('PROOF BENCHMARK BEGIN', index + 1, '/', len(WORKERS), 'workers', workers, flush=True)
            proof, row = timed_pair(arrays, u, old, workers)
            same_proof(archived, proof)
            if baseline is None:
                baseline = proof
            same_proof(baseline, proof)
            atomic(Path(directory) / f'proof-{index:02d}.json', proof)
            row.update(proof_sha256=sha(Path(directory) / f'proof-{index:02d}.json'),
                       order=index, semantic_proof_agreement=True)
            report['records'].append(row)
            report['worker_summary'] = worker_summary(report['records'])
            atomic(out, report)
            print('PROOF BENCHMARK SAVED', index + 1, 'workers', workers,
                  'seconds', round(row['pair_seconds'], 3), flush=True)
        # Separate attribution run: never mix profiled timings with the comparison.
        print('WORKER INTERNAL PROFILE BEGIN workers 4', flush=True)
        profiles = Path(directory) / 'worker-profiles'
        profiles.mkdir()
        report['active_case'] = dict(order='worker-profile', workers=4, started_monotonic=time.monotonic())
        atomic(out, report)
        proof, row = timed_pair(arrays, u, old, 4, profile_dir=profiles)
        same_proof(baseline, proof)
        report['worker_profile'] = row
        report.update(complete=True, passed=True,
                      gate_scope='Only fixed-field proof equivalence and benchmark completion; '
                                 'no native solve, convergence, new reference or study-validation pass')
        report.pop('active_case', None)
    except BaseException as exc:
        report['failure'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        atomic(out, report)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--directory', required=True)
    args = ap.parse_args()
    run(P / args.spec, P / args.directory)
