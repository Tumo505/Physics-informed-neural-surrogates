"""Portable cProfile measurements; inclusive rows and MPI ranks must not be summed."""
import pstats


def measurements(profile):
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append(dict(file=filename, line=line, function=function,
                         primitive_calls=primitive, calls=calls,
                         exclusive_seconds=own, inclusive_seconds=cumulative))
    return dict(exclusive_profiled_seconds=sum(r['exclusive_seconds'] for r in rows),
                timing_scope='Per-rank elapsed call times including waits. Inclusive rows overlap; ranks overlap. '
                             'Proof worker internals are not profiled; root pair/field wall time includes their wait. '
                             'Instrumentation overhead is included; this is not an uninstrumented throughput benchmark.',
                functions=sorted(rows, key=lambda r: r['inclusive_seconds'], reverse=True))
