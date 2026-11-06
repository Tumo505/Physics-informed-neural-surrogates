"""Reproduce the fixed-field worker comparison from registered cluster timings."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['svg.hashsalt'] = 'hpc-proof-worker-benchmark-20261106'

P = Path(__file__).resolve().parents[1]
data = json.loads((P / 'experiments/metrics/hpc_worker8_followup_20261106.json').read_text())
measurements = data['completed_cluster_benchmark']['pair_seconds_by_workers']
workers = sorted(map(int, measurements))
fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
values = [statistics.median(measurements[str(w)]) for w in workers]
ax.bar(range(len(workers)), values, color=['#9ba9b7'] * 3 + ['#147d92', '#9ba9b7'], width=.6)
for i, w in enumerate(workers):
    ax.scatter([i-.08, i+.08], measurements[str(w)], c='black', s=17, zorder=3)
    ax.text(i, values[i]+13, f'{values[i]:.1f}', ha='center', fontsize=10)
ax.set(xticks=range(len(workers)), xticklabels=workers, xlabel='Proof worker threads',
       ylabel='Endpoint + stress proof time (s)', ylim=(0, 490),
       title='Fixed archived fields: whole-element positivity checks')
ax.spines[['top', 'right']].set_visible(False)
ax.text(0, -.25, 'Bars: medians; dots: two measurements per count. Standalone proof benchmark.\n'
        'Unchanged proof criteria; this is not full-solver or full-cycle throughput.',
        transform=ax.transAxes, fontsize=8)
directory = P / 'reports/figures'
directory.mkdir(exist_ok=True)
for extension in ['png', 'svg']:
    path = directory / ('hpc_proof_worker_benchmark.' + extension)
    fig.savefig(path, dpi=180, metadata={'Date': None} if extension == 'svg' else None)
    if extension == 'svg':
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
plt.close(fig)
