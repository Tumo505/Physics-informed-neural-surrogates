"""Plot only completed, independently verified reference comparisons."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
q = json.loads((P/'experiments/metrics/dynamic_quadratic_comparison.json').read_text())
assert q['complete'] and q.get('full_cycle', True)
assert q['native_norm_verification']['passed']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
for key, label in [('coarse_P1', 'Linear: 17,625 tetrahedra'),
                   ('refined_P1', 'Linear: 141,000 tetrahedra')]:
    comparison = q['comparisons'][key]
    frames = comparison['frames']
    times = [f['time_s'] for f in frames]
    absolute = np.sqrt(np.array([f['difference_volume_L2_squared'] for f in frames])
                       / (3*q['domain_volume_m3'])) * 1000
    relative = 100*np.array([f['relative_difference_vs_P2'] for f in frames])
    axes[0].plot(times, absolute, label=label)
    axes[1].plot(times, relative, label=label)
axes[0].set(xlabel='Time (s)', ylabel='Volume component RMSE (mm)',
            title='Difference from quadratic baseline')
axes[1].set(xlabel='Time (s)', ylabel='Relative volume-L2 difference (%)',
            title='Endpoint relative differences')
for ax in axes:
    ax.legend(fontsize=8)
aggregate = '; '.join(f'{key}: {100*r["relative_spacetime_volume_L2_difference_vs_P2"]:.2f}%'
                      for key, r in q['comparisons'].items())
fig.suptitle(f'Full cycle: linear 2.5 ms, quadratic {q["reference_dt_s"]*1000:g} ms; '
             'quadratic comparison is not continuum truth\n'
             + 'Space-time volume-L2: ' + aggregate, fontsize=10)
fig.savefig(P/'reports/figures/dynamic_quadratic_reference.png', dpi=180)
print('QUADRATIC COMPARISON FIGURE COMPLETE', flush=True)
