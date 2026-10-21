"""Recorded algebraic continuation evidence; no continuum or full-cycle claim."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
first = json.loads((P/'experiments/metrics/dynamic_quadratic_third_history_recovery.json').read_text())
resume = json.loads((P/'experiments/metrics/dynamic_quadratic_third_history_resume_v1.json').read_text())
audit = json.loads((P/'experiments/metrics/dynamic_quadratic_third_history_resume_v1_verification_resolved.json').read_text())
assert audit['complete'] and audit['passed'] and resume['all_requested_steps_accepted']
records = first['history_continuation']['records']+resume['history_continuation']['records']
fractions = np.array([r['fraction'] for r in records])
force = np.array([r['residual_N'] for r in records])
minutes = np.array([r['seconds']/60 for r in records])
fig, axes = plt.subplots(1,2,figsize=(10,4.1), constrained_layout=True)
colors = ['#426983']*3+['#22806b']*4
axes[0].scatter(fractions,force,c=colors,s=45,zorder=3)
axes[0].axhline(1e-8,color='#ae493b',ls='--',label='Unchanged force tolerance')
axes[0].set(yscale='log',xlabel='Algebraic homotopy fraction',ylabel='Native residual norm (N)',
    title='Converged engineering roots',xlim=(0,1.05),ylim=(1e-12,4e-8))
axes[0].legend(fontsize=8,loc='upper right')
axes[0].annotate('Exact failed equation\nindependently verified',xy=(1,force[-1]),
    xytext=(.54,6e-11),arrowprops={'arrowstyle':'->','color':'#22806b'},fontsize=8)
axes[1].bar(np.arange(len(records)),minutes,color=colors)
axes[1].set_xticks(np.arange(len(records)),[f'{x:.3f}' for x in fractions],rotation=35)
axes[1].set(xlabel='Algebraic homotopy fraction',ylabel='Native solve wall time (min)',
    title='Local solver cost')
for i,r in enumerate(records):
    axes[1].text(i,minutes[i]+.3,f'{r["iterations"]} N',ha='center',fontsize=8)
axes[1].set_ylim(0,max(minutes)*1.22)
for ax in axes:
    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('Third-mesh recovery at 0.273125 s: one discrete physical step',fontsize=12)
fig.text(.5,-.045,'Intermediate roots are not timesteps. N = Newton updates. Bars exclude setup, auditing and the timed-out partial solve.\nFull-cycle, mesh and quadrature qualification remain incomplete.',ha='center',fontsize=8)
fig.savefig(P/'reports/figures/dynamic_quadratic_history_recovery.png',dpi=180,bbox_inches='tight')
print('Saved recorded local-recovery figure')
