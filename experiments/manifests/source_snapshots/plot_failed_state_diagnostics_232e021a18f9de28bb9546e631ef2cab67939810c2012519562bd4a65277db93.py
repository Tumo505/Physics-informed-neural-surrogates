"""Scientific diagnostic figure from immutable measured native results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];d=json.loads((P/'experiments/metrics/dynamic_quadratic_failed_state_diagnostics_v1.json').read_text());assert d['complete']
labels=['Physical anchor','Engineering root','Stalled iterate','Exact failed equation'];fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
for state_index,(r,label) in enumerate(zip(d['records'],labels)):
 for direction in r['directions']:
  f=direction['finite_differences'];axes[0].loglog([a['epsilon_m'] for a in f],[a['relative_error'] for a in f],marker='o',lw=1,alpha=.8,color=f'C{state_index}',ls='-' if direction['direction']==0 else ':',label=label if direction['direction']==0 else None)
axes[0].axhline(1e-4,color='black',ls='--',lw=1,label='Declared FD tolerance');axes[0].set(xlabel='Centered FD step (m)',ylabel='Relative tangent-action discrepancy',title='Two directions per state (solid / dotted)');axes[0].legend(fontsize=6)
linear=[r for r in d['records'] if 'linear_solve' in r]
for r,label in zip(linear,labels[2:]):
 a=r['linear_solve'];axes[1].semilogy([x['iteration'] for x in a['history']],[x['reported_residual_N'] for x in a['history']],label=label);axes[1].scatter([a['iterations']],[a['true_residual_N']],marker='x',s=45,color='black')
axes[1].set(xlabel='GMRES iteration',ylabel='Linear residual (N)',title='Tight linear solves; × = true residual');axes[1].legend(fontsize=7)
axes[2].bar(range(4),[r['native_force_N'] for r in d['records']],color=['#3579a8','#3579a8','#d2902f','#a84242']);axes[2].set_yscale('log');axes[2].axhline(1e-8,color='black',ls='--',lw=1);axes[2].set_xticks(range(4),['Anchor','Root','Stalled','Exact'],rotation=25);axes[2].set(ylabel='Reassembled force norm (N)',title='Different algebraic equations/states')
fig.suptitle('Failed-step diagnostics: checks passed; nonlinear recovery not established',fontsize=11)
fig.savefig(P/'reports/figures/dynamic_failed_state_diagnostics.png',dpi=180);plt.close(fig)
