"""Show both one-step errors and failed closed-loop rollouts without hiding failures."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];d=json.loads((P/'experiments/metrics/cardiac_temporal_balanced_v2_evaluation.json').read_text());labels=list(d['models']);fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained');x=np.arange(2)
ax[0].bar(x,[d['models'][k]['teacher_forced_component_rmse_m']*1e6 for k in labels],color=['#377eb8','#e66101']);ax[0].set(xticks=x,xticklabels=['Data','Physics'],ylabel='Teacher-forced component RMSE(µm)',title='199 future frames; true histories supplied')
ax[1].bar(x,[d['models'][k]['teacher_forced_mean_normalized_residual_rms'] for k in labels],color=['#377eb8','#e66101']);ax[1].set(xticks=x,xticklabels=['Data','Physics'],ylabel='Normalized force RMS',title='Conditional one-step consistency')
for k,label in enumerate(labels):
 a=np.load(P/f'experiments/metrics/cardiac_temporal_balanced_v2_{label}.npz');t=a['time'];err=a['rollout_frame_component_rmse_m'];ax[2].plot(t[:len(err)],err*1000,label='Data' if k==0 else 'Physics (λ=10⁶)',marker='o' if k==0 else '^',markevery=5)
 r=d['models'][label];f=r['failure']
 if f:ax[2].axvline(f['time_s'],ls='--',color=['#377eb8','#e66101'][k],alpha=.6)
ax[2].set(xlabel='Time(s)',ylabel='Recursive rollout component RMSE(mm)',title='Both rollouts stop at element inversion');ax[2].legend(fontsize=8)
fig.savefig(P/'reports/figures/temporal_study.png',dpi=180);print('TEMPORAL FIGURE WRITTEN',flush=True)
