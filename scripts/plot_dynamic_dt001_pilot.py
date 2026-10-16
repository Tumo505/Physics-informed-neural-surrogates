"""Original-timestep causal rollout, with explicit temporal split and limitations."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];cfg=json.loads((P/'configs/physics/dynamic_dt001_pilot.json').read_text());name=cfg['name'];m=json.loads((P/f'experiments/metrics/{name}.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for label,color in zip(m['models'],['#377eb8','#e66101']):
 a=np.load(P/f'experiments/metrics/{name}_{label}_prediction.npz');t=a['time'];u=a['prediction'];truth=a['truth'];axes[0].plot(t,np.sqrt(np.mean(u**2,axis=(1,2)))*1000,label=label,color=color);axes[1].plot(t,np.sqrt(np.mean((u-truth)**2,axis=(1,2)))*1000,label=label,color=color)
axes[0].plot(t,np.sqrt(np.mean(truth**2,axis=(1,2)))*1000,color='black',ls='--',label='FE reference')
for ax in axes:
 ax.axvspan(.6,.8,color='gold',alpha=.12,label='Selection block' if ax is axes[0] else None);ax.axvspan(.8,1,color='gray',alpha=.12,label='Repeated test block' if ax is axes[0] else None);ax.set_xlabel('Time(s)');ax.legend(fontsize=8)
axes[0].set(ylabel='Displacement component RMS(mm)',title='Original1ms mesh/loads; zero-state999-step rollout');axes[1].set(ylabel='Frame component RMSE(mm)',title='One geometry/cycle/seed; no true histories supplied');fig.savefig(P/'reports/figures/dynamic_dt001_rollout.png',dpi=180);plt.close(fig);print('ORIGINAL1ms FIGURE COMPLETE')
