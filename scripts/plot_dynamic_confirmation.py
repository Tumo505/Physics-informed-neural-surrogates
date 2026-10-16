"""Pooled fixed-model confirmation figures, with independent geometry intervals."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];d=json.loads((P/'experiments/metrics/dynamic_confirmation_evaluation.json').read_text());fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');colors=['#377eb8','#e66101','#4daf4a','#984ea3']
for k,(key,r) in enumerate(d['models'].items()):
 r=r['pooled_primary_and_confirmation'];g=r['per_case'];color=colors[k];axes[0,0].scatter([x['component_rmse_m']*1000 for x in g],[x['sampled_normalized_force_rms'] for x in g],color=color,label=key,marker='o' if k<2 else '^',alpha=.8,s=24)
 c=r['coverage'];xx=np.array([x['nominal'] for x in c])+(k-1.5)*.006;y=np.array([x['coverage'] for x in c]);ci=np.array([x['binomial95'] for x in c]);axes[0,1].errorbar(xx,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='o',capsize=4,color=color,label=key)
 axes[1,0].scatter([x['mean_coordinate_variance_m2']*1e6 for x in g],[x['mean_nodal_l2_m']*1000 for x in g],color=color,marker='o' if k<2 else '^',alpha=.8,s=24,label=key)
 axes[1,1].plot([x['retained'] for x in r['selective_risk']],[x['mean_nodal_l2_m']*1000 for x in r['selective_risk']],marker='o',color=color,label=key)
axes[0,0].set(xlabel='Full-cycle component RMSE(mm)',ylabel='Sampled normalized force RMS',yscale='log',title='Fixed ensembles;20 independent test geometries');axes[0,0].legend(fontsize=8)
axes[0,1].plot([.86,.99],[.86,.99],color='gray',ls='--');axes[0,1].set(xlim=(.875,.975),ylim=(0,1.05),xlabel='Nominal joint trajectory coverage',ylabel='Empirical coverage; exact95% intervals',title='Same20 calibration geometries; no recalibration')
axes[1,0].set(xlabel='Mean coordinate sample variance(mm²)',ylabel='Mean nodal L2 error(mm)',xscale='log',yscale='log',title='Raw uncertainty versus error; not intervals alone')
axes[1,1].set(xlabel='Fraction retained by increasing ensemble variance',ylabel='Mean nodal L2 error(mm)',title='Selective risk on20 independent test geometries');axes[1,1].legend(fontsize=8)
fig.savefig(P/'reports/figures/dynamic_confirmation.png',dpi=180);plt.close(fig);print('CONFIRMATION FIGURE COMPLETE',flush=True)
