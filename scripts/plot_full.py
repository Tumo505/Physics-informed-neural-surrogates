"""Plot only saved quantitative outputs and authentic mesh nodes."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];d=json.loads((P/'experiments/metrics/full_evaluation.json').read_text());a=np.load(P/'experiments/metrics/full_plot_data.npz');fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
keys=['released_gnn_same503','full_point_baseline','full_gnn_ensemble'];labels=['Released GNN','Full-data point NN','Full-data GNN ensemble'];axes[0].bar(labels,[d['models'][k]['mean_nodal_l2_cm'] for k in keys],color=['#6684a3','#999999','#007f73']);axes[0].set(ylabel='Mean nodal error (cm)',title='Same 503 held-out geometries');axes[0].tick_params(axis='x',rotation=15)
axes[1].plot([.8,1],[.8,1],'k--');cov=d['coverage'];y=[c['coverage'] for c in cov];axes[1].errorbar([c['nominal'] for c in cov],y,yerr=np.array([[v-c['ci95'][0],c['ci95'][1]-v] for v,c in zip(y,cov)]).T,fmt='o',color='#007f73');axes[1].set(xlabel='Nominal simultaneous geometry coverage',ylabel='Observed test coverage',ylim=(.7,1.01),title='134 calibration / 503 test')
axes[2].scatter(a['variance'],a['error'],s=9,alpha=.6,color='#007f73');axes[2].set(xlabel='Mean coordinate ensemble variance (cm²)',ylabel='Mean nodal error (cm)',title=f"Uncertainty–error Spearman ρ={d['spearman_uncertainty_error']:.2f}");fig.suptitle('Full-data passive LV surrogates — five seeds, all 2,250 training geometries');fig.savefig(P/'reports/figures/full_results.png',dpi=200);plt.close(fig)
if (P/'experiments/metrics/physics_experiment.json').exists():
 b=json.loads((P/'experiments/metrics/physics_experiment.json').read_text());names=list(b['results']);fig,ax=plt.subplots(1,2,figsize=(9,4),layout='constrained')
 for i,key in enumerate(['mean_component_rmse_m','mean_normalized_residual_rms']):
  vals=[b['results'][n][key] for n in names];ax[i].bar(['Data-only','Physics-guided'],vals,color=['#888888','#007f73']);ax[i].set_ylabel('Displacement component RMSE (m)' if i==0 else 'Normalized free-DOF force residual RMS');ax[i].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
 fig.suptitle('Separate cardiac startup pilot: 1 geometry, 20/6/6 load cases, 1 matched seed');fig.savefig(P/'reports/figures/cardiac_comparison.png',dpi=200);plt.close(fig)
X=np.load(P/'data/raw/lvData/rawData/test/real-node-coords.npy',mmap_mode='r')[64];truth=a['truth_case0'];pred=a['ensemble_case0'];maximum=max(np.linalg.norm(truth,axis=-1).max(),np.linalg.norm(pred,axis=-1).max());fig=plt.figure(figsize=(14,4.5),layout='constrained')
for i,(coordinates,values,title) in enumerate([(X+truth,np.linalg.norm(truth,axis=-1),'Reference FEM'),(X+pred,np.linalg.norm(pred,axis=-1),'Full-data GNN ensemble'),(X,np.linalg.norm(pred-truth,axis=-1),'Nodal error on reference')]):
 ax=fig.add_subplot(1,3,i+1,projection='3d');sc=ax.scatter(*coordinates.T,c=values,s=2,cmap='viridis' if i<2 else 'magma',vmin=0,vmax=maximum if i<2 else None);ax.set_title(title);ax.set_xlabel('x(cm)');ax.set_ylabel('y(cm)');ax.set_box_aspect(np.ptp(X,axis=0));ax.view_init(20,40);fig.colorbar(sc,ax=ax,shrink=.6,pad=.1,label='Displacement magnitude(cm)' if i<2 else 'L2 error(cm)')
fig.suptitle('Actual Dalton test:64, 6784 nodes; same magnitude scale, no deformation exaggeration');fig.savefig(P/'reports/figures/full_prediction.png',dpi=200);plt.close(fig)
print('full-data figures written')
