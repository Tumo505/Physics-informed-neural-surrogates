"""Scientific figures from saved predictions, using authentic nodes and graph edges."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
P=Path(__file__).resolve().parents[1];figdir=P/'reports/figures';figdir.mkdir(exist_ok=True)
a=np.load(P/'experiments/metrics/reference_varying_case0.npz');X=a['coords'];u=a['truth'];p=a['prediction'];top=np.load(P/'data/raw/lvData/topologyData/real-node-topology.npy');top=top[top[:,0]<top[:,1]]
fig=plt.figure(figsize=(18,5.4),layout='constrained');vmax=max(np.linalg.norm(u,axis=-1).max(),np.linalg.norm(p,axis=-1).max())
for i,(coords,c,title,limit) in enumerate([(X,np.zeros(len(X)),'Reference geometry',vmax),(X+u,np.linalg.norm(u,axis=-1),'Reference FEM displacement',vmax),(X+p,np.linalg.norm(p,axis=-1),'Released GNN prediction',vmax),(X,np.linalg.norm(p-u,axis=-1),'Nodal prediction error',None)]):
 ax=fig.add_subplot(1,4,i+1,projection='3d');ax.add_collection3d(Line3DCollection(coords[top],colors='gray',linewidths=.12,alpha=.1));m=ax.scatter(*coords.T,c=c,s=2,cmap='viridis' if i<3 else 'magma',vmin=0,vmax=limit);ax.set_title(title,fontsize=10);ax.set_xlabel('x (cm)');ax.set_ylabel('y (cm)');ax.set_zlabel('z (cm)');ax.set_box_aspect(np.ptp(X,axis=0));ax.view_init(20,40)
 if i>0:fig.colorbar(m,ax=ax,shrink=.55,pad=.10,label='|u| (cm)' if i<3 else '|prediction − reference| (cm)')
fig.suptitle('Dalton varying-geometry test:0 — actual 6,784-node graph, not a volume-strain reconstruction',fontsize=12)
fig.savefig(figdir/'reference_prediction.png',dpi=200);plt.close(fig)
if (P/'experiments/metrics/ensemble_pilot.json').exists():
 d=json.loads((P/'experiments/metrics/ensemble_pilot.json').read_text());a=np.load(P/'experiments/metrics/ensemble_plot_data.npz');fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
 cov=d['coverage'];axs[0].plot([.8,1],[.8,1],'k--');y=[r['coverage'] for r in cov];axs[0].errorbar([r['nominal'] for r in cov],y,yerr=np.array([[yy-r['exact_binomial_95CI'][0],r['exact_binomial_95CI'][1]-yy] for yy,r in zip(y,cov)]).T,fmt='o');axs[0].set(xlabel='Nominal geometry coverage',ylabel='Empirical coverage (64 test geometries)',ylim=(.65,1.02),title='Simultaneous nodal L2 balls')
 axs[1].scatter(a['uncertainty'],a['error'],s=14);axs[1].set(xlabel='Mean sample ensemble variance (cm²)',ylabel='Mean nodal error (cm)',title=f"Spearman ρ = {d['spearman_uncertainty_mean_error']:.2f}")
 axs[2].plot([r['retained_fraction'] for r in d['selective_risk']],[r['mean_nodal_l2_cm'] for r in d['selective_risk']],'o-');axs[2].set(xlabel='Fraction retained by low uncertainty',ylabel='Mean nodal error (cm)',title='Selective risk; descriptive pilot')
 fig.suptitle('Data-only five-seed pilot: 64 train / 16 validation / 134 calibration / 64 test');fig.savefig(figdir/'uncertainty_pilot.png',dpi=200);plt.close(fig)
print('Saved',list(figdir.glob('*.png')))
