"""Scientific figures from frozen geometry study outputs and actual FE surfaces."""
from pathlib import Path
import json,argparse,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
P=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--run',default='cardiac_geometry_models_v2');ap.add_argument('--raw-root',default='data/processed/cardiac_geometry_v1');ap.add_argument('--prefix',default='geometry');args=ap.parse_args();name=args.run;d=json.loads((P/f'experiments/metrics/{name}_evaluation.json').read_text());labels=list(d['models']);colors=['#377eb8','#e66101'];fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
for k,label in enumerate(labels):
 r=d['models'][label]['splits']['test'];g=r['per_geometry'];axes[0,0].scatter([x['component_rmse_m']*1e6 for x in g],[x['normalized_residual_rms'] for x in g],color=colors[k],label=label,s=25,marker='o' if k==0 else '^')
 cover=r['coverage'];x=np.array([c['nominal'] for c in cover]);y=np.array([c['coverage'] for c in cover]);ci=np.array([c['binomial95'] for c in cover]);axes[1,0].errorbar(x+(k-.5)*.006,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='o',capsize=4,color=colors[k],label=label)
 axes[1,1].scatter([x['variance_m2']*1e12 for x in g],[x['mean_nodal_l2_m']*1e6 for x in g],color=colors[k],label=label,s=25,marker='o' if k==0 else '^')
for b,p in zip(d['models'][labels[0]]['splits']['test']['per_geometry'],d['models'][labels[1]]['splits']['test']['per_geometry']):axes[0,0].plot([b['component_rmse_m']*1e6,p['component_rmse_m']*1e6],[b['normalized_residual_rms'],p['normalized_residual_rms']],color='grey',alpha=.3,lw=.6,zorder=0)
axes[0,0].set(xlabel='Component RMSE (µm)',ylabel='Normalized force RMS',yscale='log',title='Five-member ensembles; 16 test geometries');axes[0,0].legend(fontsize=8)
s=d['matched_seed_comparisons'];xx=np.arange(len(s));axes[0,1].bar(xx-.18,[r['data_residual_rms'] for r in s],.36,color=colors[0],label='data');axes[0,1].bar(xx+.18,[r['physics_residual_rms'] for r in s],.36,color=colors[1],label='physics',hatch='//');axes[0,1].set(xticks=xx,xticklabels=[r['seed'] for r in s],xlabel='Matched initialization seed',ylabel='Mean normalized force RMS',title='Individual models on same 16 geometries');axes[0,1].legend()
axes[1,0].plot([.85,1],[.85,1],ls='--',color='grey');axes[1,0].set(xlim=(.87,.98),ylim=(-.05,1.05),xlabel='Nominal joint coverage',ylabel='Empirical geometry coverage',title='Joint four-load coverage; exact 95% intervals')

for k,label in enumerate(labels):
 shift=d['models'][label]['splits']['shift']['coverage'];axes[1,0].scatter([c['nominal']+(k-.5)*.006 for c in shift],[c['coverage'] for c in shift],marker='x',color=colors[k],label=f'Shift {label}: '+','.join(str(c["covered"])+"/"+str(c["n_geometries"]) for c in shift))
axes[1,0].legend(fontsize=7,loc='center left')
axes[1,1].set(xscale='log',yscale='log',xlabel='Mean coordinate sample variance (µm²)',ylabel='Mean nodal error (µm)',title='Raw ensemble uncertainty; not calibrated alone')
axes[1,1].xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'));axes[1,1].xaxis.set_minor_formatter(NullFormatter())
fig.savefig(P/f'reports/figures/{args.prefix}_study.png',dpi=180);plt.close(fig)
g=d['models'][labels[0]]['splits']['test']['per_geometry'][0]['geometry'];geo=np.load(P/args.raw_root/f'geometry_{g:03d}.npz');X=geo['coords'];faces=geo['facets'][np.isin(geo['facet_markers'],[1,2])];faces=faces[X[faces].mean(1)[:,1]>=0];true=geo['displacements'][3];pred=[np.load(P/f'experiments/metrics/{name}_{label}_predictions.npz')['test'].mean(0)[3] for label in labels];values=[np.linalg.norm(u,axis=-1)*1000 for u in [true,*pred]];error=np.linalg.norm(pred[1]-true,axis=-1)*1e6;norm=Normalize(0,max(v.max() for v in values));enorm=Normalize(0,max(error.max(),1e-12));fig=plt.figure(figsize=(15,5.3));fig.subplots_adjust(top=.80,bottom=.18,left=.01,right=.99,wspace=.05);axs=[]
for i,(title,u,value) in enumerate(zip(['FE reference','Data ensemble','Physics ensemble','Physics error'],[true,*pred,true],[*values,error])):
 ax=fig.add_subplot(1,4,i+1,projection='3d');axs.append(ax);cmap=plt.cm.viridis if i<3 else plt.cm.inferno;nn=norm if i<3 else enorm;verts=(X+u)*1000;poly=Poly3DCollection(verts[faces],facecolors=cmap(nn(value[faces].mean(1))),edgecolors='none',linewidths=0);ax.add_collection3d(poly);ax.auto_scale_xyz(verts[:,0],verts[:,1],verts[:,2]);ax.set_box_aspect(np.ptp(verts,axis=0));ax.view_init(elev=20,azim=-60);ax.set(title=title,xlabel='X(mm)',ylabel='Y(mm)',zlabel='Z(mm)');ax.tick_params(labelsize=7)
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=axs[:3],orientation='horizontal',fraction=.04,label='Displacement magnitude (mm)');fig.colorbar(plt.cm.ScalarMappable(norm=enorm,cmap='inferno'),ax=axs[3],orientation='horizontal',fraction=.04,label='Nodal L2 error (µm)');fig.suptitle(f'Held-out numerical geometry{g}, maximum startup load\nActual endocardial/epicardial FE boundary triangles; Y≥0 cut-away',y=.98,fontsize=12)
fig.savefig(P/f'reports/figures/{args.prefix}_prediction.png',dpi=180,bbox_inches='tight');plt.close(fig);print('GEOMETRY FIGURES WRITTEN',flush=True)
