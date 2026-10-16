"""Scientific plots drawn only from frozen full-cycle numerical experiment outputs."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
P=Path(__file__).resolve().parents[1];d=json.loads((P/'experiments/metrics/dynamic_final_evaluation.json').read_text());keys=list(d['models']);colors=['#377eb8','#e66101','#4daf4a','#984ea3'];fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');examples={}
for key,color in zip(keys,colors):
 a=dict(np.load(P/f'experiments/metrics/dynamic_{key.replace("/","_")}_example.npz'));examples[key]=a;t=a['time'];error=np.sqrt(np.mean((a['prediction']-a['truth'])**2,axis=(1,2)))*1e6;axes[0,1].plot(t,error,label=key,color=color)
 if key.startswith('full/'):
  axes[0,0].plot(t,np.sqrt(np.mean(a['prediction']**2,axis=(1,2)))*1000,label=key,color=color)
 r=d['models'][key]['splits']['test'];axes[1,0].scatter([x['component_rmse_m']*1e6 for x in r['per_case']],[x['sampled_normalized_force_rms'] for x in r['per_case']],label=key,color=color,marker='o' if key.startswith('full/') else '^')
 c=r['coverage'];x=np.array([z['nominal'] for z in c])+(keys.index(key)-1.5)*.006;y=np.array([z['coverage'] for z in c]);ci=np.array([z['binomial95'] for z in c]);axes[1,1].errorbar(x,y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='o',capsize=3,label=key,color=color)
a=examples[keys[0]];axes[0,0].plot(a['time'],np.sqrt(np.mean(a['truth']**2,axis=(1,2)))*1000,color='black',ls='--',label='FE reference');axes[0,0].set(xlabel='Time(s)',ylabel='Displacement component RMS(mm)',title='Complete zero-state rollout; first held-out geometry');axes[0,0].legend(fontsize=8)
axes[0,1].set(xlabel='Time(s)',ylabel='Frame component RMSE(µm)',title='No true histories supplied after initialization');axes[0,1].legend(fontsize=8)
axes[1,0].set(xlabel='Full-cycle component RMSE(µm)',ylabel='Sampled normalized force RMS',yscale='log',title='Four independent primary test geometries');axes[1,0].legend(fontsize=8)
axes[1,1].plot([.86,.99],[.86,.99],ls='--',color='gray');axes[1,1].set(xlim=(.875,.975),ylim=(-.05,1.05),xlabel='Nominal joint trajectory coverage',ylabel='Geometry coverage; exact95% intervals',title='Twenty calibration geometries; n=4 test');axes[1,1].legend(fontsize=8,loc='lower left');fig.savefig(P/'reports/figures/dynamic_full_cycle.png',dpi=180);plt.close(fig)
# OOD gate and error: input-support decisions are distinct from calibrated intervals.
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');splits=['test','load_shift','shift'];x=np.arange(3)
for k,key in enumerate(keys):
 r=d['models'][key]['splits'];axes[0].bar(x+(k-1.5)*.18,[r[s]['mean_component_rmse_m']*1e6 for s in splits],width=.18,color=colors[k],label=key)
preferred=d['preferred_from_validation_only'];r=d['models'][preferred]['splits'];axes[1].bar(x,[r[s]['FE_routes'] for s in splits],color=['#377eb8','#e66101','#984ea3']);axes[1].set(xticks=x,xticklabels=['Primary test','Load shift','Shape+load shift'],ylabel='Requests routed to FE',title=f'Input-support gate; {preferred}')
for j,s in enumerate(splits):axes[1].text(j,r[s]['FE_routes']+.1,f"{r[s]['FE_routes']}/{r[s]['n_trajectories']}",ha='center')
axes[0].set(xticks=x,xticklabels=['Primary test','Load shift','Shape+load shift'],ylabel='Full-cycle component RMSE(µm)',title='NN errors before routing; shift lacks guarantees');axes[0].legend(fontsize=8);axes[1].set_ylim(0,max(r[s]['n_trajectories'] for s in splits)+1);fig.savefig(P/'reports/figures/dynamic_shift_fallback.png',dpi=180);plt.close(fig)
# Actual endocardial/epicardial boundary, peak prescribed load chosen from inputs.
full=[k for k in keys if k.startswith('full/')];a=examples[full[0]];b=examples[full[1]];j=int(np.argmax(a['pressure']));X=a['coords'];faces=a['facets'][np.isin(a['facet_markers'],[1,2])];faces=faces[X[faces].mean(1)[:,1]>=0];fields=[a['truth'][j],a['prediction'][j],b['prediction'][j]];magnitudes=[np.linalg.norm(u,axis=-1)*1000 for u in fields];error=np.linalg.norm(fields[2]-fields[0],axis=-1)*1e6;norm=Normalize(0,max(v.max() for v in magnitudes));enorm=Normalize(0,error.max());fig=plt.figure(figsize=(15,5.3));fig.subplots_adjust(top=.8,bottom=.18,left=.01,right=.98,wspace=.05);axs=[]
for i,(title,u,value) in enumerate(zip(['FE reference','Data ensemble','Physics ensemble','Physics error'],[*fields,fields[0]],[*magnitudes,error])):
 ax=fig.add_subplot(1,4,i+1,projection='3d');axs.append(ax);cmap=plt.cm.viridis if i<3 else plt.cm.inferno;nn=norm if i<3 else enorm;verts=(X+u)*1000;poly=Poly3DCollection(verts[faces],facecolors=cmap(nn(value[faces].mean(1))),edgecolors='none',linewidths=0);ax.add_collection3d(poly);ax.auto_scale_xyz(verts[:,0],verts[:,1],verts[:,2]);ax.set_box_aspect(np.ptp(verts,axis=0));ax.view_init(elev=20,azim=-60);ax.set(title=title,xlabel='X(mm)',ylabel='Y(mm)',zlabel='Z(mm)');ax.tick_params(labelsize=7)
fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=axs[:3],orientation='horizontal',fraction=.04,label='Displacement magnitude(mm)');fig.colorbar(plt.cm.ScalarMappable(norm=enorm,cmap='inferno'),ax=axs[3],orientation='horizontal',fraction=.04,label='Nodal L2 error(µm)');fig.suptitle(f'First held-out geometry, peak prescribed load at {a["time"][j]:.3f}s\nActual FE boundary triangles; Y≥0 cut-away',y=.98);fig.savefig(P/'reports/figures/dynamic_mesh_prediction.png',dpi=180,bbox_inches='tight');plt.close(fig)
if (P/'experiments/metrics/dynamic_mesh_convergence.json').exists():
 time=json.loads((P/'experiments/metrics/dynamic_time_convergence.json').read_text());mesh=json.loads((P/'experiments/metrics/dynamic_mesh_convergence.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for c in time['cases']:axes[0].plot(np.arange(1,time['n_common_endpoints']+1)*.005,100*np.array(c['endpoint_relative_volume_L2_errors']),label=c['resolution'])
 axes[0].set(xlabel='Time(s)',ylabel='Relative volume-L2 difference(%)',yscale='log',title='Whole-cycle timestep sensitivity vs1ms');axes[0].legend();axes[1].plot([r['time_s'] for r in mesh['cases']],[100*r['relative_volume_L2_difference'] for r in mesh['cases']]);axes[1].set(xlabel='Time(s)',ylabel='Relative volume-L2 difference(%)',title=f'Two spatial meshes; same{mesh["dt_s"]*1000:g}ms timestep');fig.savefig(P/'reports/figures/dynamic_convergence.png',dpi=180);plt.close(fig)
print('DYNAMIC FIGURES COMPLETE',flush=True)
