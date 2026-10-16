from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for name in ['control_v4','capacity_v4','fine_graph_v4']:
 d=json.loads((P/f'experiments/metrics/{name}/gnn_11.json').read_text());h=[r for r in d['history'] if 'validation_component_mse_cm2' in r and r['epoch']>=0];ax[0].plot([r['epoch']+1 for r in h],np.sqrt([r['validation_component_mse_cm2'] for r in h]),label=name)
ax[0].set(xlabel='Epoch',ylabel='Validation component RMSE(cm)',yscale='log',title='Single-seed architecture screening');ax[0].legend(fontsize=8)
d=json.loads((P/'experiments/metrics/graph_v4_evaluation.json').read_text());old=json.loads((P/'experiments/metrics/full_evaluation.json').read_text());val=[old['models']['released_gnn_same503']['mean_nodal_l2_cm'],d['previous_ensemble_mean_nodal_l2_cm'],d['mean_nodal_l2_cm']];ax[1].bar(['Released','Previous ensemble','Updated ensemble'],val,color=['grey','#377eb8','#e66101']);ax[1].set(ylabel='Mean nodal L2 error(cm)',title='Repeated benchmark: same503 geometries');ax[1].tick_params(axis='x',labelsize=8)
fig.savefig(P/'reports/figures/graph_study.png',dpi=180);print('GRAPH FIGURE WRITTEN',flush=True)
