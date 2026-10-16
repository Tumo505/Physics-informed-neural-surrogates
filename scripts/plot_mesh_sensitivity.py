"""Plot measured two-level FE discretization sensitivity."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
d=json.loads((P/'experiments/metrics/mesh_sensitivity.json').read_text())
if not d.get('complete'):raise RuntimeError('Mesh study incomplete')
cases=d['cases'];p=[c['pressure_Pa']/1000 for c in cases]
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
y=[100*c['relative_displacement_volume_L2_difference'] for c in cases]
ax.bar(p,y,width=1.6,color='#377eb8')
for x,v in zip(p,y):ax.text(x,v+.06,f'{v:.3f}%',ha='center')
ax.set(xlabel='Startup pressure (kPa)',ylabel='Relative displacement volume-L2 difference (%)',ylim=(0,3.5),xticks=p,title=f'{d["meshes"]["coarse"]["tetrahedra"]:,} → {d["meshes"]["refined"]["tetrahedra"]:,} tetrahedra; same polyhedral domain')
fig.savefig(P/'reports/figures/mesh_sensitivity.png',dpi=180)
print('mesh sensitivity figure written')
