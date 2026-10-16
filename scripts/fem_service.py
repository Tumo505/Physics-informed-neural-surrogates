"""Local file/stdio assembly bridge. All scientific payloads remain local NPZ.

Returns full residual and the exact VJP of normalized residual loss using the
trusted assembled tangent. This includes derivative of follower-pressure loads.
"""
from pathlib import Path
import sys,json,traceback
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
import numpy as np
# Imported solver may print; preserve a dedicated output stream for control JSON.
control=sys.stdout;sys.stdout=sys.stderr
from lv_surrogate.physics.reference_fem import setup,assign_nodes,residual,solve_at_pressure,df
geo,mat,problem=setup();cohort=np.load(P/'data/processed/cardiac/cohort.npz');scale=float(cohort['force_scale_N']);v2d=cohort['v2d'];N=problem.u_space.dim()
print(json.dumps({'ready':True,'mesh_volume_m3':df.assemble(df.Constant(1.)*df.dx(domain=geo.mesh)),'endo_area_m2':df.assemble(df.Constant(1.)*problem.ds(1))}),file=control,flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('quit'):break
  if 'solve' in req:
   u,record=solve_at_pressure(problem,req['solve']);np.savez(req['response'],u=u);print(json.dumps({'ok':True,'record':record}),file=control,flush=True);continue
  a=np.load(req['request']);u=a['u'];p=float(a['pressure']);problem.parameters['p'].assign(p);assign_nodes(problem,u)
  R=residual(problem);r=R.get_local();loss=float(np.mean((r/scale)**2));arrays={'residual':r[v2d].reshape(-1,3),'loss':np.array(loss)}
  if req.get('gradient',True):
   J=df.assemble(problem._problem._J);g=R.copy();J.transpmult(R,g);arrays['gradient']=(2/(N*scale**2))*g.get_local()[v2d].reshape(-1,3)
  np.savez(req['response'],**arrays);print(json.dumps({'ok':True,'loss':loss}),file=control,flush=True)
 except Exception as exc:
  traceback.print_exc();print(json.dumps({'ok':False,'error':str(exc)}),file=control,flush=True)
