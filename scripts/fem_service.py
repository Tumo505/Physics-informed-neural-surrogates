"""Local exact FE residual/tangent service, including histories and geometry variants."""
from pathlib import Path
import sys,json,traceback
from collections import OrderedDict
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
import numpy as np
control=sys.stdout;sys.stdout=sys.stderr
from lv_surrogate.physics.reference_fem import setup,assign_nodes,assign_history,residual,solve_at_pressure,df,ufl
active=(None,.001)
geo,mat,problem=setup();cohort=np.load(P/'data/processed/cardiac/cohort.npz');scale=float(cohort['force_scale_N']);v2d=cohort['v2d'];N=problem.u_space.dim()
cache=OrderedDict();cache[active]=(geo,mat,problem,scale,v2d,cohort['facets'].copy())
print(json.dumps({'ready':True,'mesh_volume_m3':df.assemble(df.Constant(1.)*df.dx(domain=geo.mesh)),'endo_area_m2':df.assemble(df.Constant(1.)*problem.ds(1))}),file=control,flush=True)
for line in sys.stdin:
 try:
  req=json.loads(line)
  if req.get('quit'):break
  path=req.get('geometry_path');dt=float(req.get('dt',.001));key=(path,dt)
  if key!=active:
   if key in cache:
    geo,mat,problem,scale,v2d,facets=cache[key];cache.move_to_end(key)
   else:
    f=Path(path).resolve() if path else P/'data/processed/cardiac/cohort.npz'
    if not f.is_relative_to(P/'data/processed'):raise ValueError('Geometry must be a local processed artifact')
    with np.load(f) as a:scales=a['geometry_scales'] if 'geometry_scales' in a else None;v2d=a['v2d'].copy();scale=float(a['force_scale_N']);facets=a['facets'].copy()
    geo,mat,problem=setup(dt=dt,geometry_scales=scales);cache[key]=(geo,mat,problem,scale,v2d,facets)
    if len(cache)>32:cache.popitem(last=False)
   N=problem.u_space.dim();active=key
  if 'solve' in req:
   u,record=solve_at_pressure(problem,req['solve']);np.savez(req['response'],u=u);print(json.dumps({'ok':True,'record':record}),file=control,flush=True);continue
  a=np.load(req['request']);u=a['u'];p=float(a['pressure']);problem.parameters['p'].assign(p);assign_nodes(problem,u)
  assign_history(problem,**{k:a[k] for k in ['u_old','v_old','a_old'] if k in a})
  R=residual(problem);r=R.get_local();mask=np.ones(N)
  region=req.get('region','all')
  if region!='all':
   boundary_nodes=np.unique(cohort['facets'] if not path else facets)
   boundary_dofs=v2d.reshape(-1,3)[boundary_nodes].reshape(-1)
   if region=='boundary':mask[:]=0;mask[boundary_dofs]=1
   elif region=='interior':mask[boundary_dofs]=0
   else:raise ValueError('Unknown residual region')
  # Divide by TOTAL dofs: interior + boundary losses equal the full loss.
  loss=float(np.mean(mask*(r/scale)**2));arrays={'residual':r[v2d].reshape(-1,3),'loss':np.array(loss)}
  if req.get('gradient',True):
   if req.get('assembled_vjp',False):
    J=df.assemble(problem._problem._J);weighted=R.copy();weighted.set_local(mask*r);weighted.apply('insert');g=R.copy();J.transpmult(weighted,g)
   else:
    if not hasattr(problem,'_vjp_weights'):
     problem._vjp_weights=df.Function(problem.u_space)
     problem._vjp_form=df.action(df.adjoint(problem._problem._J),problem._vjp_weights)
    problem._vjp_weights.vector().set_local(mask*r);problem._vjp_weights.vector().apply('insert')
    g=df.assemble(problem._vjp_form)
   arrays['gradient']=(2/(N*scale**2))*g.get_local()[v2d].reshape(-1,3)
  np.savez(req['response'],**arrays);print(json.dumps({'ok':True,'loss':loss}),file=control,flush=True)
 except Exception as exc:
  traceback.print_exc();print(json.dumps({'ok':False,'error':str(exc)}),file=control,flush=True)
