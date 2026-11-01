"""Independent serial native replay of every accepted field, including the inherited chain."""
import argparse,json,sys,time,mmap
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,df,residual
from lv_surrogate.physics.quadratic_solver import make_quadratic_problem
from lv_surrogate.physics.quadratic import QuadraticKinematics
from lv_surrogate.physics.durable_reference import sha,atomic_json
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--output',required=True);ap.add_argument('--require-complete',action='store_true');args=ap.parse_args();out=P/args.output;assert not out.exists(),'Never retry an existing independent audit'
p=json.loads((P/args.protocol).read_text());cfg=json.loads((P/p['config']).read_text());root=P/'data/processed'/cfg['name'];mp=root/'manifest.json';m=json.loads(mp.read_text());count=m['completed_steps'];assert count>0 and (not args.require_complete or m['complete'] and count==m['n_steps'])
for f,h in p['frozen_source_sha256'].items():assert sha(P/f)==h
assert m['config_sha256']==sha(P/p['config']);data=np.load(root/'inputs.npz');fields=np.load(root/'displacement_coefficients.npy',mmap_mode='r');proofs=json.loads((root/'positivity.json').read_text());assert len(proofs)==count and len(m['steps'])==count
for i,x in enumerate(proofs):assert x['step']==i and all(x[s]['passed'] and x[s]['minimum_Bernstein_bound']>1e-9 for s in ['endpoint','stress'])
for x in m.get('lineage',[]):
 for k,h in [('protocol','protocol_sha256'),('audit','audit_sha256')]:assert sha(P/x[k])==x[h]
 a=json.loads((P/x['audit']).read_text());assert a['passed']
geo,material,base=setup(dt=cfg['dt_s'],refinement_level=cfg['refinement_level'],geometry_scales=cfg['geometry_scales']);df.parameters['form_compiler']['quadrature_degree']=cfg['quadrature_degree'];params=dict(base.parameters);params['function_space']='P_2';problem=make_quadratic_problem(geo,material,params,'gamg')
np.testing.assert_allclose(problem.u_space.tabulate_dof_coordinates(),data['dof_coordinates'],atol=1e-14,rtol=0);np.testing.assert_array_equal(geo.mesh.cells(),data['cells'])
kin=QuadraticKinematics(problem.u_space,data['qpoints'],data['qweights']);np.testing.assert_array_equal(kin.maps,data['canonical_cell_dofs']);records=[];result=dict(complete=False,passed=False,accepted_steps=count,full_cycle=m['complete'],config_sha256=m['config_sha256'],protocol_sha256=sha(P/args.protocol),audited_manifest_sha256=sha(mp),executed_source_sha256=sha(Path(__file__)),records=records,scope='Independent serial native exact-history replay/force at EVERY accepted state,canonical quadrature versus UFL norms on fixed frames,whole-element proof/source linkage. Strict1e-8N; no numerical-discretization or continuum truth claim. Original lost individual SNES reasons remain explicitly lost and traced,never synthesized.')
bary=kin.quad_bary;basis=np.concatenate((bary*(2*bary-1),np.stack([4*bary[:,i]*bary[:,j] for i,j in kin.edges],axis=1)),axis=1)
selected=set(np.linspace(0,count-1,min(count,6),dtype=int));started=time.perf_counter()
try:
 for name in ['u','u_old','v_old','a_old']:getattr(problem,name).vector().zero()
 for i in range(count):
  current=fields[i];problem.u.vector().set_local(current);problem.u.vector().apply('insert');problem.parameters['p'].assign(float(data['pressure'][i]));force=float(residual(problem).norm('l2'));row=dict(step=i,time_s=float(data['time'][i]),native_residual_N=force if np.isfinite(force) else None);records.append(row)
  assert np.isfinite(force) and force<1e-8,f'Independent native force failed at{ i }: { force }'
  rec=m['steps'][i];assert rec['step']==i and rec['residual_l2_N']<1e-8
  if rec['SNES_reason'] is None:
   provenance=rec['native_acceptance_provenance'];assert rec['native_reason_record_lost']
   for k,h in [('failure_path','failure_sha256'),('independent_audit','independent_audit_sha256'),('accepted_fields','accepted_fields_sha256')]:assert sha(P/provenance[k])==provenance[h]
   failure=json.loads((P/provenance['failure_path']).read_text());assert failure['persisted_checkpoint_steps']<=i<failure['accepted_steps_before_failure']
   np.testing.assert_array_equal(current,np.load(P/provenance['accepted_fields'],mmap_mode='r')[provenance['audit_record']])
  else:assert rec['SNES_reason']>0
  if i in selected:
   norm=0.
   for start in range(0,len(kin.maps),8192):
    stop=min(start+8192,len(kin.maps));values=np.einsum('qk,ekc->eqc',basis,current[kin.maps[start:stop]]);norm+=float(np.einsum('e,q,eqc,eqc->',kin.measure[start:stop],kin.weights,values,values))
   native=float(df.assemble(df.inner(problem.u,problem.u)*df.dx(metadata={'quadrature_degree':cfg['quadrature_degree']})));np.testing.assert_allclose(norm,native,rtol=1e-11,atol=1e-24);row['independent_norm_relative_error']=abs(norm-native)/max(native,1e-30)
  problem._update_fields()
  if (i+1)%16==0 or i+1==count:
   result['checked_steps']=i+1;result['wall_seconds']=time.perf_counter()-started;atomic_json(out,result);print('INDEPENDENT ALL-STATE FORCE',i+1,'/',count,force,flush=True)
   if hasattr(fields._mmap,'madvise'):fields._mmap.madvise(mmap.MADV_DONTNEED)
 saved=np.load(root/'restart.npz');errors={}
 for name in ['u','u_old','v_old','a_old']:
  actual=getattr(problem,name).vector().get_local();np.testing.assert_allclose(actual,saved[name],rtol=1e-7,atol=1e-10);errors[name]=float(np.linalg.norm(actual-saved[name])/max(np.linalg.norm(saved[name]),1e-30))
 result.update(complete=True,passed=True,final_checkpoint_history_relative_errors=errors,checkpoint_sha256=sha(root/'restart.npz'),positivity_sha256=sha(root/'positivity.json'),wall_seconds=time.perf_counter()-started)
 if count==m['seed_steps']:
  m['prefix_independent_native_audit_passed']=True;m['prefix_native_audit']=args.output;atomic_json(mp,m)
except Exception as exc:result.update(complete=True,passed=False,failure=str(exc),wall_seconds=time.perf_counter()-started)
finally:atomic_json(out,result)
assert result['passed'],result.get('failure');print('ALL-STATE INDEPENDENT NATIVE AUDIT PASSED',count,flush=True)
