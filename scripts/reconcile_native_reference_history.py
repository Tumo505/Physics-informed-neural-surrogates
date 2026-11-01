"""Read-only native history reconciliation; preserve the failed audit and original checkpoint."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'src'))
from lv_surrogate.physics.reference_fem import setup,df,residual
from lv_surrogate.physics.quadratic_solver import make_quadratic_problem
from lv_surrogate.physics.durable_reference import sha,atomic_json
ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args();pp=P/args.protocol;p=json.loads(pp.read_text());out=P/p['output'];assert not out.exists() and not(P/p['canonical_checkpoint']).exists()
for f,h in p['source_sha256'].items():assert sha(P/f)==h
root=P/'data/processed'/p['reference'];m=json.loads((root/'manifest.json').read_text());cfg=m['config'];a=json.loads((P/p['failed_audit']).read_text());assert a['complete'] and not a['passed'] and len(a['records'])==m['completed_steps']==458
assert 'Not equal to tolerance' in a['failure'];assert all(x['native_residual_N'] is not None and x['native_residual_N']<1e-8 for x in a['records']);assert all(x.get('independent_norm_relative_error',0)<1e-11 for x in a['records'])
fields=np.load(root/'displacement_coefficients.npy',mmap_mode='r');data=np.load(root/'inputs.npz');original=np.load(root/'restart.npz');proofs=json.loads((root/'positivity.json').read_text());assert len(proofs)==458
for x in proofs:assert all(x[s]['passed'] and x[s]['minimum_Bernstein_bound']>1e-9 for s in ['endpoint','stress'])
geo,material,base=setup(dt=cfg['dt_s'],refinement_level=cfg['refinement_level'],geometry_scales=cfg['geometry_scales']);df.parameters['form_compiler']['quadrature_degree']=cfg['quadrature_degree'];params=dict(base.parameters);params['function_space']='P_2';problem=make_quadratic_problem(geo,material,params,'gamg');np.testing.assert_allclose(problem.u_space.tabulate_dof_coordinates(),data['dof_coordinates'],rtol=0,atol=1e-14)
selected=set(np.linspace(0,457,8,dtype=int));forces=[];digest=hashlib.sha256();started=time.perf_counter()
for name in ['u','u_old','v_old','a_old']:getattr(problem,name).vector().zero()
for i in range(458):
 u=fields[i];digest.update(np.asarray(u).tobytes());problem.u.vector().set_local(u);problem.u.vector().apply('insert')
 if i in selected:
  problem.parameters['p'].assign(float(data['pressure'][i]));force=float(residual(problem).norm('l2'));assert np.isfinite(force) and force<1e-8;forces.append(dict(step=i,native_residual_N=force))
 problem._update_fields()
 if (i+1)%16==0:print('NATIVE RECONCILIATION',i+1,flush=True)
state={k:getattr(problem,k).vector().get_local() for k in ['u','u_old','v_old','a_old']};np.testing.assert_array_equal(state['u_old'],fields[457]);errors={}
for k in state:
 diff=state[k]-original[k];errors[k]=dict(maximum_absolute_difference=float(np.max(abs(diff))),relative_L2_difference=float(np.linalg.norm(diff)/max(np.linalg.norm(original[k]),1e-30)))
 # Diagnose roundoff-scale history disagreement; do not waive the original pointwise check.
 assert errors[k]['relative_L2_difference']<1e-9 and errors[k]['maximum_absolute_difference']<1e-7
cp=P/p['canonical_checkpoint'];np.savez_compressed(cp,completed_steps=458,**state)
result=dict(complete=True,passed=True,physical_step_generated=False,original_audit_passed=False,original_pointwise_tolerance_unchanged=True,all_458_previous_native_force_checks_passed=True,source_prefix_fields_sha256=digest.hexdigest(),native_replay_checkpoint_sha256=sha(cp),original_checkpoint_sha256=sha(root/'restart.npz'),failed_audit_sha256=sha(P/p['failed_audit']),positivity_sha256=sha(root/'positivity.json'),protocol_sha256=sha(pp),source_sha256=sha(Path(__file__)),history_differences=errors,repeated_independent_force_checks=forces,wall_seconds=time.perf_counter()-started,scope='Original full-seed audit remainsFAILED at pointwisehistorycomparison;all458nativeforces/normsandwhole-elementproofs passed. Separate prospective canonicalhistorycandidate serialized from actualserialnative_update_fields replayofunchanged458physicalfields,notNumPyrecurrence,journalrepairornewphysicalsolve. Originalcheckpoint/fields/SNESrecords retained. Candidatecannotbeuseduntilnewseparateprotocol/nativeallstateauditpassesunchangedpointwisetolerance;no relaxationorfullreference/studyclaim.')
atomic_json(out,result);print('NATIVE HISTORY CANDIDATE QUALIFIED FOR SEPARATE REVALIDATION',errors,flush=True)
