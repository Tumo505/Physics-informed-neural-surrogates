"""Acceptance and runtime gates for a four-substep solver study, never a production chain."""
import time
from registry import artifact, atomic, sha


def initialize(run, failed):
    import launch_refined_production as production
    source=production.load(failed/'plan.json')
    assert source['coupled_newton'].get('dynamic_elements'), 'Require the failed expanding-constraint production run'
    production.initialize_expanding_retry(run,failed)
    plan=production.load(run/'plan.json')
    state=production.load(failed/'state.json')
    trace=production.load(failed/f'factor-4/substep-021/trace-slice-{state["slice"]:03d}.json')
    seeds=set(plan['coupled_newton']['initial_elements'])
    seeds.update(trace.get('coupled_elements',[]))
    assert len(seeds)<=plan['coupled_newton']['maximum_elements']
    plan['coupled_newton'].update(initial_elements=sorted(seeds),residual_minimizing=True)
    plan.update(stop_state=556,requested_substeps=24,
        solver_study=dict(start_substep=20,stop_substep=24,wall_budget_seconds=12*3600,
                          maximum_newton_directions_per_substep=60,automatic_successor=False),
        scope='Bounded constrained residual-minimization study on refined substeps 21-24. '
              'All native force, history and whole-element positivity checks retained. '
              'Success authorizes review, not automatic full-trajectory production.')
    pressure=production.load(run/'pressure.json')
    schedule=pressure['factors']['4']
    for key in ('endpoint_times','pressure_times','pressures'):
        schedule[key]=schedule[key][:24]
    atomic(run/'pressure.json',pressure)
    plan['frozen'][str(run/'pressure.json')]=sha(run/'pressure.json')
    artifact(production.ROOT,run,run/'pressure.json','Unchanged forcing restricted to the bounded study interval')
    atomic(run/'plan.json',plan)
    artifact(production.ROOT,run,run/'plan.json','Bounded solver study with explicit acceptance and runtime gates')
    state=production.load(run/'state.json')
    state['requested_substeps']=24
    atomic(run/'state.json',state)


def finish(run, started, stop_reason, worker_exit):
    import launch_refined_production as production
    plan=production.load(run/'plan.json');study=plan['solver_study']
    rows=production.production_records(run,verify_all_checkpoints=True)
    new=[row for row in rows if row['substep']>study['start_substep']]
    passed=len(rows)==study['stop_substep'] and worker_exit==0 and stop_reason=='target_reached'
    times=[row['solve_wall_seconds'] for row in new]
    elapsed=max(0.,time.time()-started)
    result=dict(complete=True,passed=passed,stop_reason=stop_reason,worker_exit=worker_exit,
        accepted_before=study['start_substep'],accepted_after=len(rows),
        target_substeps=list(range(study['start_substep']+1,study['stop_substep']+1)),
        failing_substep_accepted=any(row['substep']==21 for row in new),
        subsequent_substeps_accepted=sum(row['substep']>21 for row in new),
        method='constrained_residual_minimization',
        elapsed_wall_seconds=elapsed,accepted_substep_solve_seconds=times,
        mean_accepted_substep_solve_seconds=sum(times)/len(times) if times else None,
        final_force_N=new[-1]['native_force_N'] if new else None,
        automatic_production_authorized=False,original_reference_steps=554,
        checkpoint_sha256=rows[-1]['checkpoint_sha256'],scope=plan['scope'])
    summary_path=run/'factor-4/summary.json'
    if summary_path.exists():
        result['worker_failure']=production.load(summary_path).get('failure')
    trace_paths=sorted((run/'factor-4').glob('substep-*/trace-slice-*.json'))
    if trace_paths:
        latest=trace_paths[-1]
        trace=production.load(latest)
        steps=trace.get('custom_newton_steps',[])
        result.update(latest_trace=str(latest.relative_to(run)),latest_newton_iterations=len(steps),
                      latest_force_N=trace.get('custom_force_N',steps[-1]['initial_force_N'] if steps else None))
    atomic(run/'result.json',result)
    artifact(production.ROOT,run,run/'result.json','Bounded solver acceptance and runtime report')
    return 'study_passed' if passed else 'study_stopped'
