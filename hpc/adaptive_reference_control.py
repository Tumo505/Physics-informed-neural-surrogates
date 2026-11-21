"""Transactional step doubling and quadrature checks for a separate reference pilot."""
from dataclasses import dataclass
import numpy as np


class TrialRejected(RuntimeError):pass
class BudgetExpired(TimeoutError):pass


@dataclass
class Snapshot:
    time: float
    u: np.ndarray
    v: np.ndarray
    a: np.ndarray


def field_error(left,right,*,rtol=1e-3,atol=(1e-7,1e-5,1e-2)):
    """Maximum over every dof: a localized defect cannot disappear in global L2."""
    values={}
    for key,tol in zip(('u','v','a'),atol,strict=True):
        x,y=getattr(left,key),getattr(right,key)
        assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        values[key]=float(np.max(abs(x-y)/(tol+rtol*np.maximum(abs(x),abs(y))),initial=0))
    return max(values.values()),values


def gradient_error(left,right,*,atol=1e-3,rtol=1e-3):
    x,y=np.asarray(left),np.asarray(right)
    assert x.shape==y.shape and x.shape[-2:]==(3,3)
    if not np.isfinite(x).all() or not np.isfinite(y).all():return float('inf')
    error=np.linalg.norm(x-y,axis=(-2,-1))
    scale=atol+rtol*np.maximum(np.linalg.norm(x,axis=(-2,-1)),np.linalg.norm(y,axis=(-2,-1)))
    return float(np.max(error/scale,initial=0))


def joint_error(left,right,local_fields,policy):
    error,fields=field_error(left,right,rtol=policy['field_rtol'],atol=policy['field_atol'])
    local=gradient_error(local_fields(left),local_fields(right),atol=policy['gradient_atol'],rtol=policy['gradient_rtol'])
    return max(error,local),dict(owned_dof_scaled_errors=fields,hotspot_gradient_scaled_error=local)


def quality_error(endpoint,stress,exponent,policy):
    e,s=np.asarray(endpoint),np.asarray(stress)
    if not np.isfinite(e).all() or not np.isfinite(s).all():return float('inf')
    je,js=np.linalg.det(e),np.linalg.det(s)
    if np.any(je<=0) or np.any(js<=0):return float('inf')
    logarithmic=float(np.max(abs(np.log(je)-np.log(js)),initial=0))/policy['maximum_log_j_stage_change']
    ce=np.sum(e*e,axis=(-2,-1));cs=np.sum(s*s,axis=(-2,-1))
    delta=exponent*abs(je**(-2/3)*ce-js**(-2/3)*cs)
    isotropic=float(np.max(delta,initial=0))/policy['maximum_isotropic_stage_change']
    return max(logarithmic,isotropic)


def attempt(start,dt,advance,local_fields,stage_quality,policy,deadline):
    """Five tentative solves, all from the same accepted histories.

    Full vs two half steps controls time error. Two half steps at the higher
    quadrature control quadrature error. Only the higher-quadrature pair may
    be committed, after ALL gates pass. No tentative state overwrites start.
    """
    for value in (start.u,start.v,start.a):value.flags.writeable=False
    deadline();full=advance(start,dt,policy['quadrature'][0],'full')
    deadline();low_mid=advance(start,dt/2,policy['quadrature'][0],'low-half-1')
    deadline();low=advance(low_mid,dt/2,policy['quadrature'][0],'low-half-2')
    deadline();high_mid=advance(start,dt/2,policy['quadrature'][1],'high-half-1')
    deadline();high=advance(high_mid,dt/2,policy['quadrature'][1],'high-half-2')
    for snapshot in (full,low,high):assert abs(snapshot.time-start.time-dt)<1e-12
    temporal,t_detail=joint_error(full,low,local_fields,policy)
    quadrature,q_detail=joint_error(low,high,local_fields,policy)
    quality=max(stage_quality(start,high_mid),stage_quality(high_mid,high))
    score=max(temporal,quadrature,quality)
    finite=lambda x:float(x) if np.isfinite(x) else None
    report=dict(quadrature_degrees=list(policy['quadrature']),temporal_error=temporal,quadrature_error=quadrature,stage_resolution_error=finite(quality),
                temporal_detail=t_detail,quadrature_detail=q_detail,maximum_scaled_error=finite(score))
    return score<=1,(high_mid,high),report


def integrate(start,dt,advance,local_fields,stage_quality,policy,deadline,commit,reject):
    policy=dict(policy)
    accepted=0;attempts=0;good=0
    while start.time<policy['end_time']-1e-13:
        deadline()
        if attempts>=policy['maximum_attempts']:raise TrialRejected('Adaptive attempt budget exhausted')
        h=min(dt,policy['end_time']-start.time)
        attempts+=1
        try:
            passed,pair,report=attempt(start,h,advance,local_fields,stage_quality,policy,deadline)
        except BudgetExpired:raise
        except TrialRejected as exc:
            passed=False;pair=None;report=dict(solve_failure=str(exc))
        if not passed:
            reject(start,h,report);good=0
            quality=report.get('stage_resolution_error',0)
            if report.get('quadrature_error',0)>max(1.,report.get('temporal_error',0),quality if quality is not None else float('inf')):
                ladder=policy.get('quadrature_ladder',policy['quadrature'])
                high=ladder.index(policy['quadrature'][1])
                if high+1>=len(ladder):
                    raise TrialRejected('Quadrature error remains above tolerance at the highest configured degree')
                policy['quadrature']=[ladder[high],ladder[high+1]]
                continue
            if h/2<policy['minimum_dt']-1e-15:
                raise TrialRejected('Minimum time step reached without a qualified advance')
            dt=h/2;continue
        # The durable commit callback must succeed before changing the accepted state.
        commit(start,pair,h,report)
        start=pair[1];accepted+=2
        good=good+1 if report['maximum_scaled_error']<.125 else 0
        dt=min(policy['maximum_dt'],h*2) if good>=2 else h
        if good>=2:good=0
    return start,dict(accepted_physical_substeps=accepted,attempts=attempts)
