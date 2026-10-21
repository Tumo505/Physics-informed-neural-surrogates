"""Adaptive algebraic homotopy; intermediate roots are never physical samples."""
import time
import numpy as np
from .reference_fem import residual


def recover_equation(problem, coefficients, start_step, pressures, protocol,
                     result, persist, kinematics):
    """Connect the preceding accepted equation to the exact next equation.

    The final histories, dt, pressure, material and weak form are unchanged.
    Failed subproblems roll back to the last accepted homotopy root. No
    generalized-alpha history update is permitted during this procedure.
    """
    options = protocol['history_continuation']
    assert start_step >= 2
    dt = float(problem.parameters['dt'])
    beta, gamma = float(problem._beta), float(problem._gamma)
    current = {k: getattr(problem, k).vector().get_local().copy()
               for k in ['u_old', 'v_old', 'a_old']}
    previous_u = np.array(coefficients[start_step-2])
    previous_a = ((beta-gamma)*current['a_old']
        -(current['u_old']-previous_u-dt*current['v_old'])/dt**2)/(.5-gamma+beta)
    previous = dict(u_old=previous_u, a_old=previous_a,
                    v_old=current['v_old']-dt*((1-gamma)*previous_a+gamma*current['a_old']))
    def assign(history, pressure, u):
        for name, values in history.items():
            f = getattr(problem, name)
            f.vector().set_local(values); f.vector().apply('insert')
        problem.parameters['p'].assign(float(pressure))
        problem.u.vector().set_local(u); problem.u.vector().apply('insert')
    root = current['u_old'].copy()
    assign(previous, pressures[start_step-1], root)
    prior_force = float(residual(problem).norm('l2'))
    result['history_continuation'] = dict(previous_equilibrium_residual_N=prior_force,
        records=[], last_accepted_fraction=0., physical_samples=False)
    persist()
    assert np.isfinite(prior_force) and prior_force < 1e-6, 'Previous equation replay failed'
    fraction, increment = 0., float(options['initial_increment'])
    settings = problem.solver._solver.parameters
    original_maximum = settings['maximum_iterations']
    try:
        for attempt in range(options['maximum_attempts']):
            target = min(1., fraction+increment)
            history = {k: (1-target)*previous[k]+target*current[k] for k in current}
            pressure = (1-target)*pressures[start_step-1]+target*pressures[start_step]
            assign(history, pressure, root)
            settings['maximum_iterations'] = options['maximum_newton_per_increment']
            tick = time.perf_counter()
            failure = None
            try:
                problem.solver.solve()
            except RuntimeError as error:
                failure = str(error)
                if result.get('budget_exhausted'):
                    raise
            snes = problem.solver._snes
            force = float(residual(problem).norm('l2'))
            candidate = problem.u.vector().get_local()
            finite = np.isfinite(force) and np.isfinite(candidate).all()
            endpoint = stress = None
            if finite:
                endpoint, _ = kinematics.audit(candidate)
                stress, _ = kinematics.audit(.4*history['u_old']+.6*candidate)
            accepted = bool(failure is None and snes.getConvergedReason()>0 and finite
                and force < protocol['absolute_tolerance_N'] and min(endpoint, stress)>0)
            record = dict(attempt=attempt, fraction=target, increment=increment,
                accepted=accepted, residual_N=force if np.isfinite(force) else None,
                SNES_reason=int(snes.getConvergedReason()),
                iterations=int(snes.getIterationNumber()), seconds=time.perf_counter()-tick,
                minimum_endpoint_sampled_detF=endpoint, minimum_stress_sampled_detF=stress,
                failure=failure)
            result['history_continuation']['records'].append(record)
            if accepted:
                root = candidate.copy(); fraction = target
                result['history_continuation']['last_accepted_fraction'] = fraction
                increment = min(options['maximum_increment'], increment*options['growth_factor'])
            else:
                increment *= .5
            persist()
            print('HISTORY CONTINUATION', target, accepted, force, flush=True)
            if fraction == 1.:
                # At lambda=1 this is the unchanged original algebraic equation.
                return root
            if increment < options['minimum_increment']:
                raise RuntimeError('Adaptive homotopy exhausted its declared minimum increment')
        raise RuntimeError('Adaptive homotopy exhausted its declared attempt count')
    finally:
        settings['maximum_iterations'] = original_maximum
        assign(current, pressures[start_step], root)
