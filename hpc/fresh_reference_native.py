"""Run the frozen reference generator with the registered fresh-Jacobian solver."""
import os
import sys

import worker8_native


def solve_fresh(problem, adapter, history, audit, options, trace, persist):
    from lv_surrogate.physics.admissible_newton_gentle_shrink import solve

    assert options['custom_absolute_tolerance_N'] == 1e-9
    assert 0 < options['wall_budget_seconds'] <= 1800
    trace['solver_policy'] = 'fresh_jacobian_two_good_steps'
    solve(problem, adapter, history, audit, options, trace, persist,
          policy='two_good_steps', reuse_jacobian=False, reuse_gamg=False,
          late_forcing_floor=1e-5)
    assert 0 <= trace['custom_force_N'] < 1e-9
    persist()


if __name__ == '__main__':
    assert os.environ.get('PBS_JOBID'), 'Use PBS compute jobs'
    assert sys.argv[1] == 'generate'
    assert sys.argv[sys.argv.index('--qualification-stop') + 1] in ('553', '554')
    from lv_surrogate.physics import admissible_newton_regularized_forcing

    # The generator imports this binding when worker8_native starts it.
    admissible_newton_regularized_forcing.solve = solve_fresh
    worker8_native.main()
