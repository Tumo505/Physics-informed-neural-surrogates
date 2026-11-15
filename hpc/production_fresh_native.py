"""Select the qualified fresh-Jacobian solver for production trajectory segments."""
import os
import sys

from fresh_reference_native import solve_fresh, worker8_native


if __name__ == '__main__':
    assert os.environ.get('PBS_JOBID'), 'Use PBS compute jobs'
    assert sys.argv[1] == 'generate'
    stop = int(sys.argv[sys.argv.index('--qualification-stop') + 1])
    assert 554 <= stop <= 1600
    from lv_surrogate.physics import admissible_newton_regularized_forcing

    admissible_newton_regularized_forcing.solve = solve_fresh
    worker8_native.main()
