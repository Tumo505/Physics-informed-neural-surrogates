"""Regularization update for the state 551 recovery experiment."""


def retry_linear_accuracy(attempt, attempts, *, rtol, limit):
    """Tighten the internal solve, retaining the original true-residual gate."""
    for reduction in (.1, .01):
        last = attempts[-1]
        if last['qualified'] or last['reason'] <= 0 or last['true_residual_N'] is None:
            break
        attempts.append(attempt(False, rtol * reduction, min(1e-12, limit * .05)))
    return attempts


def update_shift(mu, good_full_steps, *, accepted, full_step, ratio, minimum, maximum):
    """Require two good full steps at the same shift before halving it."""
    if not accepted:
        return min(maximum, mu * 10), 0
    if not full_step or ratio is None or ratio < .25:
        return min(maximum, mu * 4), 0
    if ratio <= .75:
        return mu, 0
    if good_full_steps == 0:
        return mu, 1
    return max(minimum, mu * .5), 0
