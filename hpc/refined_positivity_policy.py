"""Additional subdivision for inconclusive refined-cell proofs; unchanged margin and witnesses."""
import copy

from lv_surrogate.physics.bernstein import BernsteinAudit


def extend_unresolved_depth(audit, depth):
    assert depth == 32 and audit.margin == 1e-9
    restores = []
    for _, child in audit.shards:
        resolve = child.resolve
        deeper = copy.copy(child)
        deeper.max_depth = depth

        def retry(deformation, vertices, resolve=resolve, deeper=deeper):
            try:
                return resolve(deformation, vertices)
            except ValueError as exc:
                if str(exc) != 'Whole-element positivity remains unresolved at maximum depth':
                    raise
            # Call the arithmetic directly: the copied child's instance resolver still
            # contains the depth-20/24 closure and must not be called recursively.
            proof = BernsteinAudit.resolve(deeper, deformation, vertices)
            return dict(proof, subdivision_retry_depth=depth)

        child.resolve = retry
        restores.append((child, resolve))

    def restore():
        for child, resolve in restores:
            child.resolve = resolve

    return restore
