"""SPD rank-local ILU product preconditioner for the scaled normal operator.

C approximates the rank diagonal block of J/S plus a pivot shift.
Apply (C.T*C)^-1 = C^-1*C^-T; the objective and constraints are unchanged.
The product is positive definite even when C is nonsymmetric.
"""
import time
import numpy as np
from scipy.sparse import csr_matrix,diags,eye
from scipy.sparse.linalg import spilu


class LocalNormalPC:
    def __init__(self,jacobian,scaling,mu):
        started=time.perf_counter()
        block=jacobian.getDiagonalBlock()
        indptr,indices,values=block.getValuesCSR()
        n=len(scaling)
        B=csr_matrix((values,indices,indptr),shape=(n,n))@diags(1/scaling)
        self.factor=None;errors=[]
        for multiplier in (1.,10.):
            shift=multiplier*np.sqrt(mu)
            try:
                self.factor=spilu((B+shift*eye(n,format='csr')).tocsc(),
                                  drop_tol=1e-3,fill_factor=3.,permc_spec='COLAMD')
                break
            except RuntimeError as exc:errors.append(str(exc))
        if self.factor is None:raise RuntimeError('Local normal preconditioner factorization failed: '+'; '.join(errors))
        self.detail=dict(kind='local_ilu_normal',local_dofs=n,local_matrix_nnz=B.nnz,
            local_factor_nnz=self.factor.L.nnz+self.factor.U.nnz,pivot_shift=shift,
            setup_seconds=time.perf_counter()-started)

    def apply(self,pc,x,y):
        first=self.factor.solve(np.asarray(x.getArray(readonly=True)),trans='T')
        y.getArray()[:]=self.factor.solve(first)

    def close(self):self.factor=None
