"""Independent physical P2 displacement norm in canonical tetrahedral coordinates."""
import itertools
import numpy as np

def squared_norm(data,u,chunk=8192):
 X=data['coords'][data['cells']];measure=np.abs(np.linalg.det(np.stack([X[:,i]-X[:,0] for i in [1,2,3]],axis=-1)));q=data['qpoints'];b=np.column_stack([1-q.sum(axis=1),q]);edges=list(itertools.combinations(range(4),2));basis=np.concatenate([b*(2*b-1),np.stack([4*b[:,i]*b[:,j] for i,j in edges],axis=1)],axis=1);maps=data['canonical_cell_dofs'];result=0.
 for start in range(0,len(maps),chunk):
  stop=min(start+chunk,len(maps));values=np.einsum('qk,ekc->eqc',basis,u[maps[start:stop]]);result+=float(np.einsum('e,q,eqc,eqc->',measure[start:stop],data['qweights'],values,values))
 return result
