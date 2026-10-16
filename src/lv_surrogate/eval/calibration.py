"""Geometry-level simultaneous displacement balls with split-conformal correction."""
import math
import numpy as np

def conformal_radius(scores,alpha):
 scores=np.asarray(scores,dtype=float)
 if not 0<alpha<1 or scores.ndim!=1 or len(scores)==0 or not np.all(np.isfinite(scores)) or np.any(scores<0):raise ValueError('invalid calibration scores/alpha')
 k=math.ceil((len(scores)+1)*(1-alpha))
 return float('inf') if k>len(scores) else float(np.partition(scores,k-1)[k-1])

def geometry_scores(pred,true):return np.linalg.norm(np.asarray(pred)-np.asarray(true),axis=-1).max(axis=-1)
