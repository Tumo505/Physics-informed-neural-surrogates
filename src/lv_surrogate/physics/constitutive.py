"""Nondimensional compressible neo-Hookean UNIT TEST model, NOT cardiac material."""
import torch

def unit_neo_hookean(F,mu=2.,lam=3.):
 J=torch.linalg.det(F)
 if not torch.isfinite(J).all() or (J<=0).any():raise ValueError('nonpositive deformation determinant')
 logJ=J.log();return .5*mu*((F*F).sum((-1,-2))-3)-mu*logJ+.5*lam*logJ**2
