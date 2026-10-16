"""Linear tetrahedral finite-strain kinematics, reference coordinates only."""
import torch

def tetra_geometry(X,cells):
 xe=X[cells];dm=(xe[:,1:]-xe[:,:1]).transpose(1,2);det=torch.linalg.det(dm)
 if not torch.isfinite(dm).all() or (det.abs()<1e-14).any():raise ValueError('degenerate/nonfinite reference tetrahedra')
 inv=torch.linalg.inv(dm);grads=torch.cat((-inv.sum(1,keepdim=True),inv),1)
 return inv,det.abs()/6,grads

def deformation_gradient(X,u,cells):
 inv,vol,grads=tetra_geometry(X,cells);xe=(X+u)[cells];F=(xe[:,1:]-xe[:,:1]).transpose(1,2)@inv
 if not torch.isfinite(F).all():raise ValueError('nonfinite deformation')
 return F

def green_lagrange(F):
 return .5*(F.transpose(-1,-2)@F-torch.eye(3,dtype=F.dtype,device=F.device))
