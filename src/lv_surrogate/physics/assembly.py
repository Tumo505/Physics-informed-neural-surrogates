"""Internal energy gradient for synthetic unit verification; no cardiac load assembly."""
import torch
from .kinematics import tetra_geometry,deformation_gradient
from .constitutive import unit_neo_hookean

def unit_energy(X,u,cells):
 _,vol,_=tetra_geometry(X,cells);F=deformation_gradient(X,u,cells);return (vol*unit_neo_hookean(F)).sum()

def unit_internal_force(X,u,cells):return torch.autograd.grad(unit_energy(X,u,cells),u,create_graph=True)[0]
