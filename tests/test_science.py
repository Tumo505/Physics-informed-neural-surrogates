import numpy as np
import pytest
import torch
from lv_surrogate.physics.kinematics import tetra_geometry,deformation_gradient,green_lagrange
from lv_surrogate.physics.assembly import unit_energy,unit_internal_force
from lv_surrogate.physics.constitutive import unit_neo_hookean
from lv_surrogate.eval.calibration import conformal_radius,geometry_scores
from lv_surrogate.models.message_passing import GraphSurrogate

def mesh():return torch.tensor([[0.,0,0],[1.,0,0],[0.,1,0],[0.,0,1]],dtype=torch.float64),torch.tensor([[0,1,2,3]])
def test_volume_basis():
 X,c=mesh();inv,v,g=tetra_geometry(X,c);assert v.item()==pytest.approx(1/6);torch.testing.assert_close(g.sum(1),torch.zeros((1,3),dtype=X.dtype));torch.testing.assert_close(X[c].transpose(1,2)@g,torch.eye(3,dtype=X.dtype)[None])
def test_identity_translation():
 X,c=mesh();u=torch.ones_like(X,requires_grad=True)*.25;F=deformation_gradient(X,u,c);torch.testing.assert_close(green_lagrange(F),torch.zeros_like(F));assert abs(unit_energy(X,u,c).item())<1e-13;torch.testing.assert_close(unit_internal_force(X,u,c),torch.zeros_like(u))
def test_rigid_rotation():
 X,c=mesh();R=torch.tensor([[0.,-1,0],[1.,0,0],[0.,0,1]],dtype=X.dtype);F=deformation_gradient(X,X@R.T-X,c);assert unit_neo_hookean(F).abs().max()<1e-13
def test_affine():
 X,c=mesh();A=torch.tensor([[1.1,.1,0],[0,1.05,.02],[0,0,.95]],dtype=X.dtype);F=deformation_gradient(X,X@A.T-X,c);torch.testing.assert_close(F,A[None]);torch.testing.assert_close(green_lagrange(F),(.5*(A.T@A-torch.eye(3,dtype=X.dtype)))[None])
def test_force_gradient_fd():
 X,c=mesh();u=(.03*X).requires_grad_();f=unit_internal_force(X,u,c);h=1e-6
 up=u.detach().clone();um=up.clone();up[1,0]+=h;um[1,0]-=h
 fd=(unit_energy(X,up,c)-unit_energy(X,um,c))/(2*h);assert f[1,0].item()==pytest.approx(fd.item(),rel=1e-7,abs=1e-9)
 # Residual-loss gradient requires second derivatives.
 loss=(f*f).sum();grad=torch.autograd.grad(loss,u)[0]
 def residual_loss(q):q=q.clone().requires_grad_();r=unit_internal_force(X,q,c);return (r*r).sum()
 fd=(residual_loss(up)-residual_loss(um))/(2*h);assert grad[1,0].item()==pytest.approx(fd.item(),rel=1e-6,abs=1e-8)
def test_force_balance_and_perturbation():
 X,c=mesh();u=torch.zeros_like(X,requires_grad=True);assert unit_internal_force(X,u,c).norm()<1e-12
 up=(.04*X).requires_grad_();f=unit_internal_force(X,up,c);assert f.norm()>.01;torch.testing.assert_close(f.sum(0),torch.zeros(3,dtype=X.dtype),atol=1e-12,rtol=0)
def test_bad_jacobians():
 with pytest.raises(ValueError):unit_neo_hookean(torch.diag(torch.tensor([-1.,1,1]))[None])
 X,c=mesh();X[3]=X[0]
 with pytest.raises(ValueError):tetra_geometry(X,c)
def test_conformal_order_statistic():
 assert conformal_radius(np.arange(1,20),.1)==18
 assert np.isinf(conformal_radius([1,2],.05))
 with pytest.raises(ValueError):conformal_radius([np.nan],.1)
 with pytest.raises(ValueError):conformal_radius([], .1)
def test_geometry_score():
 a=np.zeros((2,4,3));b=a.copy();b[0,2,0]=2;np.testing.assert_equal(geometry_scores(a,b),[2,0])
def test_gnn_permutation_equivariance_and_backward():
 torch.manual_seed(11);m=GraphSurrogate(4,2,8,2);x=torch.randn(4,4);g=torch.randn(2);e=torch.randn(4,4);topo=torch.tensor([[0,1],[1,2],[2,3],[3,0]]);y=m(x,g,e,topo);perm=torch.tensor([2,0,3,1]);inv=torch.argsort(perm);yp=m(x[perm],g,e,inv[topo]);torch.testing.assert_close(yp,y[perm]);y.square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
