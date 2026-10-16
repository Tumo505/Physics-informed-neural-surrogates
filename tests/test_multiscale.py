import torch
from lv_surrogate.models.multiscale import MultiscaleSurrogate

def test_full_and_queried_node_predictions_agree():
 torch.manual_seed(2);net=MultiscaleSurrogate(node_dim=5,global_dim=2,width=8,steps=2)
 points=torch.randn(2,12,5);coarse=torch.randn(2,4,5);g=torch.randn(2,2);e=torch.randn(2,6,4);top=torch.tensor([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2]]);parents=torch.arange(12)%4
 full=net(points,coarse,g,e,top,parents);q=torch.tensor([[1,3,8],[2,7,11]])
 partial=net(torch.gather(points,1,q[:,:,None].expand(-1,-1,5)),coarse,g,e,top,parents[q]);torch.testing.assert_close(partial,torch.gather(full,1,q[:,:,None].expand(-1,-1,3)))
 partial.square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())

def test_point_baseline_ignores_graph_inputs():
 torch.manual_seed(2);net=MultiscaleSurrogate(node_dim=5,global_dim=2,width=8,steps=2,graph=False);points=torch.randn(2,3,5);g=torch.randn(2,2)
 a=net(points,None,g,None,None,None);b=net(points,None,g,None,None,None);torch.testing.assert_close(a,b)
