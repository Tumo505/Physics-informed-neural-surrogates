"""Batched multiscale full-field surrogate with stochastic point queries.

Fixed coarse graph comes from supplied author clustering. Coarse graph edges
are computational connections, not volumetric finite elements.
"""
import torch
from torch import nn


def dense(din, hidden, dout):
    return nn.Sequential(nn.Linear(din, hidden), nn.SiLU(),
                         nn.Linear(hidden, dout))


class MultiscaleSurrogate(nn.Module):
    def __init__(self, node_dim=13, global_dim=36, width=64, steps=3, graph=True):
        super().__init__()
        self.graph = graph
        self.width = width
        if graph:
            self.node_encoder = dense(node_dim + global_dim, width, width)
            self.edge_encoder = dense(4, width, width)
            self.messages = nn.ModuleList([dense(3 * width, width, width) for _ in range(steps)])
            self.updates = nn.ModuleList([dense(2 * width, width, width) for _ in range(steps)])
            self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(steps)])
        point_dim = node_dim + global_dim + (width if graph else 0)
        self.decoder = nn.Sequential(nn.Linear(point_dim, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(),
                                     nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 3))

    def encode(self, coarse, globals_, edge_features, topology):
        if not self.graph:
            return None
        globals_tiled = globals_[:, None].expand(-1, coarse.shape[1], -1)
        v = self.node_encoder(torch.cat((coarse, globals_tiled), -1))
        e = self.edge_encoder(edge_features)
        s, r = topology.T
        degree = torch.bincount(r, minlength=coarse.shape[1]).clamp_min(1).to(v.dtype)[None, :, None]
        for message, update, norm in zip(self.messages, self.updates, self.norms):
            m = message(torch.cat((e, v[:, s], v[:, r]), -1))
            agg = torch.zeros_like(v).index_add(1, r, m) / degree
            v = norm(v + update(torch.cat((v, agg), -1)))
            e = e + m
        return v

    def decode(self, points, globals_, latent, parents):
        inputs = [points, globals_[:, None].expand(-1, points.shape[1], -1)]
        if self.graph:
            if parents.ndim == 1:
                local = latent[:, parents]
            else:
                local = torch.gather(latent, 1, parents[..., None].expand(-1, -1, self.width))
            inputs.append(local)
        return self.decoder(torch.cat(inputs, -1))

    def forward(self, points, coarse, globals_, edges, topology, parents):
        return self.decode(points, globals_, self.encode(coarse, globals_, edges, topology), parents)
