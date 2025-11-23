import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajEncoder(nn.Module):
    """
    Encode per-node per-time features -> per-sample embedding.
    Input:
      traj_feats: [B, N, T, D]
      traj_mask: [B, N, T] (0/1)
    We pool nodes and time after temporal encoding to obtain [B, emb_dim].
    """
    def __init__(self, in_dim=3, hidden=128, nhead=4, nlayer=1, out_dim=256):
        super().__init__()
        self.in_fc = nn.Linear(in_dim, hidden)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
        self.node_fc = nn.Linear(hidden, hidden)
        self.out_fc = nn.Linear(hidden, out_dim)

    def forward(self, traj_feats, traj_mask):
        # traj_feats: [B, N, T, D]
        B, N, T, D = traj_feats.shape
        x = self.in_fc(traj_feats)  # [B,N,T,H]
        # collapse nodes: process per node sequence
        # reshape to [T, B*N, H] for transformer
        x = x.permute(2,0,1,3).reshape(T, B*N, -1)
        x = self.temporal_encoder(x)  # [T, B*N, H]
        x = x.reshape(T, B, N, -1).permute(1,2,0,3)  # [B,N,T,H]
        # pool time weighted by traj_mask
        mask = traj_mask.unsqueeze(-1)  # [B,N,T,1]
        x = (x * mask).sum(dim=2) / (mask.sum(dim=2)+1e-6)  # [B,N,H]
        x = torch.relu(self.node_fc(x))
        # pool across nodes (mean)
        x = x.mean(dim=1)  # [B,H]
        out = self.out_fc(x)  # [B,out_dim]
        return out
