import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAlign(nn.Module):
    def __init__(self, traj_dim, text_dim, proj_dim=256, temperature=0.07):
        super().__init__()
        self.traj_proj = nn.Linear(traj_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)
        self.temperature = temperature

    def forward(self, traj_feats, text_feats):
        # traj_feats: [B, D1], text_feats: [B, D2]
        t = F.normalize(self.traj_proj(traj_feats), dim=-1)
        s = F.normalize(self.text_proj(text_feats), dim=-1)
        logits = (t @ s.t()) / self.temperature  # [B, B]
        labels = torch.arange(t.size(0), device=t.device)
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        return loss, logits
