import torch
import torch.nn as nn

class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden)
    def forward(self, x, adj=None):
        # x: [B,N,D]
        # adj: [B,N,N] or None
        if adj is None:
            return torch.relu(self.fc(x))
        agg = torch.bmm(adj, x)  # [B,N,D]
        return torch.relu(self.fc(agg))
