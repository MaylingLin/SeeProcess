# seeclick_ext/datasets/trajectory_dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    """
    Dataset expects either:
     - a directory of precomputed `.npz` files, each contains 'trajectories' key (list)
     - or a simple list of frame sequences (not implemented here)
    Each item returns dict:
      'image': tensor HWC->CHW float (current frame)
      'text': str (instruction)
      'traj_feats': ndarray [N_nodes, T, D] padded with zeros
      'traj_mask': [N_nodes, T] boolean mask
    """
    def __init__(self, traj_dir, text_map=None, max_nodes=8, max_T=8, feat_dim=3):
        self.traj_files = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir) if f.endswith(".npz")]
        self.max_nodes = max_nodes
        self.max_T = max_T
        self.feat_dim = feat_dim
        self.text_map = text_map or {}  # optional mapping from file->text

    def __len__(self):
        return len(self.traj_files)

    def __getitem__(self, idx):
        p = self.traj_files[idx]
        data = np.load(p, allow_pickle=True)
        trajectories = data["trajectories"].tolist()
        # build padded arrays
        N = min(len(trajectories), self.max_nodes)
        traj_feats = np.zeros((self.max_nodes, self.max_T, self.feat_dim), dtype=np.float32)
        traj_mask = np.zeros((self.max_nodes, self.max_T), dtype=np.float32)
        for i, traj in enumerate(trajectories[:self.max_nodes]):
            frames = traj["frames"]
            for t_i, f in enumerate(frames[:self.max_T]):
                traj_feats[i, t_i, :min(len(f["feat"]), self.feat_dim)] = np.array(f["feat"])[:self.feat_dim]
                traj_mask[i, t_i] = 1.0
        # placeholder: current frame image and instruction
        # for simplicity use zeros for image and empty text
        image = np.zeros((3, 224, 224), dtype=np.float32)
        text = self.text_map.get(os.path.basename(p), "")
        return {"image": torch.tensor(image), "text": text, "traj_feats": torch.tensor(traj_feats), "traj_mask": torch.tensor(traj_mask)}
