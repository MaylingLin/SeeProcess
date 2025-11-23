import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    print("skip trajectory builder test: torch not installed")
    sys.exit(0)

from data.trajectory_dataset import TrajectoryDataset
from preprocess.trajectory_builder import build_and_cache_trajectories


def test_build_traj_and_dataset():
    frames = [np.ones((240, 320, 3), dtype=np.uint8) * (50 + i * 30) for i in range(3)]
    ops = [{"frame_idx": 1, "type": "click", "position": (100, 120)}]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "sample_traj.npz"
        trajs = build_and_cache_trajectories(
            frames,
            ops=ops,
            instruction="click button",
            out_npz=str(out_path),
        )
        assert isinstance(trajs, list)
        dataset = TrajectoryDataset(tmpdir)
        sample = dataset[0]
        assert sample["image"].shape[0] == 3
        assert sample["traj_feats"].shape[0] == dataset.max_nodes
        assert sample["text"] == "click button"


if __name__ == "__main__":
    test_build_traj_and_dataset()
