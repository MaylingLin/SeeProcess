from preprocess.trajectory_builder import build_and_cache_trajectories
import numpy as np

def test_build_traj():
    frames = []
    for i in range(3):
        frames.append(np.ones((240,320,3), dtype=np.uint8)* (50+i*30))
    trajs = build_and_cache_trajectories(frames, out_npz="tests/_sample_traj.npz")
    assert isinstance(trajs, list)
    print("Trajectories built:", len(trajs))

if __name__ == "__main__":
    test_build_traj()
