"""
验证层次化长期记忆：轨迹块划分、记忆层更新以及跨层检索。
"""
import torch

from synapse.memory.trajectory_blocks import HierarchicalTrajectoryMemory


def _fake_memory_batch(batch=2, nodes=2, steps=5, feat_dim=6):
    torch.manual_seed(1)
    feats = torch.randn(batch, nodes, steps, feat_dim)
    masks = torch.zeros(batch, nodes, steps)
    masks[:, :, : steps - 1] = 1.0
    bboxes = torch.rand(batch, nodes, steps, 4)
    cls_ids = torch.randint(0, 5, (batch, nodes, steps)).float()
    timestamps = torch.arange(steps).view(1, 1, steps).repeat(batch, nodes, 1)
    text = torch.randn(batch, 64)
    ops = [
        [{"frame_idx": 1, "type": "click"}, {"frame_idx": 3, "type": "scroll"}],
        [{"frame_idx": 0, "type": "type"}],
    ]
    return feats, masks, bboxes, cls_ids, timestamps, text, ops


def test_hierarchical_memory_levels_and_links():
    feats, masks, bboxes, cls_ids, timestamps, text, ops = _fake_memory_batch()
    memory = HierarchicalTrajectoryMemory(
        base_feature_dim=feats.size(-1),
        text_feat_dim=text.size(-1),
        block_hidden_dim=64,
        cls_embed_dim=8,
        time_embed_dim=4,
        context_dim=64,
        max_low_steps=64,
        max_mid_blocks=32,
    )

    outputs = memory(
        step_features=feats,
        step_masks=masks,
        spatial_positions=bboxes,
        cls_ids=cls_ids,
        timestamps=timestamps,
        text_features=text,
        ops=ops,
    )

    assert outputs["contextual_summary"].shape == (feats.size(0), 64)
    assert outputs["block_embeddings"].dim() == 3
    assert len(outputs["cross_layer_links"]) == feats.size(0)

    # 低层记忆保存了所有有效步骤
    expected_steps = int(masks.sum().item())
    assert len(memory.low_level.entries) == expected_steps

    # 中层/高层存储了至少一条轨迹块记录
    assert len(memory.mid_level.entries) > 0
    assert len(memory.high_level.entries) == feats.size(0)

    # 检查跨层检索
    high_query = outputs["contextual_summary"][0].detach()
    high_hits = memory.retrieve(high_query, level="high", top_k=1)
    assert high_hits and high_hits[0]["level"] == "high"

    block_emb = outputs["block_embeddings"][0, 0].detach()
    mid_hits = memory.retrieve(block_emb, level="mid", top_k=1)
    assert mid_hits and mid_hits[0]["level"] == "mid"