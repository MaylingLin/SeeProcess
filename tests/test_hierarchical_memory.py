import torch

from synapse.memory.trajectory_blocks import HierarchicalTrajectoryMemory


def _fake_batch():
    B, N, T, D = 2, 2, 4, 6
    feats = torch.randn(B, N, T, D)
    masks = torch.zeros(B, N, T)
    masks[:, :, :3] = 1.0
    bboxes = torch.randn(B, N, T, 4)
    cls_ids = torch.randint(0, 5, (B, N, T)).float()
    timestamps = torch.arange(T).view(1, 1, T).repeat(B, N, 1)
    text = torch.randn(B, 32)
    ops = [
        [
            {"frame_idx": 1, "type": "search"},
            {"frame_idx": 2, "type": "filter"},
        ],
        [
            {"frame_idx": 0, "type": "open"},
            {"frame_idx": 2, "type": "submit"},
        ],
    ]
    return feats, masks, bboxes, cls_ids, timestamps, text, ops


def test_hierarchical_memory_provides_context_and_retrieval():
    feats, masks, bboxes, cls_ids, timestamps, text, ops = _fake_batch()
    memory = HierarchicalTrajectoryMemory(
        base_feature_dim=feats.size(-1),
        text_feat_dim=text.size(-1),
        block_hidden_dim=48,
        cls_embed_dim=8,
        time_embed_dim=4,
        context_dim=48,
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

    assert outputs["contextual_summary"].shape == (feats.size(0), 48)
    assert outputs["block_embeddings"].dim() == 3
    assert len(outputs["cross_layer_links"]) == feats.size(0)

    query = outputs["contextual_summary"][0].detach()
    retrievals = memory.retrieve(query, level="high", top_k=1)
    assert len(retrievals) == 1
    assert retrievals[0]["level"] == "high"
