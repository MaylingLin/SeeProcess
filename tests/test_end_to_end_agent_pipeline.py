"""
端到端连通性测试：视觉轨迹 -> 层次记忆 -> Reasoner -> 动作输出。
"""
import torch

from models.hierarchical_encoder import HierarchicalTrajectoryEncoder
from models.cross_modal import CrossModalAlign
from synapse.memory.trajectory_blocks import HierarchicalTrajectoryMemory
from synapse.reasoning import HierarchicalReasoningExecutor


def _fake_agent_batch(batch=2, tracks=2, steps=4, feat_dim=6, text_dim=64):
    torch.manual_seed(3)
    images = torch.randn(batch, 3, 64, 64)
    prev_images = torch.randn(batch, 3, 64, 64)
    track_feats = torch.randn(batch, tracks, steps, feat_dim)
    track_positions = torch.rand(batch, tracks, steps, 4)
    track_masks = torch.zeros(batch, tracks, steps)
    track_masks[:, :, : steps - 1] = 1.0
    cls_ids = torch.randint(0, 5, (batch, tracks, steps)).float()
    timestamps = torch.arange(steps).view(1, 1, steps).repeat(batch, tracks, 1)
    text_feats = torch.randn(batch, text_dim)
    ops = [
        [{"frame_idx": 0, "type": "click"}],
        [{"frame_idx": 1, "type": "type"}],
    ]
    domains = ["gui", "robotics"]
    return {
        "images": images,
        "prev_images": prev_images,
        "track_features": track_feats,
        "track_positions": track_positions,
        "track_masks": track_masks,
        "cls_ids": cls_ids,
        "timestamps": timestamps,
        "text_features": text_feats,
        "ops": ops,
        "domains": domains,
    }


def test_end_to_end_agent_pipeline():
    batch = _fake_agent_batch()
    encoder = HierarchicalTrajectoryEncoder(
        traj_feat_dim=batch["track_features"].size(-1),
        text_dim=batch["text_features"].size(-1),
        fusion_dim=128,
    ).eval()
    memory = HierarchicalTrajectoryMemory(
        base_feature_dim=batch["track_features"].size(-1),
        text_feat_dim=batch["text_features"].size(-1),
        block_hidden_dim=128,
        cls_embed_dim=8,
        time_embed_dim=4,
        context_dim=128,
    )
    reasoner = HierarchicalReasoningExecutor(
        low_dim=128,
        high_dim=128,
        text_dim=batch["text_features"].size(-1),
        hidden_dim=128,
        action_tokens=["click", "type", "submit"],
    )
    align = CrossModalAlign(traj_dim=128, text_dim=batch["text_features"].size(-1), proj_dim=32)

    # 1) 视觉轨迹编码
    enc_out = encoder(
        current_images=batch["images"],
        prev_images=batch["prev_images"],
        track_features=batch["track_features"],
        track_positions=batch["track_positions"],
        track_masks=batch["track_masks"],
        text_features=batch["text_features"],
    )

    # 2) 层次记忆更新
    memory_out = memory(
        step_features=batch["track_features"],
        step_masks=batch["track_masks"],
        spatial_positions=batch["track_positions"],
        cls_ids=batch["cls_ids"],
        timestamps=batch["timestamps"],
        text_features=batch["text_features"],
        ops=batch["ops"],
    )

    fused_with_memory = enc_out["fused_features"] + memory_out["contextual_summary"]
    assert not torch.allclose(enc_out["fused_features"], fused_with_memory)

    # 3) 跨模态对齐，确保 joint embedding 可训练
    align_loss, _ = align(fused_with_memory, batch["text_features"])
    assert align_loss.item() >= 0

    # 4) Reasoner 推理 + 执行
    action_labels = reasoner.encode_actions(batch["ops"], batch["domains"], fused_with_memory.device)
    advantages, returns = reasoner.estimate_advantages(batch["track_masks"], device=fused_with_memory.device)
    reasoner_out = reasoner(
        low_level_state=fused_with_memory,
        high_level_context=memory_out["contextual_summary"],
        text_features=batch["text_features"],
        action_labels=action_labels,
        advantages=advantages,
        returns=returns,
        domains=batch["domains"],
    )

    pred_ids = reasoner_out.action_logits.argmax(dim=-1).tolist()
    pred_tokens = [reasoner.action_vocab.id_to_token[i] for i in pred_ids]
    assert len(pred_tokens) == fused_with_memory.size(0)
    assert all(token in {"click", "type", "submit", "noop"} for token in pred_tokens)