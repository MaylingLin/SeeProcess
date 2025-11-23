"""
验证视觉轨迹表征模块：轨迹动态建模、层级化编码以及视觉-语义对齐。
"""
import torch

from models.hierarchical_encoder import HierarchicalTrajectoryEncoder
from models.cross_modal import CrossModalAlign


def _build_fake_visual_batch(batch=2, tracks=3, steps=4, feat_dim=6, img_hw=64, text_dim=512):
    torch.manual_seed(0)
    images = torch.randn(batch, 3, img_hw, img_hw)
    prev_images = torch.randn(batch, 3, img_hw, img_hw)
    track_feats = torch.randn(batch, tracks, steps, feat_dim)
    track_positions = torch.rand(batch, tracks, steps, 4)
    track_masks = torch.zeros(batch, tracks, steps)
    track_masks[:, :, : steps - 1] = 1.0  # 模拟部分缺失轨迹
    text_features = torch.randn(batch, text_dim)
    return {
        "images": images,
        "prev_images": prev_images,
        "track_features": track_feats,
        "track_positions": track_positions,
        "track_masks": track_masks,
        "text_features": text_features,
    }


def test_visual_trajectory_representation_pipeline():
    batch = _build_fake_visual_batch()
    encoder = HierarchicalTrajectoryEncoder(
        traj_feat_dim=batch["track_features"].size(-1),
        text_dim=batch["text_features"].size(-1),
        fusion_dim=128,
    ).eval()

    # 前向推理：像素层 -> 对象层 -> 融合层
    outputs = encoder(
        current_images=batch["images"],
        prev_images=batch["prev_images"],
        track_features=batch["track_features"],
        track_positions=batch["track_positions"],
        track_masks=batch["track_masks"],
        text_features=batch["text_features"],
    )

    assert outputs["pixel"]["pixel_features"].shape[0] == batch["images"].size(0)
    assert outputs["object"]["object_representations"].shape[:2] == batch["track_features"].shape[:2]
    assert outputs["fused_features"].shape[-1] == encoder.output_dim

    # 视觉-语义对齐：CrossModalAlign 需能返回有效 loss/logits
    align = CrossModalAlign(traj_dim=encoder.output_dim, text_dim=batch["text_features"].size(-1), proj_dim=32)
    loss, logits = align(outputs["fused_features"], batch["text_features"])
    assert loss.item() > 0
    assert logits.shape == (batch["images"].size(0), batch["images"].size(0))

    # 动态建模检查：轨迹发生显著变化时 pooled_nodes 应随之变化
    shifted_tracks = batch["track_features"].clone()
    shifted_tracks[:, :, 1:, :] += 0.5
    outputs_shifted = encoder(
        current_images=batch["images"],
        prev_images=batch["prev_images"],
        track_features=shifted_tracks,
        track_positions=batch["track_positions"],
        track_masks=batch["track_masks"],
        text_features=batch["text_features"],
    )
    assert not torch.allclose(outputs["pooled_nodes"], outputs_shifted["pooled_nodes"])