import torch
import torch.nn as nn

from .multimodal_fusion_part1 import MultiModalFusionModule
from .object_level_encoder import ObjectLevelEncoder
from .pixel_level_encoder import PixelLevelEncoder


class HierarchicalTrajectoryEncoder(nn.Module):
    """像素-对象-语义三层融合编码器。"""

    def __init__(
        self,
        traj_feat_dim: int,
        text_dim: int = 512,
        fusion_dim: int = 512,
        pixel_base_channels: int = 64,
        object_hidden_dim: int = 256,
        temporal_dim: int = 64,
    ):
        super().__init__()
        self.pixel_encoder = PixelLevelEncoder(
            in_channels=3,
            base_channels=pixel_base_channels,
            num_scales=4,
            enable_attention=True,
        )
        self.visual_dim = self.pixel_encoder.feature_dim
        self.object_encoder = ObjectLevelEncoder(
            appearance_dim=traj_feat_dim,
            spatial_dim=4,
            temporal_dim=temporal_dim,
            hidden_dim=object_hidden_dim,
        )
        self.node_hidden_dim = object_hidden_dim
        self.text_dim = text_dim
        self.fusion_module = MultiModalFusionModule(
            visual_dim=self.visual_dim,
            text_dim=text_dim,
            trajectory_dim=object_hidden_dim,
            spatial_dim=4,
            output_dim=fusion_dim,
        )
        self.output_dim = fusion_dim

    def forward(
        self,
        current_images: torch.Tensor,
        prev_images: torch.Tensor,
        track_features: torch.Tensor,
        track_positions: torch.Tensor,
        track_masks: torch.Tensor,
        text_features: torch.Tensor,
    ):
        pixel_outputs = self.pixel_encoder(current_images, prev_images)
        object_outputs = self.object_encoder(
            object_tracks=None,
            track_features=track_features,
            track_positions=track_positions,
            track_masks=track_masks,
        )
        node_repr = object_outputs["object_representations"]
        node_mask = (track_masks.sum(dim=-1) > 0).float()
        pooled_nodes = self._masked_mean(node_repr, node_mask)

        spatial_positions = self._gather_last_positions(track_positions, track_masks)

        fusion_outputs = self.fusion_module(
            visual_features=pixel_outputs["pixel_features"],
            text_features=text_features,
            trajectory_features=pooled_nodes,
            spatial_positions=spatial_positions,
        )

        return {
            "pixel": pixel_outputs,
            "object": object_outputs,
            "fusion": fusion_outputs,
            "fused_features": fusion_outputs["fused_features"],
            "pooled_nodes": pooled_nodes,
        }

    def _masked_mean(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        summed = (tensor * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        return summed / denom

    def _gather_last_positions(self, positions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        lengths = masks.sum(dim=-1).long().clamp(min=1)
        last_idx = (lengths - 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, positions.size(-1))
        gathered = torch.gather(positions, 2, last_idx).squeeze(2)
        pooled = gathered.mean(dim=1)
        return pooled
