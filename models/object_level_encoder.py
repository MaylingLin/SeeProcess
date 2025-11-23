import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .object_graph_network import ObjectGraphNetwork
from .object_graph_network import CrossTimeAlignment

from typing import List, Dict, Tuple, Optional

class ObjectLevelEncoder(nn.Module):
    """
    对象层表征编码器
    功能：维持交互组件的一致性，支撑跨时间操作的元素对齐
    """
    def __init__(self, 
                 appearance_dim=512,
                 spatial_dim=4,  # [x, y, w, h]
                 temporal_dim=64,
                 hidden_dim=256,
                 max_objects=20,
                 max_timesteps=10):
        super().__init__()
        self.appearance_dim = appearance_dim
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.hidden_dim = hidden_dim
        self.max_objects = max_objects
        self.max_timesteps = max_timesteps
        
        # 外观特征编码器
        self.appearance_encoder = AppearanceEncoder(appearance_dim, hidden_dim)
        
        # 空间关系编码器
        self.spatial_encoder = SpatialRelationEncoder(spatial_dim, hidden_dim)
        
        # 时序一致性编码器
        self.temporal_encoder = TemporalConsistencyEncoder(
            hidden_dim * 2, temporal_dim, max_timesteps
        )
        
        # 对象级图神经网络
        self.object_gnn = ObjectGraphNetwork(hidden_dim * 2 + temporal_dim, hidden_dim)
        
        # 跨时间对齐模块在 appearance+spatial 融合特征上运行，维度等于 hidden_dim * 2
        self.alignment_module = CrossTimeAlignment(hidden_dim * 2)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, 
                object_tracks: List[Dict],
                track_features: torch.Tensor,
                track_positions: torch.Tensor,
                track_masks: torch.Tensor):
        """
        前向传播
        
        Args:
            object_tracks: 对象轨迹列表
            track_features: 轨迹特征 [B, N, T, appearance_dim]
            track_positions: 轨迹位置 [B, N, T, spatial_dim]  
            track_masks: 有效性掩码 [B, N, T]
            
        Returns:
            object_representations: 对象级表征 [B, N, hidden_dim]
            consistency_scores: 一致性分数 [B, N]
            alignment_matrix: 时间对齐矩阵 [B, N, T, T]
        """
        B, N, T = track_features.shape[:3]
        
        # 1. 编码外观特征
        appearance_feats = self.appearance_encoder(track_features)  # [B, N, T, hidden_dim]
        
        # 2. 编码空间关系
        spatial_feats = self.spatial_encoder(track_positions)  # [B, N, T, hidden_dim]
        
        # 3. 融合外观和空间特征
        combined_feats = torch.cat([appearance_feats, spatial_feats], dim=-1)  # [B, N, T, hidden_dim*2]
        
        # 4. 时序一致性编码
        temporal_feats, consistency_scores = self.temporal_encoder(
            combined_feats, track_masks
        )  # [B, N, temporal_dim], [B, N]
        
        # 5. 时间对齐
        alignment_matrix = self.alignment_module(combined_feats, track_masks)  # [B, N, T, T]
        
        # 6. 应用时间对齐权重
        aligned_feats = self._apply_temporal_alignment(
            combined_feats, alignment_matrix, track_masks
        )  # [B, N, hidden_dim*2]
        
        # 7. 拼接时序特征
        object_feats = torch.cat([aligned_feats, temporal_feats], dim=-1)  # [B, N, hidden_dim*2 + temporal_dim]
        
        # 8. 对象图神经网络处理
        graph_feats = self.object_gnn(object_feats, track_positions[:, :, -1])  # [B, N, hidden_dim]
        
        # 9. 输出投影
        object_representations = self.output_projection(graph_feats)  # [B, N, hidden_dim]
        
        return {
            'object_representations': object_representations,
            'consistency_scores': consistency_scores,
            'alignment_matrix': alignment_matrix,
            'temporal_features': temporal_feats,
            'appearance_features': appearance_feats,
            'spatial_features': spatial_feats
        }
    
    def _apply_temporal_alignment(self, 
                                features: torch.Tensor,
                                alignment_matrix: torch.Tensor,
                                masks: torch.Tensor) -> torch.Tensor:
        """应用时间对齐权重"""
        B, N, T, D = features.shape
        
        # 使用对齐矩阵加权平均时序特征
        # alignment_matrix: [B, N, T, T] - 表示每个时刻与其他时刻的对齐权重
        aligned_features = torch.matmul(alignment_matrix, features)  # [B, N, T, D]
        
        # 根据掩码加权平均
        masks_expanded = masks.unsqueeze(-1).expand_as(aligned_features)  # [B, N, T, D]
        masked_features = aligned_features * masks_expanded
        
        # 时间维度加权平均
        valid_counts = masks.sum(dim=-1, keepdim=True) + 1e-6  # [B, N, 1]
        temporal_avg = masked_features.sum(dim=2) / valid_counts  # [B, N, D]
        
        return temporal_avg

class AppearanceEncoder(nn.Module):
    """外观特征编码器"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, features):
        """
        Args:
            features: [B, N, T, input_dim]
        Returns:
            encoded: [B, N, T, output_dim]
        """
        B, N, T, D = features.shape
        flat_features = features.view(-1, D)
        encoded_flat = self.encoder(flat_features)
        encoded = encoded_flat.view(B, N, T, -1)
        return encoded

class SpatialRelationEncoder(nn.Module):
    """空间关系编码器"""
    def __init__(self, spatial_dim, output_dim):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.position_encoder = nn.Sequential(
            nn.Linear(spatial_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        # 相对位置编码
        self.relative_encoder = nn.Sequential(
            nn.Linear(spatial_dim * 2, output_dim // 2),
            nn.ReLU(), 
            nn.Linear(output_dim // 2, output_dim)
        )
    
    def forward(self, positions):
        """
        Args:
            positions: [B, N, T, spatial_dim] - [x, y, w, h]
        Returns:
            spatial_features: [B, N, T, output_dim]
        """
        B, N, T, D = positions.shape
        
        # 绝对位置编码
        flat_positions = positions.view(-1, D)
        absolute_feats = self.position_encoder(flat_positions)
        absolute_feats = absolute_feats.view(B, N, T, -1)
        
        # 相对位置编码（与第一帧的相对位置）
        first_positions = positions[:, :, :1, :].expand(-1, -1, T, -1)  # [B, N, T, D]
        relative_positions = torch.cat([positions, first_positions], dim=-1)  # [B, N, T, 2*D]
        flat_relative = relative_positions.view(-1, D * 2)
        relative_feats = self.relative_encoder(flat_relative)
        relative_feats = relative_feats.view(B, N, T, -1)
        
        # 融合绝对和相对位置特征
        spatial_features = absolute_feats + relative_feats
        
        return spatial_features

class TemporalConsistencyEncoder(nn.Module):
    """时序一致性编码器"""
    def __init__(self, input_dim, output_dim, max_timesteps):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_timesteps = max_timesteps
        
        # LSTM用于时序建模
        self.temporal_lstm = nn.LSTM(
            input_dim, output_dim, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.1
        )
        
        # 一致性评分网络
        self.consistency_scorer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )
        
        # 特征聚合
        self.feature_aggregator = nn.Linear(output_dim * 2, output_dim)
    
    def forward(self, features, masks):
        """
        Args:
            features: [B, N, T, input_dim]
            masks: [B, N, T]
        Returns:
            temporal_features: [B, N, output_dim]
            consistency_scores: [B, N]
        """
        B, N, T, D = features.shape
        
        # 重塑为 (B*N, T, D) 用于LSTM
        flat_features = features.view(B * N, T, D)
        flat_masks = masks.view(B * N, T)
        
        # 创建packed序列以处理变长序列
        lengths = flat_masks.sum(dim=1).long()
        safe_lengths = lengths.clone()
        safe_lengths[safe_lengths <= 0] = 1
        packed_input = nn.utils.rnn.pack_padded_sequence(
            flat_features, safe_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM编码
        packed_output, (hidden, _) = self.temporal_lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=T
        )
        
        # 重塑回 (B, N, T, output_dim*2)
        lstm_output = lstm_output.view(B, N, T, -1)
        
        # 计算一致性分数（基于最后一个有效时刻的特征）
        last_indices = (safe_lengths - 1).view(B, N, 1, 1).expand(-1, -1, 1, lstm_output.size(-1))
        last_features = torch.gather(lstm_output, 2, last_indices).squeeze(2)  # [B, N, output_dim*2]
        if (lengths <= 0).any():
            zero_mask = (lengths <= 0).view(B, N, 1)
            last_features = last_features.masked_fill(zero_mask, 0.0)
            lstm_output = lstm_output.masked_fill(zero_mask.unsqueeze(2), 0.0)
        
        consistency_scores = self.consistency_scorer(last_features).squeeze(-1)  # [B, N]
        
        # 特征聚合（掩码加权平均）
        masked_output = lstm_output * masks.unsqueeze(-1)
        valid_counts = masks.sum(dim=-1, keepdim=True) + 1e-6
        temporal_features = masked_output.sum(dim=2) / valid_counts  # [B, N, output_dim*2]
        temporal_features = self.feature_aggregator(temporal_features)  # [B, N, output_dim]
        
        return temporal_features, consistency_scores

# 测试代码
if __name__ == "__main__":
    # 创建对象层编码器
    object_encoder = ObjectLevelEncoder(
        appearance_dim=512,
        spatial_dim=4,
        temporal_dim=64,
        hidden_dim=256,
        max_objects=10,
        max_timesteps=8
    )
    
    # 测试数据
    batch_size = 2
    num_objects = 5
    num_timesteps = 6
    
    track_features = torch.randn(batch_size, num_objects, num_timesteps, 512)
    track_positions = torch.randn(batch_size, num_objects, num_timesteps, 4)
    track_masks = torch.randint(0, 2, (batch_size, num_objects, num_timesteps)).float()
    
    # 确保每个轨迹至少有一个有效时刻
    track_masks[:, :, 0] = 1.0
    
    # 前向传播
    with torch.no_grad():
        outputs = object_encoder(
            object_tracks=None,  # 占位符
            track_features=track_features,
            track_positions=track_positions,
            track_masks=track_masks
        )
    
    print("对象层编码器输出:")
    print(f"对象表征维度: {outputs['object_representations'].shape}")
    print(f"一致性分数维度: {outputs['consistency_scores'].shape}")
    print(f"对齐矩阵维度: {outputs['alignment_matrix'].shape}")
    print(f"时序特征维度: {outputs['temporal_features'].shape}")
