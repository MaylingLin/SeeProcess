import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .multimodal_fusion_part2 import CrossAttentionFusion
from typing import Dict, List, Tuple, Optional
import numpy as np

class MultiModalFusionModule(nn.Module):
    """
    多模态融合模块
    将视觉、轨迹、文本等多种模态信息进行深度融合
    """
    def __init__(self,
                 visual_dim=512,
                 text_dim=512, 
                 trajectory_dim=256,
                 spatial_dim=4,
                 output_dim=512,
                 fusion_type="cross_attention",  # "cross_attention", "gated_fusion", "transformer"
                 num_layers=3,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.trajectory_dim = trajectory_dim
        self.spatial_dim = spatial_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.num_layers = num_layers
        
        # 模态特征预处理
        self.visual_preprocessor = ModalityPreprocessor(visual_dim, output_dim)
        self.text_preprocessor = ModalityPreprocessor(text_dim, output_dim)
        self.trajectory_preprocessor = ModalityPreprocessor(trajectory_dim, output_dim)
        
        # 空间位置编码器
        self.spatial_encoder = SpatialPositionEncoder(spatial_dim, output_dim)
        
        # 选择融合策略
        if fusion_type == "cross_attention":
            self.fusion_layers = CrossAttentionFusion(output_dim, num_heads, num_layers, dropout)
        elif fusion_type == "gated_fusion":
            self.fusion_layers = GatedModalityFusion(output_dim, num_layers, dropout)
        elif fusion_type == "transformer":
            self.fusion_layers = TransformerFusion(output_dim, num_heads, num_layers, dropout)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # 输出投影和归一化
        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )
        
        # 注意力可视化权重（用于分析）
        self.attention_weights = None
    
    def forward(self, 
                visual_features: torch.Tensor,
                text_features: torch.Tensor,
                trajectory_features: torch.Tensor,
                spatial_positions: Optional[torch.Tensor] = None,
                modality_masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        多模态特征融合
        
        Args:
            visual_features: [B, visual_dim] 视觉特征
            text_features: [B, text_dim] 文本特征  
            trajectory_features: [B, trajectory_dim] 轨迹特征
            spatial_positions: [B, spatial_dim] 空间位置（可选）
            modality_masks: 各模态的有效性掩码
            
        Returns:
            融合结果字典
        """
        batch_size = visual_features.size(0)
        device = visual_features.device
        
        # 1. 预处理各模态特征
        visual_processed = self.visual_preprocessor(visual_features)    # [B, output_dim]
        text_processed = self.text_preprocessor(text_features)          # [B, output_dim] 
        trajectory_processed = self.trajectory_preprocessor(trajectory_features)  # [B, output_dim]
        
        # 2. 空间位置编码
        if spatial_positions is not None:
            spatial_encoded = self.spatial_encoder(spatial_positions)   # [B, output_dim]
            # 将空间信息融入轨迹特征
            trajectory_processed = trajectory_processed + spatial_encoded
        
        # 3. 构造多模态输入序列
        # 格式: [visual, text, trajectory] 作为序列输入
        modality_sequence = torch.stack([
            visual_processed,
            text_processed, 
            trajectory_processed
        ], dim=1)  # [B, 3, output_dim]
        
        # 4. 创建模态类型嵌入
        modality_type_embeddings = self._get_modality_type_embeddings(batch_size, device)
        modality_sequence = modality_sequence + modality_type_embeddings
        
        # 5. 处理模态掩码
        if modality_masks is None:
            # 默认所有模态都有效
            attention_mask = torch.ones(batch_size, 3, device=device).bool()
        else:
            attention_mask = self._construct_attention_mask(modality_masks, batch_size, device)
        
        # 6. 多模态融合
        fusion_output = self.fusion_layers(
            modality_sequence, 
            attention_mask=attention_mask
        )
        
        # 7. 输出处理
        if isinstance(fusion_output, tuple):
            fused_features, self.attention_weights = fusion_output
        else:
            fused_features = fusion_output
            self.attention_weights = None
        
        # 8. 最终投影
        if fused_features.dim() == 3:
            # 如果输出是序列，进行池化
            fused_features = fused_features.mean(dim=1)  # [B, output_dim]
        
        final_output = self.output_projection(fused_features)
        
        return {
            'fused_features': final_output,
            'visual_processed': visual_processed,
            'text_processed': text_processed,
            'trajectory_processed': trajectory_processed,
            'attention_weights': self.attention_weights,
            'modality_sequence': modality_sequence
        }
    
    def _get_modality_type_embeddings(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """获取模态类型嵌入"""
        # 为每种模态创建固定的类型嵌入
        modality_embeddings = nn.Parameter(
            torch.randn(3, self.output_dim, device=device) * 0.02
        )
        
        # 扩展到批次维度
        type_embeddings = modality_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        if not hasattr(self, '_modality_embeddings'):
            self._modality_embeddings = nn.Parameter(modality_embeddings)
            
        return self._modality_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    
    def _construct_attention_mask(self, modality_masks: Dict[str, torch.Tensor], 
                                batch_size: int, device: torch.device) -> torch.Tensor:
        """构造注意力掩码"""
        attention_mask = torch.ones(batch_size, 3, device=device).bool()
        
        # 根据提供的掩码设置有效性
        modality_order = ['visual', 'text', 'trajectory']
        for i, modality in enumerate(modality_order):
            if modality in modality_masks:
                attention_mask[:, i] = modality_masks[modality].bool()
        
        return attention_mask

class ModalityPreprocessor(nn.Module):
    """模态预处理器"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.preprocessor = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.preprocessor(x)

class SpatialPositionEncoder(nn.Module):
    """空间位置编码器"""
    def __init__(self, spatial_dim: int, output_dim: int):
        super().__init__()
        self.spatial_dim = spatial_dim
        
        # 使用傅里叶特征编码位置信息
        self.fourier_features = FourierFeatureEncoder(spatial_dim, output_dim // 2)
        
        # 位置编码网络
        self.position_encoder = nn.Sequential(
            # FourierFeatureEncoder 返回 sin/cos 拼接后的 output_dim 维度特征
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [B, spatial_dim] 空间位置 [x, y, w, h]
        Returns:
            encoded: [B, output_dim]
        """
        # 归一化位置到[0,1]范围
        normalized_pos = torch.sigmoid(positions)
        
        # 傅里叶特征编码
        fourier_feats = self.fourier_features(normalized_pos)
        
        # 位置编码
        encoded = self.position_encoder(fourier_feats)
        
        return encoded

class FourierFeatureEncoder(nn.Module):
    """傅里叶特征编码器，用于位置编码"""
    def __init__(self, input_dim: int, num_frequencies: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        
        # 随机频率矩阵
        self.register_buffer(
            'frequency_matrix',
            torch.randn(input_dim, num_frequencies) * 10.0
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim]
        Returns:
            features: [B, num_frequencies * 2]  # sin + cos
        """
        # 计算傅里叶特征
        projected = torch.matmul(x, self.frequency_matrix)  # [B, num_frequencies]
        
        # 正弦和余弦特征
        sin_features = torch.sin(projected)
        cos_features = torch.cos(projected)
        
        # 拼接
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)
        
        return fourier_features

# 测试代码
if __name__ == "__main__":
    print("测试多模态融合模块（第一部分）...")
    
    # 创建融合模块
    fusion_module = MultiModalFusionModule(
        visual_dim=512,
        text_dim=512,
        trajectory_dim=256,
        spatial_dim=4,
        output_dim=512,
        fusion_type="cross_attention",
        num_layers=3,
        num_heads=8
    )
    
    # 测试数据
    batch_size = 4
    visual_features = torch.randn(batch_size, 512)
    text_features = torch.randn(batch_size, 512)
    trajectory_features = torch.randn(batch_size, 256)
    spatial_positions = torch.randn(batch_size, 4) * 100  # [x,y,w,h]
    
    # 前向传播
    with torch.no_grad():
        outputs = fusion_module(
            visual_features=visual_features,
            text_features=text_features,
            trajectory_features=trajectory_features,
            spatial_positions=spatial_positions
        )
    
    print("融合模块输出:")
    print(f"  融合特征维度: {outputs['fused_features'].shape}")
    print(f"  视觉特征维度: {outputs['visual_processed'].shape}")
    print(f"  文本特征维度: {outputs['text_processed'].shape}")
    print(f"  轨迹特征维度: {outputs['trajectory_processed'].shape}")
    print(f"  模态序列维度: {outputs['modality_sequence'].shape}")
    
    # 测试空间位置编码器
    spatial_encoder = SpatialPositionEncoder(4, 256)
    test_positions = torch.tensor([[100, 50, 80, 40], [200, 100, 120, 60]], dtype=torch.float)
    
    with torch.no_grad():
        spatial_encoded = spatial_encoder(test_positions)
    print(f"  空间编码维度: {spatial_encoded.shape}")
    
    print("多模态融合模块（第一部分）测试完成！")
