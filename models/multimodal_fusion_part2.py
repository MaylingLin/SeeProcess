import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional

class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合策略
    让不同模态之间相互关注和交互
    """
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 多层交叉注意力
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 前馈网络
        self.feed_forwards = nn.ModuleList([
            FeedForwardNetwork(hidden_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        # 最终聚合层
        self.final_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3个模态
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, modality_sequence: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            modality_sequence: [B, 3, hidden_dim] [visual, text, trajectory]
            attention_mask: [B, 3] 模态有效性掩码
            
        Returns:
            fused_output: [B, hidden_dim]
            attention_weights: 各层的注意力权重列表
        """
        batch_size, num_modalities, hidden_dim = modality_sequence.shape
        x = modality_sequence
        all_attention_weights = []
        
        # 多层交叉注意力处理
        for i in range(self.num_layers):
            # 残差连接的输入
            residual = x
            
            # 交叉注意力
            x, attention_weights = self.cross_attention_layers[i](x, attention_mask)
            all_attention_weights.append(attention_weights)
            
            # 残差连接和层归一化
            x = self.layer_norms[i](x + residual)
            
            # 前馈网络
            ff_residual = x
            x = self.feed_forwards[i](x)
            x = x + ff_residual
        
        # 最终聚合：将三个模态的特征拼接后压缩
        flattened = x.view(batch_size, -1)  # [B, 3*hidden_dim]
        fused_output = self.final_aggregator(flattened)  # [B, hidden_dim]
        
        return fused_output, all_attention_weights

class CrossAttentionLayer(nn.Module):
    """单层交叉注意力"""
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Q, K, V投影层
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, 3, hidden_dim] 模态序列
            attention_mask: [B, 3] 有效性掩码
            
        Returns:
            output: [B, 3, hidden_dim]
            attention_weights: [B, num_heads, 3, 3]
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算 Q, K, V
        Q = self.query_projection(x)  # [B, 3, hidden_dim]
        K = self.key_projection(x)
        V = self.value_projection(x)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 3, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, 3, 3]
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 扩展掩码到多头形式
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, 3, 1]
            expanded_mask = expanded_mask.expand(-1, self.num_heads, -1, seq_len)  # [B, num_heads, 3, 3]
            
            # 对无效位置应用大负值
            attention_scores = attention_scores.masked_fill(~expanded_mask, -1e9)
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        output = torch.matmul(attention_weights, V)  # [B, num_heads, 3, head_dim]
        
        # 重塑回原形状
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # 输出投影
        output = self.output_projection(output)
        
        return output, attention_weights

class GatedModalityFusion(nn.Module):
    """
    门控模态融合策略
    使用门控机制动态权衡不同模态的贡献
    """
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 门控单元
        self.gating_networks = nn.ModuleList([
            ModalityGatingUnit(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # 特征变换网络
        self.feature_transforms = nn.ModuleList([
            nn.ModuleDict({
                'visual': nn.Linear(hidden_dim, hidden_dim),
                'text': nn.Linear(hidden_dim, hidden_dim),
                'trajectory': nn.Linear(hidden_dim, hidden_dim)
            }) for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, modality_sequence: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            modality_sequence: [B, 3, hidden_dim]
            attention_mask: [B, 3]
            
        Returns:
            fused_output: [B, hidden_dim]
        """
        batch_size = modality_sequence.size(0)
        
        # 分解为各个模态
        visual_feat = modality_sequence[:, 0, :]    # [B, hidden_dim]
        text_feat = modality_sequence[:, 1, :]      # [B, hidden_dim]
        trajectory_feat = modality_sequence[:, 2, :] # [B, hidden_dim]
        
        # 逐层门控融合
        for layer_idx in range(self.num_layers):
            # 特征变换
            visual_transformed = self.feature_transforms[layer_idx]['visual'](visual_feat)
            text_transformed = self.feature_transforms[layer_idx]['text'](text_feat)
            trajectory_transformed = self.feature_transforms[layer_idx]['trajectory'](trajectory_feat)
            
            # 门控融合
            gated_output = self.gating_networks[layer_idx](
                visual_transformed, text_transformed, trajectory_transformed, attention_mask
            )
            
            # 残差连接和层归一化
            current_feat = (visual_feat + text_feat + trajectory_feat) / 3  # 简单平均作为残差
            fused_feat = self.layer_norms[layer_idx](gated_output + current_feat)
            
            # 更新各模态特征（共享更新）
            visual_feat = text_feat = trajectory_feat = fused_feat
        
        # 最终融合
        final_output = self.final_fusion(fused_feat)
        
        return final_output

class ModalityGatingUnit(nn.Module):
    """模态门控单元"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 门控权重计算网络
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # 3个模态的门控权重
            nn.Softmax(dim=-1)
        )
        
        # 特征交互网络
        self.interaction_network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, visual_feat: torch.Tensor, 
                text_feat: torch.Tensor,
                trajectory_feat: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            visual_feat: [B, hidden_dim]
            text_feat: [B, hidden_dim] 
            trajectory_feat: [B, hidden_dim]
            attention_mask: [B, 3]
            
        Returns:
            gated_output: [B, hidden_dim]
        """
        # 拼接所有模态特征
        concat_features = torch.cat([visual_feat, text_feat, trajectory_feat], dim=-1)
        
        # 计算门控权重
        gate_weights = self.gate_network(concat_features)  # [B, 3]
        
        # 应用注意力掩码到门控权重
        if attention_mask is not None:
            gate_weights = gate_weights * attention_mask.float()
            # 重新归一化
            gate_weights = gate_weights / (gate_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 加权融合
        weighted_visual = gate_weights[:, 0:1] * visual_feat
        weighted_text = gate_weights[:, 1:2] * text_feat  
        weighted_trajectory = gate_weights[:, 2:3] * trajectory_feat
        
        weighted_features = torch.cat([
            weighted_visual, weighted_text, weighted_trajectory
        ], dim=-1)
        
        # 特征交互
        gated_output = self.interaction_network(weighted_features)
        
        return gated_output

class FeedForwardNetwork(nn.Module):
    """前馈网络"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# 测试代码
if __name__ == "__main__":
    print("测试多模态融合策略...")
    
    batch_size = 4
    hidden_dim = 512
    modality_sequence = torch.randn(batch_size, 3, hidden_dim)
    attention_mask = torch.ones(batch_size, 3).bool()
    
    # 测试交叉注意力融合
    print("测试交叉注意力融合...")
    cross_attention_fusion = CrossAttentionFusion(
        hidden_dim=hidden_dim,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    )
    
    with torch.no_grad():
        ca_output, ca_weights = cross_attention_fusion(modality_sequence, attention_mask)
    print(f"  交叉注意力输出维度: {ca_output.shape}")
    print(f"  注意力权重层数: {len(ca_weights)}")
    
    # 测试门控融合
    print("测试门控融合...")
    gated_fusion = GatedModalityFusion(
        hidden_dim=hidden_dim,
        num_layers=3,
        dropout=0.1
    )
    
    with torch.no_grad():
        gated_output = gated_fusion(modality_sequence, attention_mask)
    print(f"  门控融合输出维度: {gated_output.shape}")
    
    print("融合策略测试完成！")