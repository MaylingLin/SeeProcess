import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class ObjectGraphNetwork(nn.Module):
    """
    对象级图神经网络
    建模对象之间的空间关系和交互
    """
    def __init__(self, 
                 node_dim,
                 hidden_dim,
                 num_layers=3,
                 num_heads=8,
                 dropout=0.1):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 节点特征投影
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        
        # 图注意力层
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # 空间关系编码
        self.spatial_relation_encoder = SpatialRelationGNN(hidden_dim)
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # 残差连接中的投影层
        self.residual_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) if i > 0 else nn.Identity()
            for i in range(num_layers)
        ])
    
    def forward(self, node_features, positions):
        """
        Args:
            node_features: [B, N, node_dim] 节点特征
            positions: [B, N, 4] 位置信息 [x, y, w, h]
            
        Returns:
            graph_features: [B, N, hidden_dim] 图增强特征
        """
        B, N, _ = node_features.shape
        
        # 投影到隐藏维度
        x = self.node_projection(node_features)  # [B, N, hidden_dim]
        
        # 计算空间邻接矩阵
        spatial_adj = self._compute_spatial_adjacency(positions)  # [B, N, N]
        
        # 逐层图卷积
        for i, (gat_layer, layer_norm, residual_proj) in enumerate(
            zip(self.gat_layers, self.layer_norms, self.residual_projections)
        ):
            # 残差连接的输入
            residual = residual_proj(x)
            
            # 图注意力
            x = gat_layer(x, spatial_adj)
            
            # 残差连接和层归一化
            x = layer_norm(x + residual)
        
        # 应用空间关系增强
        enhanced_features = self.spatial_relation_encoder(x, positions)
        
        return enhanced_features
    
    def _compute_spatial_adjacency(self, positions):
        """计算基于空间距离的邻接矩阵"""
        B, N, _ = positions.shape
        
        # 计算中心点坐标
        centers = positions[:, :, :2] + positions[:, :, 2:] / 2  # [B, N, 2]
        
        # 计算距离矩阵
        centers_expanded1 = centers.unsqueeze(2).expand(B, N, N, 2)  # [B, N, N, 2]
        centers_expanded2 = centers.unsqueeze(1).expand(B, N, N, 2)  # [B, N, N, 2]
        distances = torch.norm(centers_expanded1 - centers_expanded2, dim=-1)  # [B, N, N]
        
        # 使用高斯核计算邻接权重
        sigma = 100.0  # 空间距离的标准差
        adjacency = torch.exp(-distances.pow(2) / (2 * sigma * sigma))
        
        # 移除自连接
        eye = torch.eye(N, device=positions.device, dtype=positions.dtype)
        adjacency = adjacency * (1 - eye.unsqueeze(0))
        
        # 归一化
        row_sums = adjacency.sum(dim=-1, keepdim=True) + 1e-6
        adjacency = adjacency / row_sums
        
        return adjacency

class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        # 多头注意力的线性变换
        self.q_linear = nn.Linear(in_dim, out_dim)
        self.k_linear = nn.Linear(in_dim, out_dim)
        self.v_linear = nn.Linear(in_dim, out_dim)
        
        # 输出投影
        self.out_linear = nn.Linear(out_dim, out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, adjacency=None):
        """
        Args:
            x: [B, N, in_dim]
            adjacency: [B, N, N] 可选的邻接矩阵
            
        Returns:
            output: [B, N, out_dim]
        """
        B, N, _ = x.shape
        
        # 计算 Q, K, V
        Q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim)  # [B, N, num_heads, head_dim]
        K = self.k_linear(x).view(B, N, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(B, N, self.num_heads, self.head_dim)
        
        # 转置以便计算注意力
        Q = Q.transpose(1, 2)  # [B, num_heads, N, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, N, N]
        
        # 应用空间邻接约束（如果提供）
        if adjacency is not None:
            # 将邻接矩阵扩展到多头
            adjacency_expanded = adjacency.unsqueeze(1).expand(B, self.num_heads, N, N)
            # 对非邻接节点应用大的负值（经过softmax后变为0）
            attention_scores = attention_scores.masked_fill(adjacency_expanded < 1e-6, -1e9)
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        output = torch.matmul(attention_weights, V)  # [B, num_heads, N, head_dim]
        
        # 重新整形和投影
        output = output.transpose(1, 2).contiguous().view(B, N, self.out_dim)  # [B, N, out_dim]
        output = self.out_linear(output)
        
        return output

class SpatialRelationGNN(nn.Module):
    """空间关系图神经网络"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 相对位置编码器
        self.relative_position_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim // 2),  # [dx, dy, dw, dh, dist, angle, area_ratio, overlap]
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 空间关系融合
        self.spatial_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, node_features, positions):
        """
        Args:
            node_features: [B, N, hidden_dim]
            positions: [B, N, 4] [x, y, w, h]
            
        Returns:
            enhanced_features: [B, N, hidden_dim]
        """
        B, N, _ = positions.shape
        
        # 计算相对空间特征
        relative_features = self._compute_relative_spatial_features(positions)  # [B, N, N, 8]
        
        # 编码相对位置
        flat_relative = relative_features.view(B * N * N, 8)
        encoded_relative = self.relative_position_encoder(flat_relative)  # [B*N*N, hidden_dim]
        encoded_relative = encoded_relative.view(B, N, N, self.hidden_dim)
        
        # 聚合邻居的空间关系信息
        spatial_context = encoded_relative.mean(dim=2)  # [B, N, hidden_dim] - 平均所有邻居的关系
        
        # 融合原始特征和空间上下文
        combined_features = torch.cat([node_features, spatial_context], dim=-1)
        enhanced_features = self.spatial_fusion(combined_features)
        
        return enhanced_features
    
    def _compute_relative_spatial_features(self, positions):
        """计算相对空间特征"""
        B, N, _ = positions.shape
        
        # 扩展位置矩阵用于计算相对位置
        pos1 = positions.unsqueeze(2).expand(B, N, N, 4)  # [B, N, N, 4]
        pos2 = positions.unsqueeze(1).expand(B, N, N, 4)  # [B, N, N, 4]
        
        # 计算中心点
        center1 = pos1[:, :, :, :2] + pos1[:, :, :, 2:] / 2
        center2 = pos2[:, :, :, :2] + pos2[:, :, :, 2:] / 2
        
        # 相对位置差
        dx = center2[:, :, :, 0] - center1[:, :, :, 0]  # [B, N, N]
        dy = center2[:, :, :, 1] - center1[:, :, :, 1]
        dw = pos2[:, :, :, 2] - pos1[:, :, :, 2]
        dh = pos2[:, :, :, 3] - pos1[:, :, :, 3]
        
        # 距离
        dist = torch.sqrt(dx*dx + dy*dy)
        
        # 角度
        angle = torch.atan2(dy, dx + 1e-6)
        
        # 面积比例
        area1 = pos1[:, :, :, 2] * pos1[:, :, :, 3]
        area2 = pos2[:, :, :, 2] * pos2[:, :, :, 3]
        area_ratio = area2 / (area1 + 1e-6)
        
        # 重叠面积比例
        x1_max = torch.max(pos1[:, :, :, 0], pos2[:, :, :, 0])
        y1_max = torch.max(pos1[:, :, :, 1], pos2[:, :, :, 1])
        x2_min = torch.min(pos1[:, :, :, 0] + pos1[:, :, :, 2], pos2[:, :, :, 0] + pos2[:, :, :, 2])
        y2_min = torch.min(pos1[:, :, :, 1] + pos1[:, :, :, 3], pos2[:, :, :, 1] + pos2[:, :, :, 3])
        
        overlap_w = torch.clamp(x2_min - x1_max, min=0)
        overlap_h = torch.clamp(y2_min - y1_max, min=0)
        overlap_area = overlap_w * overlap_h
        union_area = area1 + area2 - overlap_area
        overlap_ratio = overlap_area / (union_area + 1e-6)
        
        # 堆叠所有特征
        relative_features = torch.stack([
            dx, dy, dw, dh, dist, angle, area_ratio, overlap_ratio
        ], dim=-1)  # [B, N, N, 8]
        
        return relative_features

class CrossTimeAlignment(nn.Module):
    """跨时间对齐模块"""
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # 时间注意力机制
        self.temporal_attention = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(feature_dim)
        
        # 对齐权重计算
        self.alignment_scorer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )
    
    def forward(self, temporal_features, masks):
        """
        计算跨时间对齐矩阵
        
        Args:
            temporal_features: [B, N, T, feature_dim]
            masks: [B, N, T]
            
        Returns:
            alignment_matrix: [B, N, T, T]
        """
        B, N, T, D = temporal_features.shape
        
        # 重塑为 (B*N, T, D)
        flat_features = temporal_features.view(B * N, T, D)
        flat_masks = masks.view(B * N, T)
        
        # 添加位置编码
        pos_encoded = self.positional_encoding(flat_features)
        
        # 计算时间注意力权重
        attn_output, attn_weights = self.temporal_attention(
            pos_encoded, pos_encoded, pos_encoded,
            key_padding_mask=~flat_masks.bool()
        )
        # attn_weights: [B*N, T, T]
        
        # 重塑回原始形状
        alignment_matrix = attn_weights.view(B, N, T, T)
        
        return alignment_matrix

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model] or [batch_size, seq_len, d_model]
        """
        if x.dim() == 3 and x.size(0) != x.size(1):  # batch_first=True
            seq_len = x.size(1)
            # self.pe[:seq_len].transpose(0, 1) -> [1, seq_len, d_model]
            return x + self.pe[:seq_len].transpose(0, 1)
        else:  # batch_first=False
            return x + self.pe[:x.size(0)]

# 测试代码
if __name__ == "__main__":
    # 测试对象图网络
    print("测试对象图网络...")
    object_gnn = ObjectGraphNetwork(
        node_dim=128,
        hidden_dim=256,
        num_layers=3,
        num_heads=8
    )
    
    batch_size = 2
    num_objects = 8
    node_features = torch.randn(batch_size, num_objects, 128)
    positions = torch.randn(batch_size, num_objects, 4) * 100  # 模拟位置
    
    with torch.no_grad():
        graph_output = object_gnn(node_features, positions)
    print(f"图网络输出维度: {graph_output.shape}")
    
    # 测试跨时间对齐
    print("\n测试跨时间对齐...")
    alignment_module = CrossTimeAlignment(feature_dim=256)
    
    num_timesteps = 6
    temporal_features = torch.randn(batch_size, num_objects, num_timesteps, 256)
    masks = torch.randint(0, 2, (batch_size, num_objects, num_timesteps)).float()
    
    with torch.no_grad():
        alignment_matrix = alignment_module(temporal_features, masks)
    print(f"对齐矩阵维度: {alignment_matrix.shape}")
    print("测试完成！")
