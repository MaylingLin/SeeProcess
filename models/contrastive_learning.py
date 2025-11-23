import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class ContrastiveLearningModule(nn.Module):
    """
    跨模态对比学习模块
    实现轨迹-文本、图像-文本等多种模态对比学习
    """
    def __init__(self,
                 visual_dim=512,
                 text_dim=512,
                 trajectory_dim=256,
                 projection_dim=256,
                 temperature=0.07,
                 use_hard_negatives=True,
                 negative_weight=1.0):
        super().__init__()
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.use_hard_negatives = use_hard_negatives
        self.negative_weight = negative_weight
        
        # 各模态投影层
        self.visual_projector = ModalityProjector(visual_dim, projection_dim)
        self.text_projector = ModalityProjector(text_dim, projection_dim)
        self.trajectory_projector = ModalityProjector(trajectory_dim, projection_dim)
        
        # 自适应温度参数（可学习）
        self.learnable_temperature = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        
        # 难负样本挖掘器
        if use_hard_negatives:
            self.hard_negative_miner = HardNegativeMiner(projection_dim)
        
        # 多层次对比损失权重
        self.contrastive_weights = nn.Parameter(torch.ones(4))  # [vis-text, traj-text, vis-traj, temporal]
    
    def forward(self, 
                visual_features: torch.Tensor,
                text_features: torch.Tensor, 
                trajectory_features: torch.Tensor,
                temporal_features: Optional[torch.Tensor] = None,
                instruction_masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播计算多种对比损失
        
        Args:
            visual_features: [B, visual_dim] 视觉特征
            text_features: [B, text_dim] 文本特征
            trajectory_features: [B, trajectory_dim] 轨迹特征
            temporal_features: [B, T, feature_dim] 时序特征（可选）
            instruction_masks: [B,] 指令有效性掩码
            
        Returns:
            loss_dict: 各种损失的字典
        """
        batch_size = visual_features.size(0)
        
        # 投影到统一空间
        visual_proj = self.visual_projector(visual_features)
        text_proj = self.text_projector(text_features)
        trajectory_proj = self.trajectory_projector(trajectory_features)
        
        # 归一化
        visual_proj = F.normalize(visual_proj, dim=-1)
        text_proj = F.normalize(text_proj, dim=-1)
        trajectory_proj = F.normalize(trajectory_proj, dim=-1)
        
        # 获取自适应温度
        temp = torch.exp(self.learnable_temperature).clamp(min=0.01, max=1.0)
        
        loss_dict = {}
        
        # 1. 视觉-文本对比损失
        vis_text_loss = self._compute_contrastive_loss(
            visual_proj, text_proj, temp, "visual_text"
        )
        loss_dict['visual_text_loss'] = vis_text_loss
        
        # 2. 轨迹-文本对比损失
        traj_text_loss = self._compute_contrastive_loss(
            trajectory_proj, text_proj, temp, "trajectory_text"
        )
        loss_dict['trajectory_text_loss'] = traj_text_loss
        
        # 3. 视觉-轨迹对比损失
        vis_traj_loss = self._compute_contrastive_loss(
            visual_proj, trajectory_proj, temp, "visual_trajectory"
        )
        loss_dict['visual_trajectory_loss'] = vis_traj_loss
        
        # 4. 时序对比损失（如果提供时序特征）
        if temporal_features is not None:
            temporal_loss = self._compute_temporal_contrastive_loss(
                temporal_features, text_proj, temp
            )
            loss_dict['temporal_loss'] = temporal_loss
        else:
            loss_dict['temporal_loss'] = torch.tensor(0.0, device=visual_features.device)
        
        # 5. 加权总损失
        weights = F.softmax(self.contrastive_weights, dim=0)
        total_loss = (
            weights[0] * loss_dict['visual_text_loss'] +
            weights[1] * loss_dict['trajectory_text_loss'] + 
            weights[2] * loss_dict['visual_trajectory_loss'] +
            weights[3] * loss_dict['temporal_loss']
        )
        loss_dict['total_contrastive_loss'] = total_loss
        
        # 6. 计算相似度矩阵（用于分析）
        loss_dict.update(self._compute_similarity_matrices(
            visual_proj, text_proj, trajectory_proj
        ))
        
        return loss_dict
    
    def _compute_contrastive_loss(self, 
                                features1: torch.Tensor,
                                features2: torch.Tensor, 
                                temperature: torch.Tensor,
                                modality_pair: str) -> torch.Tensor:
        """计算两个模态之间的对比损失"""
        batch_size = features1.size(0)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features1, features2.t()) / temperature
        
        # 创建正样本标签（对角线为正样本）
        labels = torch.arange(batch_size, device=features1.device)
        
        # 难负样本挖掘
        if self.use_hard_negatives:
            similarity_matrix = self.hard_negative_miner(
                similarity_matrix, labels, modality_pair
            )
        
        # 计算交叉熵损失（双向）
        loss_12 = F.cross_entropy(similarity_matrix, labels)
        loss_21 = F.cross_entropy(similarity_matrix.t(), labels)
        
        contrastive_loss = (loss_12 + loss_21) / 2
        
        return contrastive_loss
    
    def _compute_temporal_contrastive_loss(self,
                                         temporal_features: torch.Tensor,
                                         text_features: torch.Tensor,
                                         temperature: torch.Tensor) -> torch.Tensor:
        """计算时序对比损失"""
        B, T, D = temporal_features.shape
        
        # 时序特征平均池化
        temporal_pooled = temporal_features.mean(dim=1)  # [B, D]
        temporal_pooled = F.normalize(temporal_pooled, dim=-1)
        
        # 计算时序-文本对比损失
        temporal_loss = self._compute_contrastive_loss(
            temporal_pooled, text_features, temperature, "temporal_text"
        )
        
        # 额外的时序一致性损失
        if T > 1:
            # 相邻帧之间的一致性
            temporal_shifted = temporal_features[:, 1:, :]  # [B, T-1, D]
            temporal_original = temporal_features[:, :-1, :]  # [B, T-1, D]
            
            # 计算相邻帧的余弦相似度
            temporal_shifted_norm = F.normalize(temporal_shifted, dim=-1)
            temporal_original_norm = F.normalize(temporal_original, dim=-1)
            
            consistency_sim = (temporal_shifted_norm * temporal_original_norm).sum(dim=-1)  # [B, T-1]
            consistency_loss = (1 - consistency_sim).mean()  # 鼓励相邻帧相似
            
            temporal_loss = temporal_loss + 0.1 * consistency_loss
        
        return temporal_loss
    
    def _compute_similarity_matrices(self, 
                                   visual_proj: torch.Tensor,
                                   text_proj: torch.Tensor,
                                   trajectory_proj: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算各模态间的相似度矩阵"""
        similarities = {}
        
        # 视觉-文本相似度
        similarities['visual_text_sim'] = torch.matmul(visual_proj, text_proj.t())
        
        # 轨迹-文本相似度  
        similarities['trajectory_text_sim'] = torch.matmul(trajectory_proj, text_proj.t())
        
        # 视觉-轨迹相似度
        similarities['visual_trajectory_sim'] = torch.matmul(visual_proj, trajectory_proj.t())
        
        return similarities
    
    def compute_retrieval_metrics(self, 
                                visual_features: torch.Tensor,
                                text_features: torch.Tensor,
                                k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """计算检索评估指标"""
        # 投影和归一化
        visual_proj = F.normalize(self.visual_projector(visual_features), dim=-1)
        text_proj = F.normalize(self.text_projector(text_features), dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(visual_proj, text_proj.t())
        
        batch_size = similarity.size(0)
        metrics = {}
        
        # 图像到文本检索
        for k in k_values:
            # 获取top-k索引
            _, top_k_indices = torch.topk(similarity, k, dim=1)
            
            # 计算Recall@K
            correct = 0
            for i in range(batch_size):
                if i in top_k_indices[i]:
                    correct += 1
            
            recall_at_k = correct / batch_size
            metrics[f'img2text_recall@{k}'] = recall_at_k
        
        # 文本到图像检索
        similarity_t = similarity.t()
        for k in k_values:
            _, top_k_indices = torch.topk(similarity_t, k, dim=1)
            
            correct = 0
            for i in range(batch_size):
                if i in top_k_indices[i]:
                    correct += 1
            
            recall_at_k = correct / batch_size
            metrics[f'text2img_recall@{k}'] = recall_at_k
        
        return metrics

class ModalityProjector(nn.Module):
    """模态投影器"""
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(input_dim, output_dim)
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projector(x)

class HardNegativeMiner(nn.Module):
    """难负样本挖掘器"""
    def __init__(self, feature_dim, hard_ratio=0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.hard_ratio = hard_ratio
        
        # 难度评估网络
        self.difficulty_scorer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, similarity_matrix, labels, modality_pair):
        """
        挖掘难负样本
        
        Args:
            similarity_matrix: [B, B] 相似度矩阵
            labels: [B] 正样本标签
            modality_pair: 模态对名称
            
        Returns:
            enhanced_similarity: 增强的相似度矩阵
        """
        batch_size = similarity_matrix.size(0)
        device = similarity_matrix.device
        
        # 创建负样本掩码（非对角线元素）
        eye_mask = torch.eye(batch_size, device=device).bool()
        negative_mask = ~eye_mask
        
        # 提取负样本相似度
        negative_similarities = similarity_matrix[negative_mask]
        
        # 选择难负样本（相似度高的负样本）
        num_hard_negatives = int(len(negative_similarities) * self.hard_ratio)
        if num_hard_negatives > 0:
            hard_neg_values, hard_neg_indices = torch.topk(
                negative_similarities, num_hard_negatives, largest=True
            )
            
            # 对难负样本施加额外权重
            enhanced_similarity = similarity_matrix.clone()
            flat_negative_mask = negative_mask.flatten()
            flat_indices = torch.where(flat_negative_mask)[0][hard_neg_indices]
            
            # 将一维索引转换为二维坐标
            row_indices = flat_indices // batch_size
            col_indices = flat_indices % batch_size
            
            # 增强难负样本的相似度
            enhanced_similarity[row_indices, col_indices] *= 1.2
        else:
            enhanced_similarity = similarity_matrix
        
        return enhanced_similarity

class MultiScaleContrastiveLoss(nn.Module):
    """多尺度对比损失"""
    def __init__(self, scales=[1.0, 0.5, 0.25], base_temperature=0.07):
        super().__init__()
        self.scales = scales
        self.base_temperature = base_temperature
        
        # 每个尺度的温度参数
        self.scale_temperatures = nn.Parameter(
            torch.tensor([base_temperature / scale for scale in scales])
        )
    
    def forward(self, features1, features2):
        """
        计算多尺度对比损失
        
        Args:
            features1: [B, D] 第一模态特征
            features2: [B, D] 第二模态特征
            
        Returns:
            multi_scale_loss: 多尺度损失
        """
        total_loss = 0.0
        
        for i, scale in enumerate(self.scales):
            # 对特征进行缩放
            if scale != 1.0:
                scaled_feat1 = features1 * scale
                scaled_feat2 = features2 * scale
            else:
                scaled_feat1 = features1
                scaled_feat2 = features2
            
            # 归一化
            scaled_feat1 = F.normalize(scaled_feat1, dim=-1)
            scaled_feat2 = F.normalize(scaled_feat2, dim=-1)
            
            # 计算该尺度的对比损失
            temperature = self.scale_temperatures[i]
            similarity = torch.matmul(scaled_feat1, scaled_feat2.t()) / temperature
            
            labels = torch.arange(features1.size(0), device=features1.device)
            loss = F.cross_entropy(similarity, labels)
            
            # 加权累加
            weight = 1.0 / len(self.scales)
            total_loss += weight * loss
        
        return total_loss

# 测试代码
if __name__ == "__main__":
    print("测试跨模态对比学习模块...")
    
    # 创建对比学习模块
    contrastive_module = ContrastiveLearningModule(
        visual_dim=512,
        text_dim=512,
        trajectory_dim=256,
        projection_dim=256,
        temperature=0.07,
        use_hard_negatives=True
    )
    
    # 创建测试数据
    batch_size = 8
    visual_features = torch.randn(batch_size, 512)
    text_features = torch.randn(batch_size, 512)
    trajectory_features = torch.randn(batch_size, 256)
    temporal_features = torch.randn(batch_size, 10, 256)  # 10个时间步
    
    # 前向传播
    with torch.no_grad():
        loss_dict = contrastive_module(
            visual_features=visual_features,
            text_features=text_features,
            trajectory_features=trajectory_features,
            temporal_features=temporal_features
        )
    
    print("对比学习损失:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"  {key}: {value.item():.4f}")
    
    # 测试检索指标
    retrieval_metrics = contrastive_module.compute_retrieval_metrics(
        visual_features, text_features, k_values=[1, 3, 5]
    )
    
    print("检索指标:")
    for key, value in retrieval_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("对比学习模块测试完成！")