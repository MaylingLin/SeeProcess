import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Dict, Tuple

class PixelLevelEncoder(nn.Module):
    """
    像素层表征编码器
    功能：捕获界面元素的微小变化和局部细节
    """
    def __init__(self, 
                 in_channels=3,
                 base_channels=64,
                 num_scales=4,
                 enable_attention=True):
        super().__init__()
        self.num_scales = num_scales
        self.enable_attention = enable_attention
        
        # 多尺度特征提取器
        self.multiscale_encoders = nn.ModuleList()
        for i in range(num_scales):
            encoder = self._build_scale_encoder(in_channels, base_channels * (2**i))
            self.multiscale_encoders.append(encoder)
        
        # 像素级差分检测网络
        self.diff_detector = PixelDifferenceDetector(in_channels, base_channels)
        
        # 空间注意力机制在融合特征上运行，此时通道数为 base_channels * 2
        if enable_attention:
            self.spatial_attention = SpatialAttentionModule(base_channels * 2)
        
        # 特征融合层
        total_channels = sum([base_channels * (2**i) for i in range(num_scales)])
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, base_channels * 4, 3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # 全局特征池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = base_channels * 2
    
    def _build_scale_encoder(self, in_channels, out_channels):
        """构建单尺度特征编码器"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, 3, padding=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, images, prev_images=None):
        """
        前向传播
        
        Args:
            images: 当前帧图像 [B, C, H, W]
            prev_images: 前一帧图像 [B, C, H, W] (可选，用于差分检测)
            
        Returns:
            pixel_features: 像素级特征 [B, feature_dim]
            attention_maps: 注意力图 [B, 1, H, W] (如果启用)
            diff_maps: 差分图 [B, 1, H, W] (如果提供前一帧)
        """
        B, C, H, W = images.shape
        
        # 多尺度特征提取
        multiscale_features = []
        current_input = images
        
        for i, encoder in enumerate(self.multiscale_encoders):
            # 对于不同尺度，先下采样再编码
            if i > 0:
                scale_factor = 2 ** i
                scaled_input = F.interpolate(
                    current_input, 
                    scale_factor=1.0/scale_factor, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                scaled_input = current_input
            
            # 特征编码
            scale_features = encoder(scaled_input)
            
            # 上采样回原尺寸
            if i > 0:
                scale_features = F.interpolate(
                    scale_features, 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            multiscale_features.append(scale_features)
        
        # 拼接多尺度特征
        fused_features = torch.cat(multiscale_features, dim=1)  # [B, total_channels, H, W]
        
        # 特征融合
        pixel_features_map = self.fusion_conv(fused_features)  # [B, feature_dim, H, W]
        
        # 空间注意力
        attention_maps = None
        if self.enable_attention:
            attention_maps = self.spatial_attention(pixel_features_map)
            pixel_features_map = pixel_features_map * attention_maps
        
        # 全局池化得到像素级特征向量
        pixel_features = self.global_pool(pixel_features_map).squeeze(-1).squeeze(-1)  # [B, feature_dim]
        
        # 差分检测（如果提供前一帧）
        diff_maps = None
        if prev_images is not None:
            diff_maps = self.diff_detector(images, prev_images)
        
        return {
            'pixel_features': pixel_features,
            'pixel_features_map': pixel_features_map,
            'attention_maps': attention_maps,
            'diff_maps': diff_maps,
            'multiscale_features': multiscale_features
        }

class PixelDifferenceDetector(nn.Module):
    """
    像素级差分检测器
    检测相邻帧之间的细微变化
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # 差分特征提取
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels//2, 3, padding=1),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//2, 1, 3, padding=1),
            nn.Sigmoid()  # 输出[0,1]范围的变化概率
        )
        
        # 边缘增强卷积
        self.edge_detector = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        # 初始化为边缘检测核
        edge_kernel = torch.tensor([[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]], dtype=torch.float32)
        self.edge_detector.weight.data = edge_kernel.unsqueeze(0)
        self.edge_detector.weight.requires_grad = False
    
    def forward(self, current_frame, prev_frame):
        """
        检测两帧之间的差异
        
        Args:
            current_frame: [B, C, H, W]
            prev_frame: [B, C, H, W]
            
        Returns:
            diff_map: [B, 1, H, W] 差分概率图
        """
        # 拼接当前帧和前一帧
        concat_frames = torch.cat([current_frame, prev_frame], dim=1)
        
        # 生成差分概率图
        diff_map = self.diff_encoder(concat_frames)
        
        # 应用边缘增强
        edge_enhanced = torch.abs(self.edge_detector(diff_map))
        diff_map = diff_map + 0.1 * edge_enhanced
        diff_map = torch.clamp(diff_map, 0, 1)
        
        return diff_map

class SpatialAttentionModule(nn.Module):
    """
    空间注意力模块
    关注重要的像素区域
    """
    def __init__(self, in_channels):
        super().__init__()
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//8, 1, 1),
            nn.Sigmoid()
        )
        
        # 全局上下文模块
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//4, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
            
        Returns:
            attention_map: [B, 1, H, W]
        """
        # 局部注意力
        local_attention = self.attention_conv(x)
        
        # 全局上下文
        global_context = self.global_context(x)
        
        # 结合局部和全局信息
        enhanced_features = x * global_context
        final_attention = self.attention_conv(enhanced_features)
        
        return final_attention

# 测试代码
if __name__ == "__main__":
    # 创建像素层编码器
    pixel_encoder = PixelLevelEncoder(
        in_channels=3,
        base_channels=64,
        num_scales=4,
        enable_attention=True
    )
    
    # 测试数据
    batch_size = 2
    height, width = 224, 224
    current_images = torch.randn(batch_size, 3, height, width)
    prev_images = torch.randn(batch_size, 3, height, width)
    
    # 前向传播
    with torch.no_grad():
        outputs = pixel_encoder(current_images, prev_images)
    
    print("像素层编码器输出:")
    print(f"像素特征维度: {outputs['pixel_features'].shape}")
    print(f"像素特征图维度: {outputs['pixel_features_map'].shape}")
    if outputs['attention_maps'] is not None:
        print(f"注意力图维度: {outputs['attention_maps'].shape}")
    if outputs['diff_maps'] is not None:
        print(f"差分图维度: {outputs['diff_maps'].shape}")
    print(f"多尺度特征数量: {len(outputs['multiscale_features'])}")
