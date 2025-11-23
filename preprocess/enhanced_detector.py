import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from typing import List, Dict, Tuple
import torch.nn as nn
from ultralytics import YOLO

class GUIElementDetector(nn.Module):
    """
    增强版GUI元素检测器，支持多种检测后端
    替换原有的DetectorStub，提供真实的GUI元素检测能力
    """
    def __init__(self, model_type="yolo", model_path=None):
        super().__init__()
        self.model_type = model_type
        
        if model_type == "yolo":
            # 使用YOLO进行GUI元素检测
            self.model = YOLO(model_path or "yolov8n.pt")
            self.gui_classes = ["button", "text", "image", "icon", "input", "dropdown", "checkbox"]
        elif model_type == "detectron2":
            # 可选择Detectron2后端
            self._init_detectron2(model_path)
        
        # 预处理转换
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _init_detectron2(self, model_path):
        """初始化Detectron2模型（可选实现）"""
        pass
    
    def detect(self, img: np.ndarray) -> List[Dict]:
        """
        检测GUI界面中的交互元素
        
        Args:
            img: 输入图像 (H, W, 3) numpy array
            
        Returns:
            检测结果列表，每个包含 bbox, score, cls, features
        """
        if self.model_type == "yolo":
            return self._detect_yolo(img)
        elif self.model_type == "detectron2":
            return self._detect_detectron2(img)
    
    def _detect_yolo(self, img: np.ndarray) -> List[Dict]:
        """使用YOLO进行检测"""
        results = self.model(img)
        detections = []
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    score = boxes.conf[i].cpu().numpy()
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    # 提取区域特征
                    roi = img[int(y1):int(y2), int(x1):int(x2)]
                    features = self._extract_roi_features(roi)
                    
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "score": float(score),
                        "cls": self.gui_classes[cls_id] if cls_id < len(self.gui_classes) else "widget",
                        "features": features,
                        "roi_shape": roi.shape[:2] if roi.size > 0 else (0, 0)
                    })
        
        return detections
    
    def _extract_roi_features(self, roi: np.ndarray) -> np.ndarray:
        """从检测到的ROI区域提取特征"""
        if roi.size == 0:
            return np.zeros(512)  # 返回零特征向量
        
        # 颜色直方图特征
        hist_b = cv2.calcHist([roi], [0], None, [16], [0, 256])
        hist_g = cv2.calcHist([roi], [1], None, [16], [0, 256])
        hist_r = cv2.calcHist([roi], [2], None, [16], [0, 256])
        color_feat = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        
        # 纹理特征 (LBP)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        lbp_feat = self._compute_lbp_features(gray)
        
        # 几何特征
        h, w = roi.shape[:2]
        aspect_ratio = w / h if h > 0 else 0
        area = w * h
        geom_feat = np.array([aspect_ratio, area, w, h])
        
        # 拼接所有特征
        features = np.concatenate([
            color_feat / (color_feat.sum() + 1e-6),  # 归一化颜色特征
            lbp_feat,
            geom_feat / 1000.0  # 归一化几何特征
        ])
        
        # 确保特征向量长度一致
        target_len = 512
        if len(features) < target_len:
            features = np.pad(features, (0, target_len - len(features)))
        else:
            features = features[:target_len]
        
        return features
    
    def _compute_lbp_features(self, gray: np.ndarray) -> np.ndarray:
        """计算LBP纹理特征"""
        # 简化的LBP实现
        if gray.size == 0:
            return np.zeros(256)
        
        # 计算局部二值模式
        padded = np.pad(gray, 1, mode='edge')
        lbp = np.zeros_like(gray)
        
        for i in range(1, padded.shape[0] - 1):
            for j in range(1, padded.shape[1] - 1):
                center = padded[i, j]
                pattern = 0
                for k, (di, dj) in enumerate([(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]):
                    if padded[i+di, j+dj] >= center:
                        pattern += (1 << k)
                lbp[i-1, j-1] = pattern
        
        # 计算LBP直方图
        hist, _ = np.histogram(lbp, bins=256, range=(0, 255))
        return hist / (hist.sum() + 1e-6)

class TemporalDifferenceModeler:
    """
    时序差分建模器，用于捕获操作引起的界面变化
    """
    def __init__(self):
        self.target_size = (224, 224)
        self.diff_encoder = self._build_diff_encoder()
    
    def _build_diff_encoder(self):
        """构建差分编码网络"""
        return nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, 256)
        )
    
    def compute_temporal_diff(self, 
                            frames_before: List[np.ndarray], 
                            frames_after: List[np.ndarray],
                            action_event: Dict) -> np.ndarray:
        """
        计算操作前后的时序差分特征
        
        Args:
            frames_before: 操作前的帧序列
            frames_after: 操作后的帧序列  
            action_event: 操作事件信息 {type: str, position: (x,y), timestamp: float}
            
        Returns:
            时序差分特征向量
        """
        # 构建差分体积
        diff_volume = self._build_diff_volume(frames_before, frames_after)
        
        # 提取操作相关的空间注意力
        if not frames_before:
            raise ValueError("frames_before 不能为空")
        attention_mask = self._compute_action_attention(frames_before[0], action_event)
        attention_mask = attention_mask[np.newaxis, np.newaxis, :, :].astype(diff_volume.dtype, copy=False)
        
        # 应用空间注意力
        diff_volume = diff_volume * attention_mask
        
        # 通过3D CNN编码差分特征
        diff_tensor = torch.tensor(diff_volume).unsqueeze(0).float()  # (1, C, T, H, W)
        with torch.no_grad():
            diff_features = self.diff_encoder(diff_tensor)
        
        return diff_features.squeeze().numpy()
    
    def _build_diff_volume(self, frames_before: List[np.ndarray], frames_after: List[np.ndarray]) -> np.ndarray:
        """构建时序差分体积"""
        # 处理before帧
        before_frames = []
        for frame in frames_before[-3:]:  # 取最后3帧
            resized = cv2.resize(frame, self.target_size)
            before_frames.append(resized.astype(np.float32))
        
        # 处理after帧  
        after_frames = []
        for frame in frames_after[:3]:  # 取前3帧
            resized = cv2.resize(frame, self.target_size)
            after_frames.append(resized.astype(np.float32))
        
        # 计算逐帧差分
        diff_frames = []
        min_len = min(len(before_frames), len(after_frames))
        for i in range(min_len):
            diff = cv2.absdiff(after_frames[i], before_frames[i])
            diff_frames.append(diff)
        
        # 构建差分体积 (C, T, H, W)
        if diff_frames:
            diff_volume = np.stack(diff_frames, axis=0)  # (T, H, W, C)
            diff_volume = diff_volume.transpose(3, 0, 1, 2)  # (C, T, H, W)
        else:
            diff_volume = np.zeros((3, 1, *self.target_size), dtype=np.float32)
        
        return diff_volume
    
    def _compute_action_attention(self, base_frame: np.ndarray, action_event: Dict) -> np.ndarray:
        """计算操作位置的空间注意力掩码"""
        h, w = base_frame.shape[:2]
        attention_mask = np.zeros((h, w), dtype=np.float32)
        
        if 'position' in action_event:
            x, y = action_event['position']
            x, y = int(x), int(y)
            
            # 创建高斯分布的注意力掩码
            sigma = 50  # 注意力半径
            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((i - y)**2 + (j - x)**2)
                    attention_mask[i, j] = np.exp(-(dist**2) / (2 * sigma**2))
        else:
            attention_mask = np.ones((h, w), dtype=np.float32)
        
        if attention_mask.shape != self.target_size:
            attention_mask = cv2.resize(attention_mask, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        return attention_mask

# 使用示例和测试代码
if __name__ == "__main__":
    # 初始化增强检测器
    detector = GUIElementDetector(model_type="yolo")
    diff_modeler = TemporalDifferenceModeler()
    
    # 测试图像
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 测试检测
    detections = detector.detect(test_img)
    print(f"检测到 {len(detections)} 个GUI元素")
    
    # 测试时序差分
    frames_before = [test_img for _ in range(3)]
    frames_after = [test_img + np.random.randint(-10, 10, test_img.shape, dtype=np.int16) for _ in range(3)]
    action_event = {"type": "click", "position": (320, 240), "timestamp": 0.5}
    
    diff_features = diff_modeler.compute_temporal_diff(frames_before, frames_after, action_event)
    print(f"时序差分特征维度: {diff_features.shape}")
