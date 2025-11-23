import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
from filterpy.kalman import KalmanFilter

class EnhancedTracker:
    """
    增强版多目标跟踪器，集成卡尔曼滤波和深度特征匹配
    替换原有的简单IoU跟踪器
    """
    def __init__(self, 
                 max_disappeared=10,
                 iou_threshold=0.3,
                 feature_threshold=0.5,
                 use_kalman=True):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.use_kalman = use_kalman
        
        self.next_id = 1
        self.tracks = {}  # track_id -> Track object
        self.disappeared = {}  # track_id -> disappeared_count
        
        # 特征匹配网络
        self.feature_matcher = FeatureMatcher()
        
    def reset(self):
        """重置追踪器状态"""
        self.next_id = 1
        self.tracks = {}
        self.disappeared = {}
    
    def update(self, detections: List[Dict], frame_idx: int) -> Dict[int, Dict]:
        """
        更新跟踪状态
        
        Args:
            detections: 检测结果列表
            frame_idx: 当前帧索引
            
        Returns:
            活跃轨迹字典 {track_id: track_info}
        """
        if len(detections) == 0:
            # 没有检测结果，增加所有轨迹的消失计数
            self._update_disappeared_tracks()
            return self._get_active_tracks()
        
        if len(self.tracks) == 0:
            # 初始化轨迹
            self._initialize_tracks(detections, frame_idx)
            return self._get_active_tracks()
        
        # 执行数据关联
        matched_pairs, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(
            detections, frame_idx
        )
        
        # 更新匹配的轨迹
        self._update_matched_tracks(matched_pairs, detections, frame_idx)
        
        # 处理未匹配的轨迹
        self._handle_unmatched_tracks(unmatched_tracks)
        
        # 创建新轨迹
        self._create_new_tracks(unmatched_detections, detections, frame_idx)
        
        # 清理消失太久的轨迹
        self._cleanup_lost_tracks()
        
        return self._get_active_tracks()
    
    def _initialize_tracks(self, detections: List[Dict], frame_idx: int):
        """初始化轨迹"""
        for detection in detections:
            track = Track(self.next_id, detection, frame_idx, use_kalman=self.use_kalman)
            self.tracks[self.next_id] = track
            self.disappeared[self.next_id] = 0
            self.next_id += 1
    
    def _associate_detections_to_tracks(self, detections: List[Dict], frame_idx: int) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        将检测结果与现有轨迹进行关联
        
        Returns:
            matched_pairs: [(detection_idx, track_id)]
            unmatched_detections: [detection_idx]  
            unmatched_tracks: [track_id]
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        # 构建成本矩阵
        cost_matrix = self._build_cost_matrix(detections, frame_idx)
        
        # 使用匈牙利算法进行最优匹配
        detection_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        matched_pairs = []
        for det_idx, track_idx in zip(detection_indices, track_indices):
            track_id = list(self.tracks.keys())[track_idx]
            cost = cost_matrix[det_idx, track_idx]
            
            # 检查成本是否在阈值内
            if cost < 1.0:  # 成本阈值
                matched_pairs.append((det_idx, track_id))
        
        # 找出未匹配的检测和轨迹
        matched_det_indices = {pair[0] for pair in matched_pairs}
        matched_track_ids = {pair[1] for pair in matched_pairs}
        
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_indices]
        unmatched_tracks = [tid for tid in self.tracks.keys() if tid not in matched_track_ids]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _build_cost_matrix(self, detections: List[Dict], frame_idx: int) -> np.ndarray:
        """构建检测-轨迹成本矩阵"""
        num_detections = len(detections)
        num_tracks = len(self.tracks)
        cost_matrix = np.full((num_detections, num_tracks), 1e6)
        
        track_list = list(self.tracks.keys())
        
        for det_idx, detection in enumerate(detections):
            for track_idx, track_id in enumerate(track_list):
                track = self.tracks[track_id]
                
                # 计算IoU成本
                iou_cost = 1.0 - self._calculate_iou(detection['bbox'], track.get_predicted_bbox())
                
                # 计算特征相似度成本
                feature_cost = 1.0 - self._calculate_feature_similarity(
                    detection.get('features', np.zeros(512)),
                    track.get_latest_features()
                )
                
                # 计算时间成本
                time_cost = min((frame_idx - track.last_frame_idx) / 30.0, 1.0)  # 归一化到[0,1]
                
                # 加权组合成本
                total_cost = 0.5 * iou_cost + 0.3 * feature_cost + 0.2 * time_cost
                
                cost_matrix[det_idx, track_idx] = total_cost
        
        return cost_matrix
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """计算两个边界框的IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_feature_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算特征相似度（余弦相似度）"""
        # 归一化特征向量
        feat1_norm = feat1 / (np.linalg.norm(feat1) + 1e-8)
        feat2_norm = feat2 / (np.linalg.norm(feat2) + 1e-8)
        
        # 计算余弦相似度
        similarity = np.dot(feat1_norm, feat2_norm)
        return max(0.0, similarity)  # 确保非负
    
    def _update_matched_tracks(self, matched_pairs: List[Tuple], detections: List[Dict], frame_idx: int):
        """更新匹配的轨迹"""
        for det_idx, track_id in matched_pairs:
            detection = detections[det_idx]
            self.tracks[track_id].update(detection, frame_idx)
            self.disappeared[track_id] = 0  # 重置消失计数
    
    def _handle_unmatched_tracks(self, unmatched_tracks: List[int]):
        """处理未匹配的轨迹"""
        for track_id in unmatched_tracks:
            self.disappeared[track_id] += 1
            # 使用卡尔曼滤波预测位置（如果启用）
            if self.use_kalman:
                self.tracks[track_id].predict()
    
    def _create_new_tracks(self, unmatched_detections: List[int], detections: List[Dict], frame_idx: int):
        """为未匹配的检测创建新轨迹"""
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            track = Track(self.next_id, detection, frame_idx, use_kalman=self.use_kalman)
            self.tracks[self.next_id] = track
            self.disappeared[self.next_id] = 0
            self.next_id += 1
    
    def _cleanup_lost_tracks(self):
        """清理消失太久的轨迹"""
        to_delete = []
        for track_id, disappeared_count in self.disappeared.items():
            if disappeared_count > self.max_disappeared:
                to_delete.append(track_id)
        
        for track_id in to_delete:
            del self.tracks[track_id]
            del self.disappeared[track_id]
    
    def _update_disappeared_tracks(self):
        """更新所有轨迹的消失计数"""
        for track_id in list(self.disappeared.keys()):
            self.disappeared[track_id] += 1
    
    def _get_active_tracks(self) -> Dict[int, Dict]:
        """获取活跃轨迹信息"""
        active_tracks = {}
        for track_id, track in self.tracks.items():
            active_tracks[track_id] = {
                'id': track_id,
                'observations': track.get_observations(),
                'current_bbox': track.get_current_bbox(),
                'confidence': track.confidence,
                'age': track.age
            }
        return active_tracks
    
    def export_tracks(self) -> List[Dict]:
        """导出轨迹数据"""
        tracks_data = []
        for track_id, track in self.tracks.items():
            tracks_data.append({
                'id': track_id,
                'observations': track.get_observations(),
                'total_length': len(track.get_observations()),
                'confidence': track.confidence
            })
        return tracks_data

class Track:
    """
    单个轨迹对象，包含卡尔曼滤波和特征历史
    """
    def __init__(self, track_id: int, initial_detection: Dict, frame_idx: int, use_kalman: bool = True):
        self.track_id = track_id
        self.observations = []
        self.features_history = []
        self.last_frame_idx = frame_idx
        self.confidence = initial_detection.get('score', 0.5)
        self.age = 1
        self.use_kalman = use_kalman
        
        # 初始化卡尔曼滤波器
        if use_kalman:
            self.kf = self._init_kalman_filter(initial_detection['bbox'])
        
        # 添加初始观测
        self.update(initial_detection, frame_idx)
    
    def _init_kalman_filter(self, initial_bbox: List[int]) -> KalmanFilter:
        """初始化卡尔曼滤波器"""
        # 状态向量: [x_center, y_center, width, height, dx, dy, dw, dh]
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # 状态转移矩阵
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy  
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
            [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx
            [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy
            [0, 0, 0, 0, 0, 0, 1, 0],  # dw = dw
            [0, 0, 0, 0, 0, 0, 0, 1],  # dh = dh
        ])
        
        # 观测矩阵
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # 观测x
            [0, 1, 0, 0, 0, 0, 0, 0],  # 观测y
            [0, 0, 1, 0, 0, 0, 0, 0],  # 观测w
            [0, 0, 0, 1, 0, 0, 0, 0],  # 观测h
        ])
        
        # 过程噪声协方差
        kf.Q *= 0.01
        
        # 观测噪声协方差
        kf.R *= 10
        
        # 初始状态
        x1, y1, x2, y2 = initial_bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        kf.x = np.array([x_center, y_center, width, height, 0, 0, 0, 0])
        
        return kf
    
    def update(self, detection: Dict, frame_idx: int):
        """更新轨迹"""
        bbox = detection['bbox']
        features = detection.get('features', np.zeros(512))
        
        # 更新观测历史
        self.observations.append({
            'frame_idx': frame_idx,
            'bbox': bbox,
            'score': detection.get('score', 0.5),
            'cls': detection.get('cls', 'unknown')
        })
        
        # 更新特征历史
        self.features_history.append(features)
        if len(self.features_history) > 10:  # 保持最近10帧的特征
            self.features_history.pop(0)
        
        # 更新卡尔曼滤波器
        if self.use_kalman:
            x1, y1, x2, y2 = bbox
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            measurement = np.array([x_center, y_center, width, height])
            self.kf.update(measurement)
        
        self.last_frame_idx = frame_idx
        self.age += 1
        
        # 更新置信度（指数移动平均）
        new_score = detection.get('score', 0.5)
        self.confidence = 0.7 * self.confidence + 0.3 * new_score
    
    def predict(self):
        """预测下一帧位置（仅在使用卡尔曼滤波时）"""
        if self.use_kalman:
            self.kf.predict()
    
    def get_predicted_bbox(self) -> List[int]:
        """获取预测的边界框"""
        if self.use_kalman and len(self.observations) > 0:
            state = self.kf.x
            x_center, y_center, width, height = state[:4]
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)  
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            return [x1, y1, x2, y2]
        elif len(self.observations) > 0:
            return self.observations[-1]['bbox']
        else:
            return [0, 0, 0, 0]
    
    def get_current_bbox(self) -> List[int]:
        """获取当前边界框"""
        if len(self.observations) > 0:
            return self.observations[-1]['bbox']
        return [0, 0, 0, 0]
    
    def get_latest_features(self) -> np.ndarray:
        """获取最新特征"""
        if len(self.features_history) > 0:
            return self.features_history[-1]
        return np.zeros(512)
    
    def get_observations(self) -> List[Dict]:
        """获取所有观测"""
        return self.observations.copy()

class FeatureMatcher(nn.Module):
    """
    特征匹配网络，用于计算检测特征之间的相似度
    """
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.matcher = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        计算两个特征之间的相似度
        
        Args:
            feat1: [N, feature_dim]
            feat2: [M, feature_dim]
            
        Returns:
            similarity_matrix: [N, M]
        """
        N, M = feat1.size(0), feat2.size(0)
        
        # 扩展特征用于配对
        feat1_expanded = feat1.unsqueeze(1).expand(N, M, -1)  # [N, M, feature_dim]
        feat2_expanded = feat2.unsqueeze(0).expand(N, M, -1)  # [N, M, feature_dim]
        
        # 拼接特征对
        feat_pairs = torch.cat([feat1_expanded, feat2_expanded], dim=-1)  # [N, M, 2*feature_dim]
        
        # 计算相似度
        similarity_matrix = self.matcher(feat_pairs.view(-1, self.feature_dim * 2))  # [N*M, 1]
        similarity_matrix = similarity_matrix.view(N, M)  # [N, M]
        
        return similarity_matrix

# 使用示例
if __name__ == "__main__":
    # 创建增强追踪器
    tracker = EnhancedTracker(use_kalman=True)
    
    # 模拟检测结果
    detections_frame1 = [
        {'bbox': [100, 100, 200, 150], 'score': 0.9, 'cls': 'button', 'features': np.random.rand(512)},
        {'bbox': [300, 200, 400, 250], 'score': 0.8, 'cls': 'text', 'features': np.random.rand(512)}
    ]
    
    detections_frame2 = [
        {'bbox': [105, 105, 205, 155], 'score': 0.85, 'cls': 'button', 'features': np.random.rand(512)},
        {'bbox': [295, 205, 395, 255], 'score': 0.82, 'cls': 'text', 'features': np.random.rand(512)}
    ]
    
    # 更新追踪器
    tracks_1 = tracker.update(detections_frame1, 1)
    tracks_2 = tracker.update(detections_frame2, 2)
    
    print(f"第1帧轨迹数: {len(tracks_1)}")
    print(f"第2帧轨迹数: {len(tracks_2)}")
    
    # 导出轨迹
    final_tracks = tracker.export_tracks()
    print(f"最终轨迹数: {len(final_tracks)}")