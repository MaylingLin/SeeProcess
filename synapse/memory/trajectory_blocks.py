"""Trajectory block segmentation and hierarchical memory module."""

from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrajectoryStep:
    """Represents a fine-grained interaction step within a trajectory."""

    step_id: str
    feature: torch.Tensor
    bbox: torch.Tensor
    cls_id: int
    timestamp: float
    semantic_tag: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryBlock:
    """Semantic block composed of multiple trajectory steps."""

    block_id: str
    steps: List[TrajectoryStep]
    summary: str
    priority: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def step_ids(self) -> List[str]:
        return [step.step_id for step in self.steps]

    def num_steps(self) -> int:
        return len(self.steps)


class TrajectoryBlockSegmenter:
    """Segments a long trajectory into blocks with semantic boundaries."""

    def __init__(
        self,
        min_block_size: int = 3,
        max_block_size: int = 24,
        similarity_threshold: float = 0.55,
        max_idle_gap: float = 25.0,
    ):
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.similarity_threshold = similarity_threshold
        self.max_idle_gap = max_idle_gap
        self._block_counter = itertools.count()

    def segment(
        self,
        steps: Sequence[TrajectoryStep],
        ops: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[TrajectoryBlock]:
        if not steps:
            return []

        boundary_frames = self._ops_to_boundaries(ops)
        blocks: List[TrajectoryBlock] = []
        current: List[TrajectoryStep] = []

        for step in steps:
            if not current:
                current.append(step)
                continue

            boundary = False
            boundary = boundary or len(current) >= self.max_block_size
            boundary = boundary or self._frame_boundary(step, boundary_frames)
            boundary = boundary or self._semantic_change(current[-1], step)
            boundary = boundary or self._feature_shift(current[-1], step)
            boundary = boundary or self._idle_gap(current[-1], step)

            if boundary and len(current) >= self.min_block_size:
                blocks.append(self._build_block(current))
                current = [step]
            else:
                current.append(step)

        if current:
            blocks.append(self._build_block(current))
        return blocks

    def _idle_gap(self, prev_step: TrajectoryStep, cur_step: TrajectoryStep) -> bool:
        return (cur_step.timestamp - prev_step.timestamp) > self.max_idle_gap

    def _frame_boundary(self, step: TrajectoryStep, boundary_frames: set[int]) -> bool:
        frame_idx = step.metadata.get("frame_idx")
        return frame_idx in boundary_frames

    def _semantic_change(self, prev_step: TrajectoryStep, cur_step: TrajectoryStep) -> bool:
        if not prev_step.semantic_tag or not cur_step.semantic_tag:
            return False
        return prev_step.semantic_tag != cur_step.semantic_tag

    def _feature_shift(self, prev_step: TrajectoryStep, cur_step: TrajectoryStep) -> bool:
        sim = self._cosine(prev_step.feature, cur_step.feature)
        return sim < self.similarity_threshold

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        if a.numel() == 0 or b.numel() == 0:
            return 1.0
        a_norm = F.normalize(a.view(1, -1), dim=-1)
        b_norm = F.normalize(b.view(1, -1), dim=-1)
        return float((a_norm * b_norm).sum())

    def _ops_to_boundaries(
        self, ops: Optional[Sequence[Dict[str, Any]]]
    ) -> set[int]:
        frames: set[int] = set()
        if not ops:
            return frames
        for op in ops:
            idx = op.get("frame_idx")
            if idx is None:
                continue
            try:
                frames.add(int(idx))
            except (TypeError, ValueError):
                continue
        return frames

    def _build_block(self, steps: List[TrajectoryStep]) -> TrajectoryBlock:
        block_id = f"block_{next(self._block_counter)}"
        summary = self._summarize_block(steps)
        duration = steps[-1].timestamp - steps[0].timestamp
        priority = len(steps) + 0.1 * max(duration, 0.0)
        metadata = {
            "start_time": steps[0].timestamp,
            "end_time": steps[-1].timestamp,
            "semantic_tags": list({step.semantic_tag for step in steps if step.semantic_tag}),
        }
        return TrajectoryBlock(
            block_id=block_id,
            steps=list(steps),
            summary=summary,
            priority=priority,
            metadata=metadata,
        )

    def _summarize_block(self, steps: Sequence[TrajectoryStep]) -> str:
        actions: List[str] = []
        for step in steps:
            action = step.metadata.get("operation", {}).get("type")
            if not action:
                action = step.semantic_tag or step.metadata.get("action")
            if not action:
                continue
            if not actions or actions[-1] != action:
                actions.append(str(action))
        if not actions:
            return f"{len(steps)} steps"
        return " -> ".join(actions[:4])


class SinePositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        if length > self.pe.size(0):
            raise ValueError("Requested position exceeds buffer.")
        return self.pe[:length].to(device)


class TrajectoryBlockEncoder(nn.Module):
    """Self-attention encoder to compress block-level semantics."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_positions: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = SinePositionalEncoding(hidden_dim, max_len=max_positions)
        self.post_norm = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, block_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if block_tensor.size(1) == 0:
            return torch.zeros(
                block_tensor.size(0), self.hidden_dim, device=block_tensor.device
            )
        x = self.input_proj(block_tensor)
        pos = self.pos_encoding(x.size(1), x.device)
        x = x + pos
        encoded = self.encoder(x, src_key_padding_mask=mask)
        pooled = self._masked_mean(encoded, ~mask)
        return self.post_norm(pooled)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(-1)
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-5)
        return summed / denom

    def encode_blocks(self, blocks: Sequence[TrajectoryBlock]) -> torch.Tensor:
        device = next(self.parameters()).device
        if not blocks:
            return torch.zeros(0, self.hidden_dim, device=device)
        max_len = max(block.num_steps() for block in blocks)
        tensor = torch.zeros(len(blocks), max_len, self.input_dim, device=device)
        padding_mask = torch.ones(len(blocks), max_len, dtype=torch.bool, device=device)
        for bid, block in enumerate(blocks):
            for sid, step in enumerate(block.steps):
                feat = step.feature.to(device)
                if feat.shape[-1] != self.input_dim:
                    feat = self._resize_feat(feat, self.input_dim)
                tensor[bid, sid] = feat
                padding_mask[bid, sid] = False
        return self.forward(tensor, padding_mask)

    @staticmethod
    def _resize_feat(feat: torch.Tensor, target_dim: int) -> torch.Tensor:
        out = torch.zeros(target_dim, device=feat.device, dtype=feat.dtype)
        length = min(target_dim, feat.shape[-1])
        out[:length] = feat[:length]
        return out


class MemoryLayer:
    """Stores embeddings for a particular memory level with retrieval."""

    def __init__(self, level_name: str, dim: int, capacity: int, device: Optional[str] = None):
        self.level_name = level_name
        self.dim = dim
        self.capacity = capacity
        self.device = torch.device(device or "cpu")
        self.entries: deque = deque()

    def add_entries(self, entries: Sequence[Dict[str, Any]]):
        if not entries:
            return
        for entry in entries:
            emb = entry["embedding"].detach().to(self.device)
            self.entries.append(
                {
                    "id": entry["id"],
                    "embedding": emb,
                    "metadata": entry.get("metadata", {}),
                    "level": self.level_name,
                }
            )
        while len(self.entries) > self.capacity:
            self.entries.popleft()

    def retrieve(self, query: torch.Tensor, top_k: int = 1) -> List[Dict[str, Any]]:
        if len(self.entries) == 0:
            return []
        if query.dim() != 1:
            raise ValueError("Query must be 1-D tensor")
        bank = torch.stack([e["embedding"] for e in self.entries], dim=0).to(query.device)
        sims = F.cosine_similarity(bank, query.unsqueeze(0), dim=-1)
        top_k = min(top_k, sims.numel())
        values, indices = torch.topk(sims, k=top_k)
        results = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            entry = self.entries[idx]
            results.append(
                {
                    "id": entry["id"],
                    "level": self.level_name,
                    "score": score,
                    "metadata": entry["metadata"],
                }
            )
        return results

    def clear(self):
        self.entries.clear()


class GlobalPlanner(nn.Module):
    """Aggregates block embeddings into a high-level memory summary."""

    def __init__(self, hidden_dim: int, text_dim: Optional[int] = None):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.post = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.text_adapter = nn.Linear(text_dim, hidden_dim) if text_dim else None

    def forward(
        self, block_embeddings: torch.Tensor, text_context: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if block_embeddings.size(1) == 0:
            zeros = torch.zeros(
                block_embeddings.size(0), self.query.numel(), device=block_embeddings.device
            )
            weights = torch.zeros(block_embeddings.size(0), 0, device=block_embeddings.device)
            return zeros, weights
        query = self.query.unsqueeze(0).unsqueeze(0).expand(
            block_embeddings.size(0), 1, -1
        )
        attn_out, weights = self.attn(query, block_embeddings, block_embeddings)
        summary = attn_out.squeeze(1)
        if text_context is not None and self.text_adapter is not None:
            summary = summary + self.text_adapter(text_context)
        return self.post(summary), weights.squeeze(1)


class HierarchicalTrajectoryMemory(nn.Module):
    """Hierarchical memory that exposes pluggable trajectory-block operations."""

    def __init__(
        self,
        base_feature_dim: int,
        text_feat_dim: int = 512,
        block_hidden_dim: int = 512,
        cls_embed_dim: int = 32,
        time_embed_dim: int = 16,
        max_low_steps: int = 256,
        max_mid_blocks: int = 64,
        max_high_entries: int = 32,
        context_dim: Optional[int] = None,
        similarity_threshold: float = 0.55,
    ):
        super().__init__()
        self.cls_embed_dim = cls_embed_dim
        self.time_embed_dim = time_embed_dim
        self.step_dim = base_feature_dim + 4 + cls_embed_dim + time_embed_dim
        self.segmenter = TrajectoryBlockSegmenter(similarity_threshold=similarity_threshold)
        self.block_encoder = TrajectoryBlockEncoder(self.step_dim, hidden_dim=block_hidden_dim)
        self.global_planner = GlobalPlanner(block_hidden_dim, text_dim=text_feat_dim)
        self.context_dim = context_dim or block_hidden_dim
        self.context_adapter = (
            nn.Identity()
            if self.context_dim == block_hidden_dim
            else nn.Linear(block_hidden_dim, self.context_dim)
        )
        self.cls_encoder = nn.Sequential(
            nn.Linear(1, cls_embed_dim),
            nn.LayerNorm(cls_embed_dim),
            nn.GELU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.low_level = MemoryLayer("low", self.step_dim, capacity=max_low_steps)
        self.mid_level = MemoryLayer("mid", block_hidden_dim, capacity=max_mid_blocks)
        self.high_level = MemoryLayer("high", block_hidden_dim, capacity=max_high_entries)
        self.block_links: Dict[str, Dict[str, Any]] = {}
        self.high_links: Dict[str, Dict[str, Any]] = {}
        self._id_gen = itertools.count()
        self.time_scale = 1000.0

    def forward(
        self,
        step_features: torch.Tensor,
        step_masks: torch.Tensor,
        spatial_positions: torch.Tensor,
        cls_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        ops: Optional[Sequence[Sequence[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        return self.update_with_batch(
            step_features,
            step_masks,
            spatial_positions,
            cls_ids,
            timestamps=timestamps,
            text_features=text_features,
            ops=ops,
        )

    def update_with_batch(
        self,
        step_features: torch.Tensor,
        step_masks: torch.Tensor,
        spatial_positions: torch.Tensor,
        cls_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        ops: Optional[Sequence[Sequence[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        batch_results = []
        summaries = []
        cross_links = []
        B = step_features.size(0)
        text_features = text_features if text_features is not None else None
        for idx in range(B):
            sample_ops = ops[idx] if ops is not None and idx < len(ops) else None
            sample_text = text_features[idx] if text_features is not None else None
            sample = self._process_single_sequence(
                step_features[idx],
                step_masks[idx],
                spatial_positions[idx],
                cls_ids[idx],
                timestamps[idx] if timestamps is not None else None,
                text_context=sample_text,
                ops=sample_ops,
            )
            batch_results.append(sample)
            summaries.append(sample["global_summary"])
            cross_links.append(sample["cross_links"])

        block_embeddings, block_mask = self._pad_blocks(batch_results)
        global_summary = torch.stack(summaries, dim=0) if summaries else torch.zeros(0)
        contextual_summary = self.context_adapter(global_summary)
        return {
            "block_embeddings": block_embeddings,
            "block_mask": block_mask,
            "global_summary": global_summary,
            "contextual_summary": contextual_summary,
            "cross_layer_links": cross_links,
        }

    def _process_single_sequence(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        positions: torch.Tensor,
        cls_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor],
        text_context: Optional[torch.Tensor],
        ops: Optional[Sequence[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        steps = self._build_steps(features, masks, positions, cls_ids, timestamps, ops)
        blocks = self.segmenter.segment(steps, ops)
        block_embeddings = self.block_encoder.encode_blocks(blocks)
        block_entries = []
        for block, embedding in zip(blocks, block_embeddings):
            metadata = {
                "summary": block.summary,
                "num_steps": block.num_steps(),
                "priority": block.priority,
                "step_ids": block.step_ids,
                "start_time": block.metadata.get("start_time"),
                "end_time": block.metadata.get("end_time"),
            }
            block_entries.append({"id": block.block_id, "embedding": embedding, "metadata": metadata})
            self.block_links[block.block_id] = metadata
        self.mid_level.add_entries(block_entries)

        planner_input = (
            block_embeddings.unsqueeze(0)
            if block_embeddings.dim() == 2
            else block_embeddings.new_zeros(1, 0, self.block_encoder.hidden_dim)
        )
        global_summary, attn_weights = self.global_planner(
            planner_input,
            text_context=text_context.unsqueeze(0) if text_context is not None else None,
        )
        global_summary = global_summary.squeeze(0)
        high_id = self._make_id("high")
        high_metadata = {
            "attended_blocks": [entry["id"] for entry in block_entries],
            "attention": attn_weights.detach().cpu().tolist(),
        }
        self.high_level.add_entries(
            [{"id": high_id, "embedding": global_summary, "metadata": high_metadata}]
        )
        self.high_links[high_id] = high_metadata
        block_mask = torch.zeros(block_embeddings.size(0), dtype=torch.bool, device=block_embeddings.device)
        return {
            "block_embeddings": block_embeddings,
            "block_mask": block_mask,
            "global_summary": global_summary,
            "cross_links": {
                "high_id": high_id,
                "block_ids": [entry["id"] for entry in block_entries],
                "block_to_steps": {
                    entry["id"]: self.block_links.get(entry["id"], {}).get("step_ids", [])
                    for entry in block_entries
                },
            },
        }

    def _build_steps(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        positions: torch.Tensor,
        cls_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor],
        ops: Optional[Sequence[Dict[str, Any]]],
    ) -> List[TrajectoryStep]:
        steps: List[TrajectoryStep] = []
        ops_by_frame = {int(op.get("frame_idx")): op for op in ops or [] if op.get("frame_idx") is not None}
        time_grid = self._prepare_timestamps(features, timestamps)
        for node_idx in range(features.size(0)):
            for t_idx in range(features.size(1)):
                if masks[node_idx, t_idx] < 0.5:
                    continue
                base_feat = features[node_idx, t_idx]
                bbox = positions[node_idx, t_idx]
                cls_value = cls_ids[node_idx, t_idx]
                timestamp = time_grid[node_idx, t_idx]
                fused = self._fuse_step_feature(base_feat, bbox, cls_value, timestamp)
                step_id = self._make_id("step")
                frame_idx = int(timestamp.item())
                op_meta = ops_by_frame.get(frame_idx)
                semantic = None
                if op_meta:
                    semantic = op_meta.get("type") or op_meta.get("op")
                metadata = {
                    "node_index": int(node_idx),
                    "step_index": int(t_idx),
                    "frame_idx": frame_idx,
                    "operation": op_meta or {},
                }
                steps.append(
                    TrajectoryStep(
                        step_id=step_id,
                        feature=fused,
                        bbox=bbox,
                        cls_id=int(cls_value.item()),
                        timestamp=float(frame_idx),
                        semantic_tag=semantic,
                        metadata=metadata,
                    )
                )
                self.low_level.add_entries([{"id": step_id, "embedding": fused, "metadata": metadata}])
        return steps

    def _fuse_step_feature(
        self,
        feature: torch.Tensor,
        bbox: torch.Tensor,
        cls_value: torch.Tensor,
        timestamp: torch.Tensor,
    ) -> torch.Tensor:
        cls_tensor = cls_value.float().view(1, 1)
        cls_emb = self.cls_encoder(cls_tensor).squeeze(0)
        time_tensor = (timestamp.float() / self.time_scale).view(1, 1)
        time_emb = self.time_encoder(time_tensor).squeeze(0)
        return torch.cat([feature, bbox, cls_emb, time_emb], dim=-1)

    def _prepare_timestamps(
        self, features: torch.Tensor, timestamps: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if timestamps is not None:
            return timestamps.float()
        seq = torch.arange(features.size(1), device=features.device).float()
        return seq.unsqueeze(0).expand(features.size(0), -1)

    def _pad_blocks(self, batch_results: List[Dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        if not batch_results:
            device = next(self.block_encoder.parameters()).device
            return (
                torch.zeros(0, 0, self.block_encoder.hidden_dim, device=device),
                torch.zeros(0, 0, dtype=torch.bool, device=device),
            )
        max_blocks = max(result["block_embeddings"].size(0) for result in batch_results)
        device = batch_results[0]["block_embeddings"].device
        if max_blocks == 0:
            padded = torch.zeros(
                len(batch_results), 1, self.block_encoder.hidden_dim, device=device
            )
            mask = torch.ones(len(batch_results), 1, dtype=torch.bool, device=device)
            return padded, mask
        padded_blocks = []
        masks = []
        for result in batch_results:
            emb = result["block_embeddings"]
            mask = result["block_mask"]
            if emb.size(0) < max_blocks:
                padding = emb.new_zeros(max_blocks - emb.size(0), emb.size(1))
                emb = torch.cat([emb, padding], dim=0)
                pad_mask = torch.ones(max_blocks - mask.size(0), dtype=torch.bool, device=emb.device)
                mask = torch.cat([mask, pad_mask], dim=0)
            padded_blocks.append(emb)
            masks.append(mask)
        return torch.stack(padded_blocks, dim=0), torch.stack(masks, dim=0)

    def retrieve(self, query: torch.Tensor, level: str = "auto", top_k: int = 3):
        if query.dim() != 1:
            raise ValueError("Query must be a 1-D tensor")
        target_level = level
        if level == "auto":
            if query.size(0) == self.step_dim:
                target_level = "low"
            elif query.size(0) == self.block_encoder.hidden_dim:
                target_level = "mid"
            else:
                target_level = "high"
        if target_level == "low":
            return self.low_level.retrieve(query, top_k)
        if target_level == "mid":
            return self.mid_level.retrieve(query, top_k)
        return self.high_level.retrieve(query, top_k)

    def reset_memory(self):
        self.low_level.clear()
        self.mid_level.clear()
        self.high_level.clear()
        self.block_links.clear()
        self.high_links.clear()

    def _make_id(self, prefix: str) -> str:
        return f"{prefix}_{next(self._id_gen)}"
