import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, 
    CLIPTextModel, CLIPTokenizer,
    BertModel, BertTokenizer
)
from typing import List, Dict, Optional, Union
import numpy as np

class MultiModalTextEncoder(nn.Module):
    """
    多模态文本编码器
    支持多种预训练文本编码器，并针对GUI任务进行优化
    """
    def __init__(self, 
                 encoder_type="clip",  # "clip", "bert", "roberta", "sentence-transformer"
                 model_name=None,
                 output_dim=512,
                 freeze_backbone=False,
                 add_task_specific_layers=True):
        super().__init__()
        self.encoder_type = encoder_type
        self.output_dim = output_dim
        self.freeze_backbone = freeze_backbone
        
        # 根据类型初始化不同的编码器
        if encoder_type == "clip":
            model_name = model_name or "openai/clip-vit-base-patch32"
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_encoder = CLIPTextModel.from_pretrained(model_name)
            self.hidden_dim = self.text_encoder.config.hidden_size
            
        elif encoder_type == "bert":
            model_name = model_name or "bert-base-uncased"
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.text_encoder = BertModel.from_pretrained(model_name)
            self.hidden_dim = self.text_encoder.config.hidden_size
            
        elif encoder_type == "sentence-transformer":
            from sentence_transformers import SentenceTransformer
            model_name = model_name or "all-MiniLM-L6-v2"
            self.sentence_model = SentenceTransformer(model_name)
            self.hidden_dim = self.sentence_model.get_sentence_embedding_dimension()
            self.tokenizer = None  # SentenceTransformer handles tokenization internally
            
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
        
        # 冻结主干网络参数
        if freeze_backbone and hasattr(self, 'text_encoder'):
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # 任务特定层
        self.task_output_dim = 0
        if add_task_specific_layers:
            self.task_specific_layers = self._build_task_specific_layers()
            # LSTM 的隐藏维度是 hidden_dim // 4，这部分会在前向时与原始特征拼接
            self.task_output_dim = self.hidden_dim // 4
        else:
            self.task_specific_layers = nn.Identity()
        
        # 输出投影层
        projection_in_dim = self.hidden_dim + self.task_output_dim
        self.output_projection = nn.Sequential(
            nn.Linear(projection_in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # GUI任务特定的指令模板
        self.instruction_templates = {
            "click": "Click on the {element}",
            "type": "Type '{text}' in the {element}",
            "scroll": "Scroll {direction} on the {element}",
            "select": "Select {option} from the {element}",
            "navigate": "Navigate to {target}",
            "find": "Find the {element}",
            "wait": "Wait for {element} to appear"
        }
    
    def _build_task_specific_layers(self):
        """构建GUI任务特定的层"""
        return nn.Sequential(
            # 指令类型分类层
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 空间关系理解层
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.ReLU(),
            
            # 时序依赖建模层
            nn.LSTM(self.hidden_dim // 2, self.hidden_dim // 4, batch_first=True),
        )
    
    def encode_instruction(self, instructions: List[str], device=None) -> torch.Tensor:
        """
        编码指令文本
        
        Args:
            instructions: 指令文本列表
            device: 目标设备
            
        Returns:
            encoded_instructions: [batch_size, output_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        
        if self.encoder_type == "sentence-transformer":
            # 使用SentenceTransformer
            embeddings = self.sentence_model.encode(
                instructions, 
                convert_to_tensor=True,
                device=device
            )  # [batch_size, hidden_dim]
            
        else:
            # 使用Transformers模型
            # 分词
            encoded = self.tokenizer(
                instructions,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # 移动到指定设备
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            # 编码
            with torch.set_grad_enabled(not self.freeze_backbone):
                if self.encoder_type == "clip":
                    outputs = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # CLIP使用pooler_output
                    embeddings = outputs.pooler_output  # [batch_size, hidden_dim]
                    
                elif self.encoder_type == "bert":
                    outputs = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    # BERT使用CLS token的hidden state
                    embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # 应用任务特定层
        if hasattr(self, 'task_specific_layers') and not isinstance(self.task_specific_layers, nn.Identity):
            if isinstance(self.task_specific_layers[-1], nn.LSTM):
                # 如果最后一层是LSTM，需要特殊处理
                task_features = embeddings.unsqueeze(1)  # [batch_size, 1, hidden_dim]
                for layer in self.task_specific_layers[:-1]:
                    task_features = layer(task_features)
                lstm_output, _ = self.task_specific_layers[-1](task_features)
                task_features = lstm_output.squeeze(1)  # [batch_size, hidden_dim//4]
                # 拼接原始特征和任务特征
                embeddings = torch.cat([embeddings, task_features], dim=-1)
        
        # 输出投影
        final_embeddings = self.output_projection(embeddings)
        
        return final_embeddings
    
    def parse_instruction(self, instruction: str) -> Dict[str, str]:
        """
        解析指令，提取动作类型、目标元素等信息
        
        Args:
            instruction: 输入指令
            
        Returns:
            parsed: 解析结果字典
        """
        instruction_lower = instruction.lower().strip()
        
        # 简单的规则解析（实际应用中可以使用NLP模型）
        parsed = {
            "action": "unknown",
            "target": "",
            "text": "",
            "direction": "",
            "option": ""
        }
        
        # 识别动作类型
        if any(word in instruction_lower for word in ["click", "tap", "press"]):
            parsed["action"] = "click"
        elif any(word in instruction_lower for word in ["type", "enter", "input"]):
            parsed["action"] = "type"
        elif any(word in instruction_lower for word in ["scroll", "swipe"]):
            parsed["action"] = "scroll"
        elif any(word in instruction_lower for word in ["select", "choose"]):
            parsed["action"] = "select"
        elif any(word in instruction_lower for word in ["find", "locate", "search"]):
            parsed["action"] = "find"
        elif any(word in instruction_lower for word in ["wait", "until"]):
            parsed["action"] = "wait"
        elif any(word in instruction_lower for word in ["navigate", "go to"]):
            parsed["action"] = "navigate"
        
        # 提取目标元素（简化版）
        common_elements = [
            "button", "link", "text", "image", "input", "field", 
            "dropdown", "menu", "checkbox", "radio", "tab", "icon"
        ]
        for element in common_elements:
            if element in instruction_lower:
                parsed["target"] = element
                break
        
        # 提取方向信息
        if any(word in instruction_lower for word in ["up", "upward"]):
            parsed["direction"] = "up"
        elif any(word in instruction_lower for word in ["down", "downward"]):
            parsed["direction"] = "down"
        elif any(word in instruction_lower for word in ["left", "leftward"]):
            parsed["direction"] = "left"
        elif any(word in instruction_lower for word in ["right", "rightward"]):
            parsed["direction"] = "right"
        
        return parsed
    
    def generate_augmented_instructions(self, base_instruction: str, num_augmentations=3) -> List[str]:
        """
        生成数据增强的指令变体
        
        Args:
            base_instruction: 基础指令
            num_augmentations: 增强数量
            
        Returns:
            augmented_instructions: 增强指令列表
        """
        parsed = self.parse_instruction(base_instruction)
        action = parsed["action"]
        target = parsed["target"]
        
        augmentations = [base_instruction]  # 包含原始指令
        
        # 基于模板生成变体
        if action in self.instruction_templates and target:
            template = self.instruction_templates[action]
            
            # 同义词替换
            synonyms = {
                "click": ["tap", "press", "select"],
                "button": ["btn", "control", "widget"],
                "text": ["textbox", "input field", "text area"],
                "find": ["locate", "search for", "look for"]
            }
            
            for i in range(min(num_augmentations, 3)):
                augmented = template
                
                # 随机替换同义词
                for word, syns in synonyms.items():
                    if word in template.lower() and syns:
                        if i < len(syns):
                            augmented = augmented.replace(word, syns[i])
                
                augmented = augmented.format(element=target)
                if augmented not in augmentations:
                    augmentations.append(augmented)
        
        return augmentations[:num_augmentations + 1]

# 测试代码
if __name__ == "__main__":
    # 测试不同类型的文本编码器
    print("测试多模态文本编码器...")
    
    # 测试CLIP编码器
    try:
        clip_encoder = MultiModalTextEncoder(
            encoder_type="clip",
            output_dim=512,
            freeze_backbone=False
        )
        
        test_instructions = [
            "Click on the submit button",
            "Type 'hello world' in the text field",
            "Scroll down on the page",
            "Find the search icon"
        ]
        
        with torch.no_grad():
            embeddings = clip_encoder.encode_instruction(test_instructions)
        print(f"CLIP编码器输出维度: {embeddings.shape}")
        
        # 测试指令解析
        parsed = clip_encoder.parse_instruction("Click on the submit button")
        print(f"指令解析结果: {parsed}")
        
        # 测试数据增强
        augmented = clip_encoder.generate_augmented_instructions(
            "Click on the button", num_augmentations=3
        )
        print(f"增强指令: {augmented}")
        
    except Exception as e:
        print(f"CLIP测试失败: {e}")
    
    # 测试BERT编码器
    try:
        bert_encoder = MultiModalTextEncoder(
            encoder_type="bert",
            output_dim=512,
            freeze_backbone=False
        )
        
        with torch.no_grad():
            bert_embeddings = bert_encoder.encode_instruction(test_instructions[:2])
        print(f"BERT编码器输出维度: {bert_embeddings.shape}")
        
    except Exception as e:
        print(f"BERT测试失败: {e}")
    
    print("文本编码器测试完成！")
