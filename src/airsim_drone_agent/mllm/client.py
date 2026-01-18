from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .messages import Message

@dataclass
class MLLMResult:
    """统一返回：原始文本 + JSON"""
    text: str
    json: Optional[Dict[str, Any]] = None

class MLLMClient(ABC):

    @abstractmethod
    def generate(self, messages: List[Message], *, json_schema: Optional[Dict[str, Any]] = None, max_tokens: Optional[int] = None):
        """
        生成响应
        
        Args:
            messages: 消息列表
            json_schema: 可选的 JSON Schema，用于约束返回格式（某些 API 可能不支持）
            max_tokens: 可选的最大输出 token 数（某些 API 可能不支持）
        
        Returns:
            MLLMResult: 包含文本和解析后的 JSON
        """
        raise NotImplementedError