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
    def generate(self, messages: List[Message], *, 
                 json_schema: Optional[Dict[str, Any]] = None, max_tokens: int = 400):
        raise NotImplementedError