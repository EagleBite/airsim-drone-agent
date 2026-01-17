from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    DEVELOPER = "developer"

@dataclass(frozen=True)
class TextPart:
    text: str
    type: str = "text"

@dataclass(frozen=True)
class ImagePart:
    """
    图片片段（通用表示）
    - url: 既可以是 http(s) 图片链接，也可以是 base64 的 data URL (data:image/jpeg;base64,...)
    - name: 给图片一个标签(例如 FRONT_LEFT), 方便在 prompt 里引用
    """
    url: str
    name: Optional[str] = None
    type: str = "image_url"

ContentPart = Union[TextPart, ImagePart]

@dataclass
class Message:
    """一条消息: role + 多个内容片段(text/image)"""
    role: Role
    content: List[ContentPart]

    @staticmethod
    def system(text: str) -> "Message":
        return Message(role=Role.SYSTEM, content=[TextPart(text=text)])
    
    @staticmethod
    def developer(text: str) -> "Message":
        return Message(role=Role.DEVELOPER, content=[TextPart(text=text)])

    @staticmethod
    def user(parts: List[ContentPart]) -> "Message":
        return Message(role=Role.USER, content=parts)

    @staticmethod
    def assistant(text: str) -> "Message":
        return Message(role=Role.ASSISTANT, content=[TextPart(text=text)])
