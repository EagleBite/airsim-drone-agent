from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from ..client import MLLMClient, MLLMResult
from ..messages import ImagePart, Message, Role, TextPart

@dataclass
class _Cfg:
    base_url: str
    api_key: str
    model: str
    
class OpenAIProvider(MLLMClient):

    def __init__(self, cfg: _Cfg):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    @classmethod
    def from_env(cls, env_path: str = ".env") -> "OpenAIProvider":
        load_dotenv(env_path)
        base_url = os.getenv("MLLM_BASE_URL")
        api_key = os.getenv("MLLM_API_KEY")
        model = os.getenv("MLLM_MODEL", "gpt-4.1-mini")
        if not api_key:
            raise RuntimeError("未在 .env 中找到 MLLM_API_KEY")
        return cls(_Cfg(base_url=base_url, api_key=api_key, model=model))
    
    def _to_openai_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for m in messages:
            role = m.role.value
            
            parts: List[Dict[str, Any]] = []
            for p in m.content:
                if isinstance(p, TextPart):
                    parts.append({"type": "text", "text": p.text})
                elif isinstance(p, ImagePart):
                    parts.append({
                        "type": "image_url",
                        "image_url": { "url": p.url },
                    })
                else:
                    raise TypeError(f"未知 ContentPart: {type(p)}")
                
            out.append({"role": role, "content": parts})

        return out

    
    def generate(self, messages, *, json_schema = None, max_tokens = 400):
        openai_messages = self._to_openai_messages(messages)

        kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": openai_messages,
        }

        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        
        kwargs["max_completion_tokens"] = max_tokens
            
        resp = self.client.chat.completions.create(**kwargs)

        text = (resp.choices[0].message.content or "").strip()

        parsed = None
        if text:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

        return MLLMResult(text=text, json=parsed)