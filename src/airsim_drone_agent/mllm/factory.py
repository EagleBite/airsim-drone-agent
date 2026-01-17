from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

from .client import MLLMClient

def create_mllm_client(env_path: str | None = None) -> MLLMClient:
    """统一入口：根据 .env 的 MLLM_PROVIDER 创建对应实现"""
    if env_path is None:
        root = Path(__file__).resolve().parents[3]
        env_path = str(root / ".env")
    
    load_dotenv(env_path)

    provider = (os.getenv("MLLM_PROVIDER") or "").strip().lower()
    if not provider:
        raise RuntimeError(f"未配置 MLLM_PROVIDER(.env 路径：{env_path})")
    
    if provider == "openai":
        from .providers.openai_provider import OpenAIProvider
        return OpenAIProvider.from_env(env_path=env_path)
    
    raise RuntimeError(f"不支持的 MLLM_PROVIDER={provider}")

