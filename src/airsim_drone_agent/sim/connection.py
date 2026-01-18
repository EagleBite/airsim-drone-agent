from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import airsim


@dataclass
class ConnectionConfig:
    """AirSim 连接配置"""
    ip: str = "127.0.0.1"
    vehicle_name: str = ""
    timeout_sec: float = 10.0
    retry_interval_sec: float = 0.5
    max_retries: int = 20


class ConnectionManager:
    """
    连接管理器：专门负责 AirSim 连接管理
    - 连接/重连
    - API 控制
    - 连接状态检测
    """

    def __init__(self, cfg: Optional[ConnectionConfig] = None):
        self.cfg = cfg or ConnectionConfig()
        self.client = airsim.MultirotorClient(ip=self.cfg.ip)

    @property
    def vehicle_name(self) -> str:
        return self.cfg.vehicle_name
    
    def connect(self) -> None:
        """连接到 AirSim 并开启 API 控制"""
        last_err: Optional[Exception] = None
        for _ in range(self.cfg.max_retries):
            try:
                self.client.confirmConnection()
                self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
                return
            except Exception as e:
                last_err = e
                time.sleep(self.cfg.retry_interval_sec)
        raise RuntimeError(f"Failed to connect to AirSim at {self.cfg.ip}") from last_err
    
    def ensure_api_control(self, enabled: bool = True) -> None:
        """开启/关闭 API 控制"""
        self.client.enableApiControl(enabled, vehicle_name=self.vehicle_name)

    def reset(self) -> None:
        """重置仿真 (reset 后通常会关闭 API 控制，需要重新 enable)"""
        self.client.reset()
        time.sleep(0.2)
        self.ensure_api_control(True)

    def ping(self) -> bool:
        """轻量检测连接是否可用: 调用一次 getMultirotorState"""
        try:
            _ = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            return True
        except Exception:
            return False
