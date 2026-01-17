from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from airsim_drone_agent.sim.airsim_client import AirSimClient, ConnectionConfig
from airsim_drone_agent.sim.sensors import SensorHub
from airsim_drone_agent.enums import DroneCamera

class DroneEnv:
    """
    DroneEnv: 更高一层的“环境封装”
    - 管理连接
    - 提供统一观测 get_observation() 给 agent/LLM
    """

    def __init__(self, ip: str = "127.0.0.1", vehicle_name: str = "", timeout_sec: float = 10.0):
        cfg = ConnectionConfig(ip=ip, vehicle_name=vehicle_name, timeout_sec=timeout_sec)
        self.as_client = AirSimClient(cfg)
        self.client = self.as_client.client
        self.sensors = SensorHub(self.client, vehicle_name=vehicle_name)

    def connect(self) -> None:
        """连接 AirSim, 并开启 API 控制"""
        self.as_client.connect()

    def arm(self, armed: bool = True) -> None:
        """解锁/上锁"""
        self.as_client.arm(armed)

    def get_observation(
        self,
        with_imu: bool = True,
        with_gps: bool = False,
        with_baro: bool = False,
        with_mag: bool = False,
        with_lidar: bool = False,
        lidar_name: str = "LidarSensor1",
        with_rgb_meta: bool = False,
        with_depth_summary: bool = False,
        camera_name: str = "0",
        json_friendly: bool = True,
    ) -> Dict[str, Any]:
        """
        If json_friendly=True:
          - lidar returns summaries (count/min-dist) instead of raw points
          - depth returns summary stats instead of full image
          - rgb returns only metadata (width/height)
        """
        obs: Dict[str, Any] = {
            "time": {"unix_ms": int(time.time() * 1000)},
            "pose": self.sensors.vehicle_pose(),
            "state": self.sensors.multirotor_state(),
            "collision": self.sensors.collision_info(),
        }

        if with_imu:
            obs["imu"] = self.sensors.imu()
        if with_gps:
            obs["gps"] = self.sensors.gps()
        if with_baro:
            obs["barometer"] = self.sensors.barometer()
        if with_mag:
            obs["magnetometer"] = self.sensors.magnetometer()

        if with_lidar:
            cloud = self.sensors.lidar(lidar_name=lidar_name)
            if json_friendly:
                obs["lidar"] = self._lidar_summary(cloud.points, lidar_name, cloud.timestamp_ns)
            else:
                obs["lidar"] = {
                    "lidar_name": lidar_name,
                    "timestamp_ns": cloud.timestamp_ns,
                    "points": cloud.points,  # numpy array
                }

        if with_rgb_meta:
            frame = self.sensors.rgb(camera_name=camera_name)
            obs["rgb"] = {"width": frame.width, "height": frame.height, "type": frame.frame_type}

        if with_depth_summary:
            depth = self.sensors.depth(camera_name=camera_name)
            if json_friendly:
                obs["depth"] = {
                    "width": depth.width,
                    "height": depth.height,
                    "type": depth.frame_type,
                    "summary": {
                        "min": float(np.nanmin(depth.data)),
                        "max": float(np.nanmax(depth.data)),
                        "mean": float(np.nanmean(depth.data)),
                    },
                }
            else:
                obs["depth"] = {"width": depth.width, "height": depth.height, "type": depth.frame_type, "data": depth.data}

        return obs
    
    def get_rgb_frame(self, camera: DroneCamera = DroneCamera.FRONT_CENTER):
        """直接获取某个相机的 RGB 图像"""
        return self.sensors.rgb(camera=camera, as_bgr_for_cv=True)

    @staticmethod
    def _lidar_summary(points: np.ndarray, lidar_name: str, timestamp_ns: int) -> Dict[str, Any]:
        """将点云压缩成摘要: 点数 + 最近距离 (给决策层用更友好)"""
        if points.size == 0:
            return {"lidar_name": lidar_name, "timestamp_ns": timestamp_ns, "num_points": 0, "min_range": None}
        ranges = np.linalg.norm(points, axis=1)
        return {
            "lidar_name": lidar_name,
            "timestamp_ns": timestamp_ns,
            "num_points": int(points.shape[0]),
            "min_range": float(np.min(ranges)),
        }