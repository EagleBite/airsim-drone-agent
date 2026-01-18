from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from airsim_drone_agent.sim.connection import ConnectionManager, ConnectionConfig
from airsim_drone_agent.sim.flight_controller import FlightController
from airsim_drone_agent.sim.sensors import SensorHub
from airsim_drone_agent.enums import DroneCamera


class AirSimClient:
    """
    AirSim 客户端：最外层封装，组合所有功能模块
    - 连接管理 (ConnectionManager)
    - 飞行控制 (FlightController)
    - 数据获取 (SensorHub)
    - 提供统一接口给 agent/LLM
    """

    def __init__(self, ip: str = "127.0.0.1", vehicle_name: str = "", timeout_sec: float = 10.0):
        """
        初始化 AirSim 客户端
        
        Args:
            ip: AirSim 服务器 IP 地址
            vehicle_name: 载具名称
            timeout_sec: 连接超时时间
        """
        # 连接管理
        cfg = ConnectionConfig(ip=ip, vehicle_name=vehicle_name, timeout_sec=timeout_sec)
        self.connection = ConnectionManager(cfg)
        self.client = self.connection.client
        
        # 飞行控制
        self.flight = FlightController(self.client, vehicle_name=vehicle_name)
        
        # 数据获取
        self.sensors = SensorHub(self.client, vehicle_name=vehicle_name)

    @property
    def vehicle_name(self) -> str:
        """载具名称"""
        return self.connection.vehicle_name

    # -----------------------------
    # 连接管理接口
    # -----------------------------
    def connect(self) -> None:
        """连接 AirSim, 并开启 API 控制"""
        self.connection.connect()

    def ensure_api_control(self, enabled: bool = True) -> None:
        """开启/关闭 API 控制"""
        self.connection.ensure_api_control(enabled)

    def reset(self) -> None:
        """重置仿真"""
        self.connection.reset()

    def ping(self) -> bool:
        """检测连接是否可用"""
        return self.connection.ping()

    # -----------------------------
    # 飞行控制接口
    # -----------------------------
    def arm(self, armed: bool = True) -> None:
        """解锁/上锁"""
        self.flight.arm(armed)

    def takeoff(self, timeout_sec: float = 10.0) -> None:
        """起飞"""
        self.flight.takeoff(timeout_sec=timeout_sec)

    def land(self, timeout_sec: float = 20.0) -> None:
        """降落"""
        self.flight.land(timeout_sec=timeout_sec)

    def hover(self) -> None:
        """悬停"""
        self.flight.hover()

    def move_to_position(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float = 5.0,
        timeout_sec: float = 60.0,
    ) -> None:
        """移动到指定位置 (NED 坐标系)"""
        self.flight.move_to_position(x, y, z, velocity=velocity, timeout_sec=timeout_sec)

    def move_to_z(
        self,
        z: float,
        velocity: float = 2.0,
        timeout_sec: float = 30.0,
    ) -> None:
        """移动到指定高度"""
        self.flight.move_to_z(z, velocity=velocity, timeout_sec=timeout_sec)

    def move_by_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float,
    ) -> None:
        """按指定速度移动"""
        self.flight.move_by_velocity(vx, vy, vz, duration)

    def move_by_velocity_z(
        self,
        vx: float,
        vy: float,
        z: float,
        duration: float,
    ) -> None:
        """按指定速度移动，同时保持指定高度"""
        self.flight.move_by_velocity_z(vx, vy, z, duration)

    def rotate_to_yaw(
        self,
        yaw: float,
        margin: float = 5.0,
        timeout_sec: float = 30.0,
    ) -> None:
        """旋转到指定偏航角"""
        self.flight.rotate_to_yaw(yaw, margin=margin, timeout_sec=timeout_sec)

    def rotate_by_yaw_rate(
        self,
        yaw_rate: float,
        duration: float,
    ) -> None:
        """按偏航角速度旋转"""
        self.flight.rotate_by_yaw_rate(yaw_rate, duration)

    def move_on_path(
        self,
        path: list[tuple[float, float, float]],
        velocity: float = 5.0,
        timeout_sec: float = 120.0,
    ) -> None:
        """沿路径移动"""
        self.flight.move_on_path(path, velocity=velocity, timeout_sec=timeout_sec)

    # -----------------------------
    # 数据获取接口
    # -----------------------------
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
        获取观测数据
        
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
