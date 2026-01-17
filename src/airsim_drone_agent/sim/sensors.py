from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import airsim
import numpy as np

from airsim_drone_agent.enums import DroneCamera

@dataclass
class ImageFrame:
    """图像帧数据容器"""
    width: int
    height: int
    frame_type: str   # "scene" / "depth" / "segmentation" ...
    data: np.ndarray  # uint8(H,W,3) or float32(H,W)

@dataclass
class LidarCloud:
    """激光雷达点云容器"""
    lidar_name: str
    timestamp_ns: int
    points: np.ndarray  # (N,3) float32

class SensorHub:
    """
    传感器数据中心：只负责“调用 AirSim API + 数据解析
    ”"""

    def __init__(self, client: airsim.MultirotorClient, vehicle_name: str = ""):
        self.client = client
        self.vehicle_name = vehicle_name

    # -----------------------------
    # 车辆状态：位置/速度/加速度等
    # -----------------------------
    def multirotor_state(self) -> Dict[str, Any]:
        s = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = s.kinematics_estimated
        return {
            "timestamp_ns": int(getattr(s, "timestamp", 0)),
            "landed_state": int(getattr(s, "landed_state", -1)),
            "ready": bool(getattr(s, "ready", True)) if hasattr(s, "ready") else True,
            "kinematics": self._kinematics(kin),
        }
    
    def vehicle_pose(self) -> Dict[str, Any]:
        p = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        return {
            "position_ned": self._vec3(p.position),
            "orientation_quat": self._quat(p.orientation),
        }
    
    def collision_info(self) -> Dict[str, Any]:
        c = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        impact_point = getattr(c, "impact_point", airsim.Vector3r(0, 0, 0))
        normal = getattr(c, "normal", airsim.Vector3r(0, 0, 0))
        return {
            "has_collided": bool(getattr(c, "has_collided", False)),
            "object_name": str(getattr(c, "object_name", "")),
            "impact_point_ned": self._vec3(impact_point),
            "normal_ned": self._vec3(normal),
            "timestamp_ns": int(getattr(c, "time_stamp", 0)),
        }
    
     # -----------------------------
    # IMU / GPS / 气压计 / 磁力计
    # -----------------------------
    def imu(self) -> Dict[str, Any]:
        d = self.client.getImuData(vehicle_name=self.vehicle_name)
        return {
            "timestamp_ns": int(getattr(d, "time_stamp", 0)),
            "angular_velocity": self._vec3(d.angular_velocity),
            "linear_acceleration": self._vec3(d.linear_acceleration),
            "orientation_quat": self._quat(d.orientation),
        }
    
    def gps(self) -> Dict[str, Any]:
        d = self.client.getGpsData(vehicle_name=self.vehicle_name)
        gp = d.gnss.geo_point
        vel = d.gnss.velocity
        return {
            "timestamp_ns": int(getattr(d, "time_stamp", 0)),
            "geo": {"lat": float(gp.latitude), "lon": float(gp.longitude), "alt": float(gp.altitude)},
            "velocity": self._vec3(vel),
            "eph": float(getattr(d.gnss, "eph", np.nan)),
            "epv": float(getattr(d.gnss, "epv", np.nan)),
        }
    
    def barometer(self) -> Dict[str, Any]:
        d = self.client.getBarometerData(vehicle_name=self.vehicle_name)
        return {
            "timestamp_ns": int(getattr(d, "time_stamp", 0)),
            "altitude": float(getattr(d, "altitude", np.nan)),
            "pressure": float(getattr(d, "pressure", np.nan)),
            "qnh": float(getattr(d, "qnh", np.nan)),
        }

    def magnetometer(self) -> Dict[str, Any]:
        d = self.client.getMagnetometerData(vehicle_name=self.vehicle_name)
        mf = getattr(d, "magnetic_field_body", None) or getattr(d, "magnetic_field", None)
        return {
            "timestamp_ns": int(getattr(d, "time_stamp", 0)),
            "magnetic_field": self._vec3(mf),
        }
    
    # -----------------------------
    # 激光雷达
    # -----------------------------
    def lidar(self, lidar_name: str = "LidarSensor1") -> LidarCloud:
        d = self.client.getLidarData(lidar_name=lidar_name, vehicle_name=self.vehicle_name)
        pts = np.array(d.point_cloud, dtype=np.float32)
        if pts.size == 0:
            pts = np.zeros((0, 3), dtype=np.float32)
        else:
            pts = pts.reshape((-1, 3))
        return LidarCloud(lidar_name=lidar_name, timestamp_ns=int(getattr(d, "time_stamp", 0)), points=pts)

    # -----------------------------
    # 相机图像
    # -----------------------------
    @staticmethod
    def _camera_name(camera: str | DroneCamera) -> str:
        if isinstance(camera, DroneCamera):
            return camera.airsim_name
        return str(camera)

    def rgb(self, camera: str | DroneCamera = "0", as_bgr_for_cv: bool = True) -> ImageFrame:
        camera_name = self._camera_name(camera)
        req = [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False)]
        res = self.client.simGetImages(req, vehicle_name=self.vehicle_name)[0]
        img = np.frombuffer(res.image_data_uint8, dtype=np.uint8).reshape((res.height, res.width, 3))
        if as_bgr_for_cv:
            img = img[:, :, ::-1]  # RGB -> BGR
        return ImageFrame(width=res.width, height=res.height, frame_type="scene", data=img)

    def depth(self, camera: str | DroneCamera = "0") -> ImageFrame:
        camera_name = self._camera_name(camera)
        req = [airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)]
        res = self.client.simGetImages(req, vehicle_name=self.vehicle_name)[0]
        depth = np.array(res.image_data_float, dtype=np.float32).reshape((res.height, res.width))
        return ImageFrame(width=res.width, height=res.height, frame_type="depth", data=depth)
    
    # -----------------------------
    # 辅助：格式转换
    # -----------------------------
    @staticmethod
    def _vec3(v) -> Dict[str, float]:
        return {"x": float(v.x_val), "y": float(v.y_val), "z": float(v.z_val)}

    @staticmethod
    def _quat(q) -> Dict[str, float]:
        return {"w": float(q.w_val), "x": float(q.x_val), "y": float(q.y_val), "z": float(q.z_val)}

    def _kinematics(self, kin) -> Dict[str, Any]:
        return {
            "position_ned": self._vec3(kin.position),
            "orientation_quat": self._quat(kin.orientation),
            "linear_velocity_ned": self._vec3(kin.linear_velocity),
            "angular_velocity_body": self._vec3(kin.angular_velocity),
            "linear_acceleration_body": self._vec3(kin.linear_acceleration),
            "angular_acceleration_body": self._vec3(kin.angular_acceleration),
        }

    

