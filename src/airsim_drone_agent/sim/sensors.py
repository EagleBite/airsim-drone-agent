from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

class SensorHub:
    """
    传感器数据中心：负责调用 AirSim API + 数据解析 + 感知信息获取
    - 每个感知功能都是独立的函数
    - 不进行数据聚合，按需获取
    """

    def __init__(self, client: airsim.MultirotorClient, vehicle_name: str = ""):
        self.client = client
        self.vehicle_name = vehicle_name

    def get_timestamp(self) -> Dict[str, Any]:
        """获取当前时间戳"""
        return {"unix_ms": int(time.time() * 1000)}

    # -----------------------------
    # 多旋翼无人机状态：位置/速度/加速度等
    # -----------------------------
    def get_multirotor_state(self) -> Dict[str, Any]:
        """
        获取多旋翼无人机状态
        
        Returns:
            Dict[str, Any]: 包含以下字段的字典
                - timestamp_ns (int): 时间戳(单位: 纳秒)
                - landed_state (int): 着陆状态
                    - 0: 未知状态
                    - 1: 在地面上
                    - 2: 在空中
                - ready (bool): 无人机是否就绪
                - kinematics (Dict[str, Any]): 运动学信息, 详细格式请参考 _kinematics() 方法的返回值说明
        """
        s = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = s.kinematics_estimated
        return {
            "timestamp_ns": int(getattr(s, "timestamp", 0)),
            "landed_state": int(getattr(s, "landed_state", -1)),
            "ready": bool(getattr(s, "ready", True)) if hasattr(s, "ready") else True,
            "kinematics": self._kinematics(kin),
        }
    
    def get_vehicle_pose(self) -> Dict[str, Any]:
        """
        获取多旋翼无人机位姿（位置和姿态）
        
        Returns:
            Dict[str, Any]: 包含以下字段的字典
                - position_ned (Dict[str, float]): 位置 (NED坐标系, 单位: 米)
                    - x (float): 北向位置
                    - y (float): 东向位置
                    - z (float): 地向位置（负值表示高度）
                - orientation_quat (Dict[str, float]): 姿态四元数
                    - w (float): 四元数 w 分量
                    - x (float): 四元数 x 分量
                    - y (float): 四元数 y 分量
                    - z (float): 四元数 z 分量
        """
        p = self.client.simGetVehiclePose(vehicle_name=self.vehicle_name)
        return {
            "position_ned": self._vec3(p.position),
            "orientation_quat": self._quat(p.orientation),
        }
    
    def get_collision_info(self) -> Dict[str, Any]:
        """
        获取碰撞信息
        
        Returns:
            Dict[str, Any]: 包含以下字段的字典
                - has_collided (bool): 是否发生碰撞
                - object_name (str): 碰撞对象的名称 (如果发生碰撞)
                - impact_point_ned (Dict[str, float]): 碰撞点位置 (NED坐标系, 单位: 米)
                    - x (float): 北向位置
                    - y (float): 东向位置
                    - z (float): 地向位置
                - normal_ned (Dict[str, float]): 碰撞法向量 (NED坐标系, 单位向量)
                    - x (float): 北向分量
                    - y (float): 东向分量
                    - z (float): 地向分量
                - timestamp_ns (int): 碰撞时间戳 (单位: 纳秒)
        """
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
    def get_imu(self) -> Dict[str, Any]:
        d = self.client.getImuData(vehicle_name=self.vehicle_name)
        return {
            "timestamp_ns": int(getattr(d, "time_stamp", 0)),
            "angular_velocity": self._vec3(d.angular_velocity),
            "linear_acceleration": self._vec3(d.linear_acceleration),
            "orientation_quat": self._quat(d.orientation),
        }
    
    def get_gps(self) -> Dict[str, Any]:
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
    
    def get_barometer(self) -> Dict[str, Any]:
        d = self.client.getBarometerData(vehicle_name=self.vehicle_name)
        return {
            "timestamp_ns": int(getattr(d, "time_stamp", 0)),
            "altitude": float(getattr(d, "altitude", np.nan)),
            "pressure": float(getattr(d, "pressure", np.nan)),
            "qnh": float(getattr(d, "qnh", np.nan)),
        }

    def get_magnetometer(self) -> Dict[str, Any]:
        d = self.client.getMagnetometerData(vehicle_name=self.vehicle_name)
        mf = getattr(d, "magnetic_field_body", None) or getattr(d, "magnetic_field", None)
        return {
            "timestamp_ns": int(getattr(d, "time_stamp", 0)),
            "magnetic_field": self._vec3(mf),
        }
    
    # -----------------------------
    # 相机图像
    # -----------------------------
    def get_rgb_frame(self, camera: DroneCamera = DroneCamera.FRONT_CENTER, as_bgr_for_cv: bool = True) -> ImageFrame:
        """
        获取单个相机的 RGB 图像
        
        Args:
            camera: 相机枚举
            as_bgr_for_cv: 是否转换为 BGR 格式 (用于 OpenCV)
        
        Returns:
            ImageFrame: 图像帧对象
        """
        req = [airsim.ImageRequest(camera.airsim_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False)]
        res = self.client.simGetImages(req, vehicle_name=self.vehicle_name)[0]
        img = np.frombuffer(res.image_data_uint8, dtype=np.uint8).reshape((res.height, res.width, 3))
        if as_bgr_for_cv:
            img = img[:, :, ::-1]  # RGB -> BGR
        return ImageFrame(width=res.width, height=res.height, frame_type="scene", data=img)
    
    def get_rgb_frames(self, cameras: List[DroneCamera], as_bgr_for_cv: bool = True) -> List[ImageFrame]:
        """
        批量获取多个相机的 RGB 图像
        
        Args:
            cameras: 相机列表（枚举）
            as_bgr_for_cv: 是否转换为 BGR 格式 (用于 OpenCV)
        
        Returns:
            List[ImageFrame]: 图像帧列表，顺序与输入相机列表一致
        """
        if not cameras:
            return []
        
        # 构建批量请求
        requests = []
        for camera in cameras:
            requests.append(airsim.ImageRequest(
                camera.airsim_name,
                airsim.ImageType.Scene,
                pixels_as_float=False, 
                compress=False
            ))
        
        responses = self.client.simGetImages(requests, vehicle_name=self.vehicle_name)
        
        frames = []
        for res in responses:
            img = np.frombuffer(res.image_data_uint8, dtype=np.uint8).reshape((res.height, res.width, 3))
            if as_bgr_for_cv:
                img = img[:, :, ::-1]  # RGB -> BGR
            frames.append(ImageFrame(width=res.width, height=res.height, frame_type="scene", data=img))
        
        return frames

    def get_depth_frame(self, camera: DroneCamera = DroneCamera.FRONT_CENTER) -> ImageFrame:
        """
        获取单个相机的深度图像
        
        Args:
            camera: 相机枚举
        
        Returns:
            ImageFrame: 深度图像帧对象
        """
        req = [airsim.ImageRequest(camera.airsim_name, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)]
        res = self.client.simGetImages(req, vehicle_name=self.vehicle_name)[0]
        depth = np.array(res.image_data_float, dtype=np.float32).reshape((res.height, res.width))
        return ImageFrame(width=res.width, height=res.height, frame_type="depth", data=depth)
    
    def get_depth_frames(self, cameras: List[DroneCamera]) -> List[ImageFrame]:
        """
        批量获取多个相机的深度图像
        
        Args:
            cameras: 相机列表（枚举）
        
        Returns:
            List[ImageFrame]: 深度图像帧列表，顺序与输入相机列表一致
        """
        if not cameras:
            return []
        
        requests = []
        for camera in cameras:
            requests.append(airsim.ImageRequest(
                camera.airsim_name, 
                airsim.ImageType.DepthPerspective, 
                pixels_as_float=True, 
                compress=False
            ))
        
        responses = self.client.simGetImages(requests, vehicle_name=self.vehicle_name)
        
        frames = []
        for res in responses:
            depth = np.array(res.image_data_float, dtype=np.float32).reshape((res.height, res.width))
            frames.append(ImageFrame(width=res.width, height=res.height, frame_type="depth", data=depth))
        
        return frames
    
    # -----------------------------
    # 辅助：格式转换
    # -----------------------------
    @staticmethod
    def _vec3(v) -> Dict[str, float]:
        """
        将AirSim向量3D转换为字典格式
        """
        return {"x": float(v.x_val), "y": float(v.y_val), "z": float(v.z_val)}

    @staticmethod
    def _quat(q) -> Dict[str, float]:
        """
        将AirSim四元数转换为字典格式
        """
        return {"w": float(q.w_val), "x": float(q.x_val), "y": float(q.y_val), "z": float(q.z_val)}

    def _kinematics(self, kin) -> Dict[str, Any]:
        """
        将AirSim运动学数据转换为字典格式
        
        Args:
            kin: AirSim 运动学对象
            
        Returns:
            Dict[str, Any]: 包含以下字段的字典:
                - position_ned (Dict[str, float]): 位置(NED坐标系, 单位: meters)
                    - x (float): 北向位置
                    - y (float): 东向位置
                    - z (float): 地向位置(负值表示高度)
                - orientation_quat (Dict[str, float]): 姿态四元数
                    - w (float): 四元数 w 分量
                    - x (float): 四元数 x 分量
                    - y (float): 四元数 y 分量
                    - z (float): 四元数 z 分量
                - linear_velocity_ned (Dict[str, float]): 线速度(NED坐标系, 单位: m/s)
                    - x (float): 北向速度
                    - y (float): 东向速度
                    - z (float): 地向速度
                - angular_velocity_body (Dict[str, float]): 角速度(机体坐标系, 单位: rad/s)
                    - x (float): 滚转角速度
                    - y (float): 俯仰角速度
                    - z (float): 偏航角速度
                - linear_acceleration_body (Dict[str, float]): 线加速度(机体坐标系, 单位: m/s^2)
                    - x (float): 前向加速度
                    - y (float): 右向加速度
                    - z (float): 下向加速度
                - angular_acceleration_body (Dict[str, float]): 角加速度(机体坐标系, 单位: rad/s^2)
                    - x (float): 滚转角加速度
                    - y (float): 俯仰角加速度
                    - z (float): 偏航角加速度
        """
        return {
            "position_ned": self._vec3(kin.position),
            "orientation_quat": self._quat(kin.orientation),
            "linear_velocity_ned": self._vec3(kin.linear_velocity),
            "angular_velocity_body": self._vec3(kin.angular_velocity),
            "linear_acceleration_body": self._vec3(kin.linear_acceleration),
            "angular_acceleration_body": self._vec3(kin.angular_acceleration),
        }

    

