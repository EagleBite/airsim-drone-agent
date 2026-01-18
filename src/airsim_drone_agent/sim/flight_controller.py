from __future__ import annotations

from typing import Optional

import airsim


class FlightController:
    """
    飞行控制器：专门负责无人机的飞行控制
    - 位置控制
    - 速度控制
    - 姿态控制
    - 路径规划
    """

    def __init__(self, client: airsim.MultirotorClient, vehicle_name: str = ""):
        """
        初始化飞行控制器
        
        Args:
            client: AirSim 多旋翼客户端
            vehicle_name: 载具名称
        """
        self.client = client
        self.vehicle_name = vehicle_name

    # -----------------------------
    # 基础飞行操作
    # -----------------------------
    def arm(self, armed: bool = True) -> None:
        """解锁/上锁 (arm/disarm)"""
        self.client.armDisarm(armed, vehicle_name=self.vehicle_name)

    def takeoff(self, timeout_sec: float = 10.0) -> None:
        """起飞 (异步 join)"""
        self.client.takeoffAsync(timeout_sec=timeout_sec, vehicle_name=self.vehicle_name).join()

    def land(self, timeout_sec: float = 20.0) -> None:
        """降落 (异步 join)"""
        self.client.landAsync(timeout_sec=timeout_sec, vehicle_name=self.vehicle_name).join()

    def hover(self) -> None:
        """悬停 (保持当前位置和姿态)"""
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()

    # -----------------------------
    # 飞行控制：位置控制
    # -----------------------------
    def move_to_position(
        self,
        x: float,
        y: float,
        z: float,
        velocity: float = 5.0,
        timeout_sec: float = 60.0,
        drivetrain: airsim.DrivetrainType = airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode: airsim.YawMode = airsim.YawMode(),
    ) -> None:
        """
        移动到指定位置 (NED 坐标系)
        
        Args:
            x, y, z: 目标位置 (米, NED坐标系)
            注意: NED 坐标中 Z 轴向下，向上飞行请使用负数
            velocity: 移动速度 (米/秒)
            timeout_sec: 超时时间
            drivetrain: 驱动模式，影响运动约束
                - airsim.DrivetrainType.ForwardOnly: 只能朝前飞，侧向移动会受限
                - airsim.DrivetrainType.MaxDegreeOfFreedom: 全向运动，允许侧向/后退
            yaw_mode: 偏航模式 (控制机头朝向)
                - airsim.YawMode(): 保持当前朝向
                - airsim.YawMode(is_rate=False, yaw_or_rate=角度): 指定偏航角（度）
                - airsim.YawMode(is_rate=True, yaw_or_rate=角速度): 指定偏航角速度（度/秒）
        """
        self.client.moveToPositionAsync(
            x, y, z, velocity,
            timeout_sec=timeout_sec,
            drivetrain=drivetrain,
            yaw_mode=yaw_mode,
            vehicle_name=self.vehicle_name
        ).join()

    def move_to_z(
        self,
        z: float,
        velocity: float = 2.0,
        timeout_sec: float = 30.0,
        yaw_mode: airsim.YawMode = airsim.YawMode(),
    ) -> None:
        """
        移动到指定高度 (仅改变 Z 坐标)
        
        Args:
            z: 目标高度 (米, NED坐标系, 负值表示向上)
            velocity: 移动速度 (米/秒)
            timeout_sec: 超时时间
            yaw_mode: 偏航模式 (控制机头朝向)
                - airsim.YawMode(): 保持当前朝向
                - airsim.YawMode(is_rate=False, yaw_or_rate=角度): 指定偏航角（度）
                - airsim.YawMode(is_rate=True, yaw_or_rate=角速度): 指定偏航角速度（度/秒）
        """
        self.client.moveToZAsync(
            z, velocity,
            timeout_sec=timeout_sec,
            yaw_mode=yaw_mode,
            vehicle_name=self.vehicle_name
        ).join()

    # -----------------------------
    # 飞行控制：速度控制
    # -----------------------------
    def move_by_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        duration: float,
        drivetrain: airsim.DrivetrainType = airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode: airsim.YawMode = airsim.YawMode(),
    ) -> None:
        """
        按指定速度移动 (持续一段时间)
        
        Args:
            vx, vy, vz: 速度向量 (米/秒, NED坐标系)
            duration: 持续时间 (秒)
            drivetrain: 驱动模式，影响运动约束
                - airsim.DrivetrainType.ForwardOnly: 只能朝前飞，侧向移动会受限
                - airsim.DrivetrainType.MaxDegreeOfFreedom: 全向运动，允许侧向/后退
            yaw_mode: 偏航模式 (控制机头朝向)
                - airsim.YawMode(): 保持当前朝向
                - airsim.YawMode(is_rate=False, yaw_or_rate=角度): 指定偏航角（度）
                - airsim.YawMode(is_rate=True, yaw_or_rate=角速度): 指定偏航角速度（度/秒）
        """
        self.client.moveByVelocityAsync(
            vx, vy, vz, duration,
            drivetrain=drivetrain,
            yaw_mode=yaw_mode,
            vehicle_name=self.vehicle_name
        ).join()

    def move_by_velocity_z(
        self,
        vx: float,
        vy: float,
        z: float,
        duration: float,
        drivetrain: airsim.DrivetrainType = airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode: airsim.YawMode = airsim.YawMode(),
    ) -> None:
        """
        按指定速度移动，同时保持指定高度
        
        Args:
            vx, vy: 水平速度 (米/秒)
            z: 目标高度 (米, NED坐标系)
            duration: 持续时间 (秒)
            drivetrain: 驱动模式，影响运动约束
                - airsim.DrivetrainType.ForwardOnly: 只能朝前飞，侧向移动会受限
                - airsim.DrivetrainType.MaxDegreeOfFreedom: 全向运动，允许侧向/后退
            yaw_mode: 偏航模式 (控制机头朝向)
                - airsim.YawMode(): 保持当前朝向
                - airsim.YawMode(is_rate=False, yaw_or_rate=角度): 指定偏航角（度）
                - airsim.YawMode(is_rate=True, yaw_or_rate=角速度): 指定偏航角速度（度/秒）
        """
        self.client.moveByVelocityZAsync(
            vx, vy, z, duration,
            drivetrain=drivetrain,
            yaw_mode=yaw_mode,
            vehicle_name=self.vehicle_name
        ).join()

    # -----------------------------
    # 飞行控制：姿态控制
    # -----------------------------
    def move_by_angle_rates_z(
        self,
        pitch_rate: float,
        roll_rate: float,
        yaw_rate: float,
        z: float,
        duration: float,
    ) -> None:
        """
        按角速度移动，同时保持指定高度
        
        Args:
            pitch_rate: 俯仰角速度 (弧度/秒)
            roll_rate: 横滚角速度 (弧度/秒)
            yaw_rate: 偏航角速度 (弧度/秒)
            z: 目标高度 (米, NED坐标系)
            duration: 持续时间 (秒)
        """
        self.client.moveByAngleRatesZAsync(
            pitch_rate, roll_rate, yaw_rate, z, duration,
            vehicle_name=self.vehicle_name
        ).join()

    def rotate_by_yaw_rate(
        self,
        yaw_rate: float,
        duration: float,
    ) -> None:
        """
        按偏航角速度旋转
        
        Args:
            yaw_rate: 偏航角速度 (弧度/秒)
            duration: 持续时间 (秒)
        """
        self.client.rotateByYawRateAsync(
            yaw_rate, duration,
            vehicle_name=self.vehicle_name
        ).join()

    def rotate_to_yaw(
        self,
        yaw: float,
        margin: float = 5.0,
        timeout_sec: float = 30.0,
    ) -> None:
        """
        旋转到指定偏航角
        
        Args:
            yaw: 目标偏航角 (度)
            margin: 允许误差 (度)
            timeout_sec: 超时时间
        """
        self.client.rotateToYawAsync(
            yaw, margin=margin, timeout_sec=timeout_sec,
            vehicle_name=self.vehicle_name
        ).join()

    # -----------------------------
    # 飞行控制：路径规划
    # -----------------------------
    def move_on_path(
        self,
        path: list[tuple[float, float, float]],
        velocity: float = 5.0,
        timeout_sec: float = 120.0,
        drivetrain: airsim.DrivetrainType = airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode: airsim.YawMode = airsim.YawMode(),
        lookahead: float = -1.0,
        adaptive_lookahead: float = 1.0,
    ) -> None:
        """
        沿路径移动
        
        Args:
            path: 路径点列表，每个点为 (x, y, z) 元组 (NED坐标系)
            velocity: 移动速度 (米/秒)
            timeout_sec: 超时时间
            drivetrain: 驱动模式，影响运动约束
                - airsim.DrivetrainType.ForwardOnly: 只能朝前飞，侧向移动会受限
                - airsim.DrivetrainType.MaxDegreeOfFreedom: 全向运动，允许侧向/后退
            yaw_mode: 偏航模式 (控制机头朝向)
                - airsim.YawMode(): 保持当前朝向
                - airsim.YawMode(is_rate=False, yaw_or_rate=角度): 指定偏航角（度）
                - airsim.YawMode(is_rate=True, yaw_or_rate=角速度): 指定偏航角速度（度/秒）
            lookahead: 前瞻距离 (米, -1表示自动)
            adaptive_lookahead: 自适应前瞻系数
        """
        path_points = [airsim.Vector3r(x, y, z) for x, y, z in path]
        self.client.moveOnPathAsync(
            path_points, velocity,
            timeout_sec=timeout_sec,
            drivetrain=drivetrain,
            yaw_mode=yaw_mode,
            lookahead=lookahead,
            adaptive_lookahead=adaptive_lookahead,
            vehicle_name=self.vehicle_name
        ).join()
