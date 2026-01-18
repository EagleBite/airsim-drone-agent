from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from airsim_drone_agent.sim.airsim_client import AirSimClient


@dataclass
class Tool:
    """工具定义"""
    name: str                   # 工具名称
    description: str            # 工具描述
    parameters: Dict[str, Any]  # 工具参数
    func: Callable              # 工具函数


class ToolRegistry:
    """工具注册表：将 AirSimClient 的方法注册为工具"""

    def __init__(self, client: AirSimClient):
        self.client = client
        self.tools: Dict[str, Tool] = {}
        self._register_all_tools()

    def _register_all_tools(self):
        """注册所有 AirSim 控制工具"""
        
        # 基础操作
        self.register(
            name="arm",
            description="解锁或上锁无人机",
            parameters={
                "type": "object",
                "properties": {
                    "armed": {"type": "boolean", "description": "True=解锁, False=上锁"}
                },
                "required": ["armed"]
            },
            func=lambda armed: self.client.arm(armed=armed)
        )

        self.register(
            name="takeoff",
            description="无人机起飞",
            parameters={
                "type": "object",
                "properties": {
                    "timeout_sec": {"type": "number", "description": "超时时间（秒）", "default": 10.0}
                }
            },
            func=lambda timeout_sec=10.0: self.client.takeoff(timeout_sec=timeout_sec)
        )

        self.register(
            name="land",
            description="无人机降落",
            parameters={
                "type": "object",
                "properties": {
                    "timeout_sec": {"type": "number", "description": "超时时间（秒）", "default": 20.0}
                }
            },
            func=lambda timeout_sec=20.0: self.client.land(timeout_sec=timeout_sec)
        )

        self.register(
            name="hover",
            description="无人机悬停（保持当前位置和姿态）",
            parameters={"type": "object", "properties": {}},
            func=lambda: self.client.hover()
        )

        # 位置控制
        self.register(
            name="move_to_position",
            description="移动到指定位置（NED坐标系，单位：米）",
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X坐标（米）"},
                    "y": {"type": "number", "description": "Y坐标（米）"},
                    "z": {"type": "number", "description": "Z坐标（米，负值表示向上）"},
                    "velocity": {"type": "number", "description": "移动速度（米/秒）", "default": 5.0},
                    "timeout_sec": {"type": "number", "description": "超时时间（秒）", "default": 60.0}
                },
                "required": ["x", "y", "z"]
            },
            func=lambda x, y, z, velocity=5.0, timeout_sec=60.0: self.client.move_to_position(
                x=x, y=y, z=z, velocity=velocity, timeout_sec=timeout_sec
            )
        )

        self.register(
            name="move_to_z",
            description="移动到指定高度（仅改变Z坐标）",
            parameters={
                "type": "object",
                "properties": {
                    "z": {"type": "number", "description": "目标高度（米，负值表示向上）"},
                    "velocity": {"type": "number", "description": "移动速度（米/秒）", "default": 2.0},
                    "timeout_sec": {"type": "number", "description": "超时时间（秒）", "default": 30.0}
                },
                "required": ["z"]
            },
            func=lambda z, velocity=2.0, timeout_sec=30.0: self.client.move_to_z(
                z=z, velocity=velocity, timeout_sec=timeout_sec
            )
        )

        # 路径规划
        self.register(
            name="move_on_path",
            description="沿路径移动",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "array",
                        "description": "路径点列表，每个点为 [x, y, z]",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3
                        }
                    },
                    "velocity": {"type": "number", "description": "移动速度（米/秒）", "default": 5.0},
                    "timeout_sec": {"type": "number", "description": "超时时间（秒）", "default": 120.0}
                },
                "required": ["path"]
            },
            func=lambda path, velocity=5.0, timeout_sec=120.0: self.client.move_on_path(
                path=[tuple(p) for p in path], velocity=velocity, timeout_sec=timeout_sec
            )
        )

        # 图像获取工具
        self.register(
            name="get_camera_image",
            description="获取指定摄像头的图像（用于查看特定视角）",
            parameters={
                "type": "object",
                "properties": {
                    "camera": {
                        "type": "string",
                        "description": "摄像头名称，可选值: FRONT_CENTER(前方中央), FRONT_LEFT(前方左侧), FRONT_RIGHT(前方右侧), BACK_CENTER(后方中央), BOTTOM_CENTER(底部中央)",
                        "enum": ["FRONT_CENTER", "FRONT_LEFT", "FRONT_RIGHT", "BACK_CENTER", "BOTTOM_CENTER"]
                    }
                },
                "required": ["camera"]
            },
            func=lambda camera: self._get_camera_image(camera)
        )
        
        # 注意：纯视觉模式下，不提供传感器数据获取工具
        # 智能体只能通过观察图像来判断状态
    
    def _get_camera_image(self, camera_name: str) -> Dict[str, Any]:
        """
        获取指定摄像头的图像（内部方法，用于工具调用）
        
        Args:
            camera_name: 摄像头名称
        
        Returns:
            包含图像信息的字典
        """
        from airsim_drone_agent.enums import DroneCamera
        from airsim_drone_agent.mllm.image_codec import bgr_to_jpeg_data_url
        
        # 将字符串转换为枚举
        camera_map = {
            "FRONT_CENTER": DroneCamera.FRONT_CENTER,
            "FRONT_LEFT": DroneCamera.FRONT_LEFT,
            "FRONT_RIGHT": DroneCamera.FRONT_RIGHT,
            "BACK_CENTER": DroneCamera.BACK_CENTER,
            "BOTTOM_CENTER": DroneCamera.BOTTOM_CENTER,
        }
        
        if camera_name not in camera_map:
            raise ValueError(f"未知摄像头: {camera_name}")
        
        camera = camera_map[camera_name]
        frame = self.client.sensors.get_rgb_frame(camera, as_bgr_for_cv=True)
        img_url = bgr_to_jpeg_data_url(frame.data)
        
        return {
            "camera": camera_name,
            "width": frame.width,
            "height": frame.height,
            "image_url": img_url,
            "message": f"已获取 {camera_name} 摄像头的图像"
        }

    def register(self, name: str, description: str, parameters: Dict[str, Any], func: Callable):
        """注册工具"""
        self.tools[name] = Tool(name=name, description=description, parameters=parameters, func=func)

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self.tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """列出所有工具(用于 LLM)"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools.values()
        ]
    
    def list_tools_summary(self) -> str:
        """
        列出工具的简要摘要（用于系统提示词，减少长度）
        
        Returns:
            工具列表的简要文本描述
        """
        lines = []
        for tool in self.tools.values():
            required = tool.parameters.get("required", [])
            required_str = f"必填: {', '.join(required)}" if required else "无必填参数"
            lines.append(f"- {tool.name}: {tool.description} ({required_str})")
        return "\n".join(lines)

    def validate_arguments(self, name: str, arguments: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        验证工具参数是否完整
        
        Args:
            name: 工具名称
            arguments: 提供的参数
        
        Returns:
            (是否有效, 错误信息)
        """
        tool = self.get_tool(name)
        if tool is None:
            return False, f"未知工具: {name}"
        
        # 检查必填参数
        required = tool.parameters.get("required", [])
        missing = [param for param in required if param not in arguments or arguments[param] is None]
        
        if missing:
            return False, f"缺少必填参数: {', '.join(missing)}"
        
        return True, None
    
    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用工具"""
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"未知工具: {name}")
        
        # 验证参数
        is_valid, error_msg = self.validate_arguments(name, arguments)
        if not is_valid:
            raise ValueError(error_msg)
        
        try:
            return tool.func(**arguments)
        except Exception as e:
            raise RuntimeError(f"工具调用失败: {str(e)}") from e
