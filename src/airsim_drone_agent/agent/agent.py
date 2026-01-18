from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from airsim_drone_agent.mllm.client import MLLMClient
from airsim_drone_agent.mllm.messages import Message, Role, TextPart, ImagePart
from airsim_drone_agent.sim.airsim_client import AirSimClient
from airsim_drone_agent.agent.tools import ToolRegistry
from airsim_drone_agent.enums import DroneCamera
from airsim_drone_agent.mllm.image_codec import bgr_to_jpeg_data_url

from airsim_drone_agent.utils.logger import get_logger, log_data

class DroneAgent:
    """
    无人机智能体：连接 AirSimClient 和多模态大语言模型
    """

    def __init__(
        self,
        client: AirSimClient,
        mllm: MLLMClient,
        ui_callback: Optional[Any] = None
    ):
        """
        初始化智能体(纯视觉模式)
        
        Args:
            client: AirSim 客户端
            mllm: 多模态大语言模型客户端
            ui_callback: UI 回调对象（用于更新界面）
        """
        self.client = client
        self.mllm = mllm
        self.cameras = [
            DroneCamera.FRONT_CENTER,
            DroneCamera.FRONT_LEFT,
            DroneCamera.FRONT_RIGHT,
            DroneCamera.BACK_CENTER,
            DroneCamera.BOTTOM_CENTER,
        ]
        self.tools = ToolRegistry(client)
        self.logger = get_logger(__name__)
        self.ui_callback = ui_callback  # GradioUI 对象

    def _get_system_prompt(self) -> str:
        """生成系统提示词（纯视觉模式）"""
        # 使用简化的工具列表，减少提示词长度
        tools_summary = self.tools.list_tools_summary()

        camera_names = [cam.name for cam in self.cameras]
        
        system_prompt = f"""你是一个专业的无人机控制智能体，基于纯视觉信息进行决策和控制。

可用工具列表：
{tools_summary}

相机配置：
无人机搭载了5个摄像头，分别位于：
- {camera_names[0]} (FRONT_CENTER): 前方中央 - 默认提供此视角
- {camera_names[1]} (FRONT_LEFT): 前方左侧 - 可通过 get_camera_image 工具获取
- {camera_names[2]} (FRONT_RIGHT): 前方右侧 - 可通过 get_camera_image 工具获取
- {camera_names[3]} (BACK_CENTER): 后方中央 - 可通过 get_camera_image 工具获取
- {camera_names[4]} (BOTTOM_CENTER): 底部中央 - 可通过 get_camera_image 工具获取

注意：默认情况下你只能看到前方中央摄像头的图像。如果需要查看其他视角（如检查后方障碍物、查看底部情况等），请使用 get_camera_image 工具获取相应摄像头的图像。

工具调用格式：
你必须先输出简要规划，然后以 JSON 格式返回工具调用列表：
规划：
<用简要条目说明接下来要做什么，避免详细推理>
工具调用列表：
[
  {{
    "tool_name": "工具名称1",
    "arguments": {{"参数名": "参数值"}}
  }},
  {{
    "tool_name": "工具名称2",
    "arguments": {{"参数名": "参数值"}}
  }}
]

如果不需要调用工具，可以返回 {{"tool_name": null, "reason": "原因"}}。

重要：参数要求
- 每个工具定义中都包含 "required" 字段，列出了该工具的必填参数
- 如果工具定义中包含 "required" 字段，你必须为所有列出的必填参数提供合适的参数值
- 必填参数不能省略，不能为 null，必须根据工具描述和当前任务提供合理的值
- 可选参数（不在 required 列表中的参数）可以省略，将使用默认值
- 示例：如果工具 "move_to_position" 的 required 包含 ["x", "y", "z"]，则调用时必须提供这三个参数的值

重要提示：
1. 默认情况下你只能看到前方中央摄像头的图像，没有其他传感器数据（IMU、GPS、气压计等）
2. 如果需要查看其他视角（如检查后方障碍物、查看底部情况等），使用 get_camera_image 工具获取相应摄像头的图像
3. 基于视觉信息判断环境、障碍物、目标位置等
4. 使用 NED 坐标系（X=北，Y=东，Z=向下，负Z表示向上）
5. 所有距离单位是米，速度单位是米/秒
6. 角度单位：偏航角使用度，角速度使用弧度/秒
7. 通过观察图像判断无人机状态（是否在地面、高度、周围环境等）
8. 确保安全：起飞前观察地面情况，降落前确认降落区域安全
9. 根据图像中的视觉线索（如建筑物、地标等）进行导航和定位
10. 执行任何飞行动作前先解锁（arm），任务结束后再上锁（disarm）
"""
        
        return system_prompt

    def _get_status_message(self) -> Message:
        """
        获取当前无人机状态信息并转换为消息
        
        Returns:
            Message: 包含当前状态信息的消息
        """
        try:
            state = self.client.sensors.get_multirotor_state()
            
            # 提取关键信息
            kinematics = state.get("kinematics", {})
            position = kinematics.get("position_ned", {})            # 无人机位置
            velocity = kinematics.get("linear_velocity_ned", {})     # 无人机速度
            orientation = kinematics.get("orientation_quat", {})     # 无人机姿态
            
            # 更新 UI 状态
            if self.ui_callback:
                status_dict = {
                    "position_ned": position,
                    "velocity_ned": velocity,
                    "orientation_quat": orientation
                }
                self.ui_callback.update_status(status_dict)
            
            # 格式化状态信息
            status_text = f"""当前无人机状态：

位置 (NED坐标系, 单位: 米):
  - X (北): {position.get('x', 0):.2f}
  - Y (东): {position.get('y', 0):.2f}
  - Z (地): {position.get('z', 0):.2f} (高度: {-position.get('z', 0):.2f} 米)

速度 (NED坐标系, 单位: m/s):
  - X (北): {velocity.get('x', 0):.2f}
  - Y (东): {velocity.get('y', 0):.2f}
  - Z (地): {velocity.get('z', 0):.2f}

姿态 (四元数):
  - w: {orientation.get('w', 0):.4f}
  - x: {orientation.get('x', 0):.4f}
  - y: {orientation.get('y', 0):.4f}
  - z: {orientation.get('z', 0):.4f}
"""
            
            return Message.user([TextPart(text=status_text)])
        except Exception as e:
            self.logger.warning("获取状态信息失败: %s", str(e))
            return Message.user([TextPart(text="警告：无法获取当前无人机状态信息")])
    
    def _get_observation_messages(self, cameras: Optional[List[DroneCamera]] = None) -> List[Message]:
        """
        获取指定摄像头的图像并转换为消息
        
        Args:
            cameras: 要获取的摄像头列表，如果为 None 则默认只获取前方中央摄像头
        """
        if cameras is None:
            cameras = [DroneCamera.FRONT_CENTER]
        
        try:
            # 使用批量获取，一次 API 调用获取所有图像
            frames = self.client.sensors.get_rgb_frames(cameras, as_bgr_for_cv=True)
            
            if not frames or len(frames) != len(cameras):
                return [
                    Message.user([TextPart(text="警告：无法获取相机图像，请检查连接")])
                ]
            
            # 转换为图像消息
            image_parts = []
            for camera, frame in zip(cameras, frames):
                img_url = bgr_to_jpeg_data_url(frame.data)
                image_parts.append(ImagePart(url=img_url, name=camera.name))
            
            # 构建消息
            if len(cameras) == 1:
                camera_name = cameras[0].name
                text = f"当前 {camera_name} 摄像头的实时图像：\n\n请基于这个图像进行决策。"
            else:
                camera_names = ", ".join([cam.name for cam in cameras])
                text = f"当前 {len(cameras)} 个摄像头的实时图像 ({camera_names})：\n\n请基于这些图像进行决策。"
            
            messages = [
                Message.user([
                    TextPart(text=text),
                    *image_parts
                ])
            ]
            
            return messages
        except Exception as e:
            # 如果批量获取失败，返回错误消息
            return [
                Message.user([TextPart(text=f"警告：无法获取相机图像，错误：{str(e)}")])
            ]

    def _extract_plan_and_tool_calls(self, response_text: str) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """提取规划文本和工具调用 JSON"""
        import re
        plan_text = None
        tool_calls: List[Dict[str, Any]] = []
        
        # 尝试直接解析 JSON
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict) and "tool_name" in parsed:
                tool_calls = [parsed]
                plan_text = None
                return plan_text, tool_calls
            if isinstance(parsed, list) and all(isinstance(p, dict) and "tool_name" in p for p in parsed):
                tool_calls = parsed
                plan_text = None
                return plan_text, tool_calls
        except json.JSONDecodeError:
            pass
        
        # 从文本中提取 JSON（优先匹配列表）
        list_match = re.search(r'\[\s*\{[^[]*"tool_name"[\s\S]*?\}\s*\]', response_text, re.DOTALL)
        dict_match = re.search(r'\{[^{}]*"tool_name"[^{}]*\}', response_text, re.DOTALL)
        
        if list_match:
            try:
                parsed = json.loads(list_match.group())
                if isinstance(parsed, list) and all(isinstance(p, dict) and "tool_name" in p for p in parsed):
                    tool_calls = parsed
            except json.JSONDecodeError:
                tool_calls = []
        elif dict_match:
            try:
                parsed = json.loads(dict_match.group())
                if isinstance(parsed, dict) and "tool_name" in parsed:
                    tool_calls = [parsed]
            except json.JSONDecodeError:
                tool_calls = []
        
        # 规划文本取 JSON 之前的内容
        if list_match:
            plan_text = response_text[:list_match.start()].strip()
        elif dict_match:
            plan_text = response_text[:dict_match.start()].strip()
        else:
            plan_text = response_text.strip()
        
        return plan_text if plan_text else None, tool_calls

    def step(self, user_input: str, include_observation: bool = True, additional_images: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        执行一步推理和行动
        
        Args:
            user_input: 用户指令
            include_observation: 是否包含当前观测数据（默认只包含前方中央摄像头）
            additional_images: 额外的图像信息列表（用于显示 get_camera_image 工具获取的图像）
        
        Returns:
            包含工具调用和执行结果的字典
        """
        messages: List[Message] = []
        
        # 系统提示词
        messages.append(Message.system(self._get_system_prompt()))

        # 当前状态信息（位置、速度、姿态等）
        messages.append(self._get_status_message())

        # 用户指令
        messages.append(Message.user([TextPart(text=user_input)]))
        
        # 当前观测数据（图像）
        if include_observation:
            messages.extend(self._get_observation_messages())
        
        # 添加额外的图像（来自 get_camera_image 工具）
        if additional_images:
            for img_info in additional_images:
                if "image_url" in img_info:
                    camera_name = img_info.get("camera", "未知")
                    messages.append(
                        Message.user([
                            TextPart(text=f"{camera_name} 摄像头的图像："),
                            ImagePart(url=img_info["image_url"], name=camera_name)
                        ])
                    )
        
        # 调用 LLM
        result = self.mllm.generate(
            messages=messages
        )

        log_data(self.logger, result.text, title="多模态大语言模型回复")
        
        # 解析规划与工具调用
        self.logger.info("解析规划与工具调用...")
        plan_text, tool_calls = self._extract_plan_and_tool_calls(result.text)
        
        # 更新 UI：记录 LLM 输出
        if self.ui_callback:
            self.ui_callback.add_llm_output({
                "understanding": plan_text,
                "tool_name": tool_calls[0].get("tool_name") if tool_calls else None,
                "arguments": tool_calls[0].get("arguments", {}) if tool_calls else {},
                "tool_calls": tool_calls or [],
                "raw_response": result.text
            })
        
        if not tool_calls:
            return {
                "tool_name": None,
                "tool_calls": [],
                "reason": "无法解析工具调用",
                "llm_response": result.text,
                "plan": plan_text
            }
        
        # 如果返回的是明确不调用工具
        if len(tool_calls) == 1 and tool_calls[0].get("tool_name") is None:
            return {
                "tool_name": None,
                "tool_calls": tool_calls,
                "reason": tool_calls[0].get("reason", "未调用工具"),
                "llm_response": result.text,
                "plan": plan_text
            }
        
        # 验证所有工具必填参数
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            arguments = tool_call.get("arguments", {})
            if tool_name is None:
                return {
                    "tool_name": None,
                    "tool_calls": tool_calls,
                    "reason": "工具列表中存在空的 tool_name",
                    "llm_response": result.text,
                    "plan": plan_text
                }
            is_valid, error_msg = self.tools.validate_arguments(tool_name, arguments)
            if not is_valid:
                self.logger.warning("工具调用参数验证失败: %s - %s", tool_name, error_msg)
                return {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": None,
                    "success": False,
                    "error": error_msg,
                    "error_type": "missing_required_parameters",
                    "plan": plan_text
                }
        
        # 执行工具调用（按顺序）
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            arguments = tool_call.get("arguments", {})
            try:
                tool_result = self.tools.call_tool(tool_name, arguments)
                result_dict = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": tool_result,
                    "success": True,
                    "plan": plan_text
                }
                results.append(result_dict)
                # 更新 UI：工具调用成功
                if self.ui_callback:
                    self.ui_callback.add_tool_call(result_dict)
            except Exception as e:
                self.logger.error("工具调用执行失败: %s - %s", tool_name, str(e))
                result_dict = {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": None,
                    "success": False,
                    "error": str(e),
                    "error_type": "execution_error",
                    "plan": plan_text
                }
                if self.ui_callback:
                    self.ui_callback.add_tool_call(result_dict)
                return {
                    "tool_name": tool_name,
                    "tool_calls": tool_calls,
                    "results": results,
                    "result": None,
                    "success": False,
                    "error": str(e),
                    "error_type": "execution_error",
                    "plan": plan_text
                }
        
        # 返回批量执行结果
        if len(results) == 1:
            return results[0]
        
        return {
            "tool_name": "__batch__",
            "tool_calls": tool_calls,
            "results": results,
            "success": True,
            "plan": plan_text
        }

    def chat(self, user_input: str, max_steps: int = 10) -> List[Dict[str, Any]]:
        """
        多轮对话，直到任务完成或达到最大步数
        
        Args:
            user_input: 初始用户指令
            max_steps: 最大执行步数
        
        Returns:
            执行历史列表
        """
        history = []
        current_input = user_input
        
        for step_num in range(max_steps):
            # 检查上一步是否获取了图像
            additional_images = None
            if step_num > 0 and history:
                prev_result = history[-1]
                if prev_result.get("tool_name") == "get_camera_image":
                    if (prev_result.get("success") and 
                        isinstance(prev_result.get("result"), dict) and 
                        "image_url" in prev_result.get("result", {})):
                        additional_images = [prev_result["result"]]
                elif prev_result.get("tool_name") == "__batch__":
                    batch_images = []
                    for item in prev_result.get("results", []):
                        if (item.get("tool_name") == "get_camera_image" and 
                            item.get("success") and 
                            isinstance(item.get("result"), dict) and 
                            "image_url" in item.get("result", {})):
                            batch_images.append(item["result"])
                    if batch_images:
                        additional_images = batch_images
            
            result = self.step(current_input, include_observation=(step_num == 0), additional_images=additional_images)
            history.append(result)
            
            # 如果不需要调用工具，结束
            if result.get("tool_name") is None:
                break
            
            # 如果工具调用失败，结束
            if not result.get("success", False):
                break
            
            # 准备下一步的输入（基于执行结果）
            tool_result = result.get("result")
            
            # 特殊处理：如果调用了 get_camera_image 工具
            if result['tool_name'] == "get_camera_image" and isinstance(tool_result, dict) and "image_url" in tool_result:
                camera_name = tool_result.get("camera", "未知")
                current_input = f"工具 get_camera_image 执行完成，已获取 {camera_name} 摄像头的图像。图像信息：宽度 {tool_result.get('width')}px，高度 {tool_result.get('height')}px。请基于这个图像进行分析或继续执行任务。"
            elif result['tool_name'] == "__batch__":
                batch_results = result.get("results", [])
                batch_result_str = json.dumps(batch_results, indent=2, ensure_ascii=False)
                current_input = f"已顺序执行 {len(batch_results)} 个工具调用，结果如下：{batch_result_str}。请继续执行任务或说明任务已完成。"
            else:
                # 其他工具的正常处理
                if isinstance(tool_result, dict):
                    tool_result_str = json.dumps(tool_result, indent=2, ensure_ascii=False)
                else:
                    tool_result_str = str(tool_result)
                
                current_input = f"工具 {result['tool_name']} 执行完成，结果：{tool_result_str}。请继续执行任务或说明任务已完成。"
        
        # 更新 UI：添加最终回复
        if self.ui_callback and history:
            final_response = f"任务执行完成，共执行 {len(history)} 步。"
            if not history[-1].get("success"):
                final_response += f"最后一步执行失败: {history[-1].get('error', '未知错误')}"
            self.ui_callback.add_assistant_response(final_response)
        
        return history
