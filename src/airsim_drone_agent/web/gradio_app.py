"""
Gradio Web 界面：快速构建可视化界面
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional
import gradio as gr

from airsim_drone_agent.utils.logger import get_logger

logger = get_logger(__name__)


class GradioUI:
    """Gradio 界面管理器"""
    
    def __init__(self):
        # Gradio Chatbot 需要列表格式，每个元素是 [用户消息, 助手消息] 的元组
        self.chat_history: List[tuple[str, str]] = []  # (用户输入, 助手回复)
        self.llm_outputs: List[Dict[str, Any]] = []  # LLM 的完整输出
        self.tool_calls: List[Dict[str, Any]] = []  # 工具调用历史
        self.current_status: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()
    
    def add_user_message(self, message: str):
        """添加用户消息"""
        with self.lock:
            # 添加新的对话对，助手回复为空
            self.chat_history.append((message, ""))
    
    def add_assistant_response(self, response: str):
        """添加助手回复"""
        with self.lock:
            if self.chat_history:
                # 更新最后一对对话的助手回复
                user_msg = self.chat_history[-1][0]
                self.chat_history[-1] = (user_msg, response)
            else:
                # 如果没有用户消息，创建一个空的用户消息
                self.chat_history.append(("", response))
    
    def add_llm_output(self, output: Dict[str, Any]):
        """添加 LLM 输出（理解和工具调用）"""
        with self.lock:
            self.llm_outputs.append(output)
            # 只保留最近 100 条
            if len(self.llm_outputs) > 100:
                self.llm_outputs.pop(0)
    
    def add_tool_call(self, tool_call: Dict[str, Any]):
        """添加工具调用"""
        with self.lock:
            self.tool_calls.append(tool_call)
            # 只保留最近 50 条
            if len(self.tool_calls) > 50:
                self.tool_calls.pop(0)
    
    def update_status(self, status: Dict[str, Any]):
        """更新当前状态"""
        with self.lock:
            self.current_status = status
    
    def get_latest_llm_output(self) -> Dict[str, Any]:
        """获取最新的 LLM 输出"""
        with self.lock:
            return self.llm_outputs[-1] if self.llm_outputs else {}
    
    def get_tool_history(self) -> List[Dict[str, Any]]:
        """获取工具调用历史（最近 20 条）"""
        with self.lock:
            return self.tool_calls[-20:]
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        with self.lock:
            return self.current_status or {}


def create_gradio_app(agent_callback=None):
    """
    创建 Gradio 应用
    
    Args:
        agent_callback: 用于处理用户输入的代理函数 (user_input) -> None
    """
    ui = GradioUI()
    
    def chat_fn(message, history):
        """处理聊天输入"""
        if not message.strip():
            return history, ""
        
        # 添加用户消息
        ui.add_user_message(message)
        
        # 调用代理处理
        if agent_callback:
            try:
                agent_callback(message, ui)
            except Exception as e:
                error_msg = f"执行错误: {str(e)}"
                logger.error(error_msg, exc_info=True)
                ui.add_assistant_response(error_msg)
        
        # 返回更新后的历史
        # 如果 Chatbot 使用 type="messages"，需要字典格式
        with ui.lock:
            # 转换为字典格式：{"role": "user/assistant", "content": "..."}
            formatted_history = []
            for user_msg, assistant_msg in ui.chat_history:
                if user_msg:
                    formatted_history.append({"role": "user", "content": str(user_msg)})
                if assistant_msg:
                    formatted_history.append({"role": "assistant", "content": str(assistant_msg)})
            return formatted_history, ""
    
    def update_displays():
        """更新所有显示内容"""
        latest = ui.get_latest_llm_output()
        plan_text = latest.get("understanding", "")
        tool_calls = latest.get("tool_calls", [])
        raw_response = latest.get("raw_response", "")
        return (
            plan_text,
            tool_calls,
            ui.get_tool_history(),
            ui.get_status(),
            raw_response
        )
    
    # 创建整个页面容器
    with gr.Blocks(title="AirSim 无人机智能体") as app:
        gr.Markdown("# 🚁 AirSim 无人机智能体 - 可视化界面")
        
        with gr.Row():
            # 左侧：对话区域
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=500,
                    show_label=True,
                    container=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="输入指令",
                        placeholder="输入指令，例如：向前飞10米",
                        scale=4,
                        lines=2
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)
                
                clear_btn = gr.Button("清空对话", variant="secondary")
            
            # 右侧：状态面板
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("本次规划"):
                        plan_display = gr.Textbox(
                            value="",
                            label="规划 / 理解",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Tab("本次工具调用"):
                        tool_calls_display = gr.JSON(
                            value=[],
                            label="工具调用列表"
                        )
                    
                    with gr.Tab("工具历史"):
                        tool_history_display = gr.JSON(
                            value=[],
                            label="工具调用历史（最近 20 条）"
                        )
                    
                    with gr.Tab("当前状态"):
                        status_display = gr.JSON(
                            value={},
                            label="无人机状态"
                        )
                    
                    with gr.Tab("LLM 原始输出"):
                        raw_output = gr.Textbox(
                            value="",
                            label="原始响应",
                            lines=10,
                            interactive=False
                        )
        
        # 事件绑定
        msg.submit(chat_fn, [msg, chatbot], [chatbot, msg]).then(
            update_displays, None, [plan_display, tool_calls_display, tool_history_display, status_display, raw_output]
        )
        submit_btn.click(chat_fn, [msg, chatbot], [chatbot, msg]).then(
            update_displays, None, [plan_display, tool_calls_display, tool_history_display, status_display, raw_output]
        )
        clear_btn.click(
            lambda: ([], "", [], [], {}, ""),
            None,
            [chatbot, plan_display, tool_calls_display, tool_history_display, status_display, raw_output]
        )
        
        # 页面加载时初始化显示
        def on_load():
            """页面加载时更新显示"""
            return update_displays()
        
        app.load(
            on_load,
            None,
            [plan_display, tool_calls_display, tool_history_display, status_display, raw_output]
        )
    
    return app, ui


def launch_gradio_app(agent_callback, server_name="127.0.0.1", server_port=7860, share=False):
    """
    启动 Gradio 应用
    
    Args:
        agent_callback: 处理用户输入的代理函数
        server_name: 服务器地址
        server_port: 服务器端口
        share: 是否创建公共链接
    """
    app, ui = create_gradio_app(agent_callback)
    logger.info(f"Gradio 界面启动: http://{server_name}:{server_port}")
    app.launch(server_name=server_name, server_port=server_port, share=share)
