"""
启动 Web 可视化界面
"""

import threading
import time
import gradio as gr
from airsim_drone_agent.sim import AirSimClient
from airsim_drone_agent.mllm import create_mllm_client
from airsim_drone_agent.agent import DroneAgent
from airsim_drone_agent.web import create_gradio_app
from airsim_drone_agent.utils.logger import setup_logging, get_logger


def agent_callback(user_input: str, ui):
    """处理用户输入的代理函数"""
    logger = get_logger("web_ui")
    
    try:
        # 执行单步或聊天
        if "任务" in user_input or "执行" in user_input or len(user_input) > 50:
            # 多步任务
            history = agent.chat(user_input, max_steps=10)
            response = f"任务执行完成，共 {len(history)} 步"
        else:
            # 单步执行
            result = agent.step(user_input, include_observation=True)
            if result.get("tool_name"):
                tool_name = result["tool_name"]
                success = result.get("success", False)
                plan = result.get("plan")
                if tool_name == "__batch__":
                    tool_calls = result.get("tool_calls", [])
                    summary = ", ".join([c.get("tool_name", "unknown") for c in tool_calls])
                    header = f"调用工具列表: {summary}, 结果: {'成功' if success else '失败'}"
                else:
                    header = f"调用工具: {tool_name}, 结果: {'成功' if success else '失败'}"
                if plan:
                    response = f"规划:\n{plan}\n\n{header}"
                else:
                    response = header
            else:
                plan = result.get("plan")
                if plan:
                    response = f"规划:\n{plan}\n\n{result.get('reason', '未调用工具')}"
                else:
                    response = result.get("reason", "未调用工具")
        
        ui.add_assistant_response(response)
        
    except Exception as e:
        error_msg = f"执行错误: {str(e)}"
        logger.error(error_msg, exc_info=True)
        ui.add_assistant_response(error_msg)


def main():
    setup_logging()
    logger = get_logger("web_ui")
    
    logger.info("初始化 AirSim 客户端...")
    client = AirSimClient(ip="127.0.0.1")
    client.connect()
    logger.info("连接成功！")
    
    logger.info("初始化 MLLM 客户端...")
    try:
        mllm = create_mllm_client()
        logger.info("MLLM 客户端初始化成功！")
    except Exception as e:
        logger.error(f"MLLM 客户端初始化失败: {e}")
        logger.info("请确保已配置 .env 文件中的 MLLM_PROVIDER, MLLM_API_KEY 等")
        return
    
    # 创建 Gradio UI（先创建 UI 对象）
    app, ui = create_gradio_app(agent_callback)
    
    # 创建 Agent（传入 UI 回调）
    global agent
    agent = DroneAgent(client=client, mllm=mllm, ui_callback=ui)
    logger.info("智能体创建成功！")
    
    # 启动 Gradio 界面
    logger.info("启动 Web 界面...")
    logger.info("访问地址: http://127.0.0.1:7860")
    app.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
