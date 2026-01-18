"""
测试无人机智能体功能

使用示例：
    python scripts/test_agent.py
"""

from airsim_drone_agent.sim import AirSimClient
from airsim_drone_agent.mllm import create_mllm_client
from airsim_drone_agent.agent import DroneAgent
from airsim_drone_agent.utils.logger import setup_logging, get_logger

def main():
    # 设置日志
    setup_logging()
    logger = get_logger("test_agent")
    
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
    
    logger.info("创建智能体（纯视觉模式，使用5个摄像头）...")
    agent = DroneAgent(
        client=client,
        mllm=mllm,
    )
    logger.info("智能体创建成功！")
    
    # 示例 1: 单步执行
    logger.info("="*60)
    logger.info("示例 1: 单步执行 - 起飞并向前飞 5 米")
    logger.info("="*60)
    result = agent.step("起飞并向前飞 5 米")
    logger.info(f"执行结果: {result}")
    
    # 示例 2: 多轮对话
    # logger.info("\n" + "="*60)
    # logger.info("示例 2: 多轮对话 - 执行飞行任务")
    # logger.info("="*60)
    
    # # 注意：实际执行前请确保无人机已解锁并准备就绪
    # task = """
    # 请执行以下任务：
    # 1. 先获取当前状态
    # 2. 如果无人机未解锁，请解锁
    # 3. 起飞到高度 -5 米
    # 4. 移动到位置 (10, 10, -5)
    # 5. 悬停
    # 6. 获取最终状态
    # """
    
    # logger.info(f"任务: {task}")
    # history = agent.chat(task, max_steps=10)
    
    # logger.info("\n执行历史:")
    # for i, step in enumerate(history, 1):
    #     logger.info(f"\n步骤 {i}:")
    #     logger.info(f"  工具: {step.get('tool_name')}")
    #     logger.info(f"  参数: {step.get('arguments')}")
    #     logger.info(f"  成功: {step.get('success')}")
    #     if step.get('result'):
    #         logger.info(f"  结果: {step.get('result')}")
    #     if step.get('error'):
    #         logger.info(f"  错误: {step.get('error')}")
    
    logger.info("="*60)
    logger.info("测试完成！")
    logger.info("="*60)

if __name__ == "__main__":
    main()
