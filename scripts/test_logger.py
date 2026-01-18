"""
测试优化后的 Logger 功能

演示：
1. 彩色日志输出 (错误红色、警告黄色、正常白色)
2. 数据格式化展示 (Dict/List)

运行：
    python scripts/test_logger.py
"""

from airsim_drone_agent.utils.logger import get_logger, log_data
import logging

def main():
    # 获取日志记录器
    logger = get_logger("test_logger")
    
    print("=" * 60)
    print("测试彩色日志输出")
    print("=" * 60)
    
    # 测试不同级别的日志（会显示不同颜色）
    logger.debug("这是一条 DEBUG 信息（青色）")
    logger.info("这是一条 INFO 信息（白色）")
    logger.warning("这是一条 WARNING 信息（黄色）")
    logger.error("这是一条 ERROR 信息（红色）")
    logger.critical("这是一条 CRITICAL 信息（红色）")
    
    print("\n" + "=" * 60)
    print("测试数据格式化展示")
    print("=" * 60)
    
    # 测试简单的字典
    simple_dict = {
        "x": 10.5,
        "y": 20.3,
        "z": -5.0,
        "active": True
    }
    log_data(logger, simple_dict, title="简单位置数据")
    
    # 测试嵌套字典
    nested_dict = {
        "drone_state": {
            "position": {"x": 0.0, "y": 0.0, "z": -10.0},
            "velocity": {"vx": 5.0, "vy": 0.0, "vz": 0.0},
            "orientation": {"roll": 0.0, "pitch": 0.0, "yaw": 45.0}
        },
        "sensors": {
            "imu": {"accel": [0.1, 0.2, 9.8], "gyro": [0.0, 0.0, 0.0]},
            "gps": {"lat": 39.9042, "lon": 116.4074, "alt": 100.0}
        },
        "status": "flying"
    }
    log_data(logger, nested_dict, title="无人机状态数据")
    
    # 测试列表
    path_list = [
        (0.0, 0.0, -10.0),
        (10.0, 10.0, -10.0),
        (20.0, 10.0, -10.0),
        (20.0, 0.0, -10.0),
    ]
    log_data(logger, path_list, title="飞行路径点")
    
    # 测试混合数据结构
    mixed_data = {
        "waypoints": [
            {"id": 1, "pos": [0, 0, -10], "speed": 5.0},
            {"id": 2, "pos": [10, 10, -10], "speed": 5.0},
            {"id": 3, "pos": [20, 0, -10], "speed": 3.0},
        ],
        "config": {
            "max_speed": 10.0,
            "timeout": 60.0,
            "enabled": True
        }
    }
    log_data(logger, mixed_data, title="混合数据结构")
    
    # 测试空数据结构
    empty_dict = {}
    empty_list = []
    log_data(logger, empty_dict, title="空字典")
    log_data(logger, empty_list, title="空列表")
    
    # 测试字符串截断
    long_string = "这是一个很长的字符串，" * 20
    log_data(logger, {"description": long_string}, title="长字符串测试")
    
    # 测试 DATA 级别
    test_data = {"level": "data", "message": "这是 DATA 级别的数据"}
    log_data(logger, test_data, title="DATA 级别数据")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n提示：")
    print("- 错误信息显示为红色")
    print("- 警告信息显示为黄色")
    print("- 正常信息显示为白色")
    print("- DEBUG 信息显示为青色")
    print("- 数据展示使用 DATA 级别，显示为绿色，并在顶部显示数据类型信息")
    print("- 文件日志中不包含颜色代码，保持纯文本格式")

if __name__ == "__main__":
    main()
