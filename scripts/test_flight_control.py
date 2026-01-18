"""
测试无人机飞行控制功能

使用示例：
    python scripts/test_flight_control.py
"""

from airsim_drone_agent.sim.airsim_client import AirSimClient
import time

def main():
    # 创建环境
    Drone1 = AirSimClient(ip="127.0.0.1")
    
    print("正在连接 AirSim...")
    Drone1.connect()
    print("连接成功！")
    
    # 解锁
    print("解锁无人机...")
    Drone1.arm(armed=True)
    time.sleep(1)
    
    # 起飞
    print("起飞...")
    Drone1.takeoff(timeout_sec=10.0)
    print("起飞完成！")
    
    # 悬停 2 秒
    print("悬停 2 秒...")
    Drone1.hover()
    time.sleep(2)

    print(Drone1.sensors.get_vehicle_pose())
    
    # 示例 1: 位置控制 - 移动到指定位置
    print("移动到位置 (10, 10, -5)...")
    Drone1.move_to_position(x=10.0, y=10.0, z=-5.0, velocity=5.0)
    print("到达目标位置！")
    time.sleep(1)
    
    # 示例 2: 高度控制 - 改变高度
    print("上升到高度 -10 米...")
    Drone1.move_to_z(z=-10.0, velocity=2.0)
    print("高度调整完成！")
    time.sleep(1)
    
    # 示例 3: 速度控制 - 按速度移动
    print("以速度 (5, 0, 0) 移动 3 秒...")
    Drone1.move_by_velocity(vx=5.0, vy=0.0, vz=0.0, duration=3.0)
    print("速度控制完成！")
    time.sleep(1)
    
    # 示例 4: 速度控制 + 高度保持
    print("以速度 (0, 5, 0) 移动，保持高度 -10 米，持续 3 秒...")
    Drone1.move_by_velocity_z(vx=0.0, vy=5.0, z=-10.0, duration=3.0)
    print("速度控制完成！")
    time.sleep(1)
    
    # 示例 5: 旋转控制 - 旋转到指定偏航角
    print("旋转到偏航角 90 度...")
    Drone1.rotate_to_yaw(yaw=90.0, margin=5.0)
    print("旋转完成！")
    time.sleep(1)
    
    # 示例 6: 路径规划 - 沿路径移动
    print("沿路径移动...")
    path = [
        (0.0, 0.0, -10.0),
        (10.0, 10.0, -10.0),
        (20.0, 10.0, -10.0),
        (20.0, 0.0, -10.0),
        (0.0, 0.0, -10.0),
    ]
    Drone1.move_on_path(path, velocity=5.0)
    print("路径飞行完成！")
    time.sleep(1)
    
    # 悬停
    print("悬停...")
    Drone1.hover()
    time.sleep(2)
    
    # 降落
    print("降落...")
    Drone1.land(timeout_sec=20.0)
    print("降落完成！")
    
    print("测试完成！")

if __name__ == "__main__":
    main()
