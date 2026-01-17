from enum import Enum

class DroneCamera(Enum):
    """五摄像头逻辑名称 -> AirSim camera_name 映射"""
    FRONT_CENTER = "0"
    FRONT_LEFT = "1"
    FRONT_RIGHT = "2"
    BOTTOM_CENTER = "3"
    BACK_CENTER = "4"

    @property
    def airsim_name(self) -> str:
        return self.value
