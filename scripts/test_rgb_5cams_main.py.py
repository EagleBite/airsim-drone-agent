from airsim_drone_agent.sim.sensors import ImageFrame


from pathlib import Path
from pprint import pprint

from airsim_drone_agent.sim.airsim_client import AirSimClient
from airsim_drone_agent.enums import DroneCamera
from airsim_drone_agent.mllm import create_mllm_client, Message, TextPart, ImagePart
from airsim_drone_agent.mllm.image_codec import bgr_to_jpeg_data_url
from airsim_drone_agent.utils.logger import setup_logging, get_logger, log_data
import cv2

def main():
    setup_logging()
    logger = get_logger(__name__)

    Drone1 = AirSimClient(vehicle_name="Drone1")
    Drone1.connect()

    ROOT = Path(__file__).resolve().parents[1]
    out_dir = ROOT / "outputs" / "rgb_5cams_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    user_parts = [TextPart("你将看到无人机的 5 个摄像头画面。请理解场景并输出 JSON。")]
    cams = [
        DroneCamera.FRONT_CENTER,
        DroneCamera.FRONT_LEFT,
        DroneCamera.FRONT_RIGHT,
        DroneCamera.BOTTOM_CENTER,
        DroneCamera.BACK_CENTER,
    ]

    frames = Drone1.sensors.get_rgb_frames(cameras=cams, as_bgr_for_cv=True)
    for i, frame in enumerate(frames):
        img_bgr = frame.data
        save_path = out_dir / f"{frame.frame_type}_{i}.png"
        cv2.imwrite(str(save_path), img_bgr[:, :, ::-1])
        logger.info("已保存：%s", save_path)

        cam_name = cams[i].name
        user_parts.append(TextPart(f"摄像头：{cam_name}"))
        user_parts.append(ImagePart(
            url=bgr_to_jpeg_data_url(img_bgr),
            name=cam_name
        ))

    system_text = (
        "你是无人机视觉理解模块。\n"
        "只输出 JSON, 不要输出额外解释文字。\n"
        "输出 JSON 字段必须包含：\n"
        "- scene_summary: 字符串\n"
        "- obstacles: 数组 (每项包含 camera/desc/risk)\n"
        "- suggestion: 字符串 (forward/left/right/up/down/back/hover)\n"
        "- confidence: 0 到 1 的数字\n"
    )

    messages = [
        Message.system(system_text),
        Message.user(user_parts)
    ]

    client = create_mllm_client()
    result = client.generate(messages, max_tokens=300)

    log_data(logger, result.text, title="模型原始输出")
    log_data(logger, result.json, title="解析到的 JSON")


if __name__ == "__main__":
    main()