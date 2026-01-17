from pathlib import Path
from pprint import pprint

from airsim_drone_agent.sim import DroneEnv
from airsim_drone_agent.enums import DroneCamera
from airsim_drone_agent.mllm import create_mllm_client, Message, TextPart, ImagePart
from airsim_drone_agent.mllm.image_codec import bgr_to_jpeg_data_url
import cv2

def main():
    env = DroneEnv()
    env.connect()

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

    for cam in cams:
        cam_name = cam.name
        frame = env.sensors.rgb(camera=cam, as_bgr_for_cv=True)
        img_bgr = frame.data

        print(f"[{cam_name}] camera_name={cam.airsim_name}  shape={img_bgr.shape}  dtype={img_bgr.dtype}")

        save_path = out_dir / f"{cam.name}.png"
        cv2.imwrite(str(save_path), img_bgr)
        print(f"  已保存：{save_path}")

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

    print("模型原始输出：\n", result.text)
    print("\n解析到的 JSON: \n", result.json)


if __name__ == "__main__":
    main()