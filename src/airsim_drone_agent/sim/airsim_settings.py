from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

def default_airsim_settings_path() -> Path:
    """
    AirSim 默认读取的配置路径：
      Windows: %USERPROFILE%\\Documents\\AirSim\\settings.json
    """
    home = Path(os.path.expanduser("~"))
    return home / "Documents" / "AirSim" / "settings.json"

def apply_settings(template_settings_json: Path, target: Optional[Path] = None, overwrite: bool = True) -> Path:
    """
    将仓库里的 settings.json 模板复制到 AirSim 默认路径，方便一键配置。
    """
    template_settings_json = Path(template_settings_json)
    if not template_settings_json.exists():
        raise FileNotFoundError(f"Template settings not found: {template_settings_json}")

    target = Path(target) if target else default_airsim_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not overwrite:
        return target

    shutil.copyfile(template_settings_json, target)
    return target