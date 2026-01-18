from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import threading
from typing import Any
import cv2
import numpy as np

_config_lock = threading.Lock()
_configured = False

# 自定义日志级别 DATA (在 INFO 和 WARNING 之间)
DATA_LEVEL = 25
logging.addLevelName(DATA_LEVEL, "DATA")

# ANSI 颜色代码
class Colors:
    """ANSI 颜色代码"""
    RESET = '\033[0m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    WHITE = '\033[37m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器（仅用于控制台输出）"""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.WHITE,
        DATA_LEVEL: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # 获取原始格式化的消息
        log_message = super().format(record)
        
        # 根据日志级别添加颜色
        color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)
        colored_message = f"{color}{log_message}{Colors.RESET}"
        
        return colored_message


def format_data(data: Any, indent: int = 2, max_depth: int = 10, current_depth: int = 0) -> str:
    """
    格式化数据(Dict/List)为易读的字符串
    
    Args:
        data: 要格式化的数据(Dict、List 或其他)
        indent: 缩进空格数
        max_depth: 最大递归深度
        current_depth: 当前递归深度
    
    Returns:
        格式化后的字符串
    """
    if current_depth >= max_depth:
        return "... (max depth reached)"
    
    indent_str = " " * (indent * current_depth)
    
    if isinstance(data, dict):
        if not data:
            return "{}"
        lines = ["{"]
        for key, value in data.items():
            key_str = f'"{key}"' if isinstance(key, str) else str(key)
            value_str = format_data(value, indent, max_depth, current_depth + 1)
            lines.append(f"{indent_str}  {key_str}: {value_str}")
        lines.append(f"{indent_str}}}")
        return "\n".join(lines)
    
    elif isinstance(data, (list, tuple)):
        if not data:
            return "[]" if isinstance(data, list) else "()"
        # 如果列表很短且元素都是简单类型，可以单行显示
        if len(data) <= 5 and all(isinstance(item, (int, float, bool, str, type(None))) for item in data):
            items_str = ", ".join(format_data(item, indent, max_depth, current_depth + 1) for item in data)
            return f"[{items_str}]" if isinstance(data, list) else f"({items_str})"
        # 否则多行显示
        lines = ["[" if isinstance(data, list) else "("]
        for item in data:
            item_str = format_data(item, indent, max_depth, current_depth + 1)
            lines.append(f"{indent_str}  {item_str},")
        lines.append(f"{indent_str}]" if isinstance(data, list) else f"{indent_str})")
        return "\n".join(lines)
    
    elif isinstance(data, str):
        # 如果字符串太长，截断并添加省略号
        max_len = 100
        if len(data) > max_len:
            return f'"{data[:max_len]}..." (length: {len(data)})'
        return f'"{data}"'
    
    elif isinstance(data, (int, float, bool, type(None))):
        return str(data)
    
    elif isinstance(data, np.ndarray):
        shape_str = "x".join(map(str, data.shape))
        dtype_str = str(data.dtype)
        return f"ndarray(shape={shape_str}, dtype={dtype_str})"
    
    else:
        # 对于其他类型，尝试使用 repr，如果太长则截断
        repr_str = repr(data)
        max_len = 200
        if len(repr_str) > max_len:
            return f"{repr_str[:max_len]}... (type: {type(data).__name__})"
        return repr_str


def setup_logging(level=logging.DEBUG, log_dir: Path | None = None, enable_colors: bool = True):
    """
    配置根日志记录器（仅配置一次）
    
    Args:
        level: 日志级别
        log_dir: 日志文件目录，如果为 None 则使用默认路径
        enable_colors: 是否启用控制台颜色输出
    
    Returns:
        配置好的根日志记录器
    """
    global _configured
    with _config_lock:
        if _configured:
            return logging.getLogger()

        if log_dir is None:
            log_dir = Path.cwd() / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger()
        logger.setLevel(level)

        ch = logging.StreamHandler()
        ch.setLevel(level)
        if enable_colors:
            ch.setFormatter(ColoredFormatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        else:
            ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(ch)

        # 过滤第三方库的调试日志（httpcore, httpx 等）
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

        _configured = True
        logger.debug("Logging configured, level=%s, log_dir=%s, colors=%s", level, log_dir, enable_colors)
        return logger


def get_logger(name: str = None):
    """
    获取配置好的日志记录器
    
    Args:
        name: 日志记录器名称，如果为 None 则返回根日志记录器
    
    Returns:
        配置好的日志记录器
    """
    setup_logging()
    return logging.getLogger(name)


def get_data_type_info(data: Any) -> str:
    """
    获取数据类型信息
    
    Args:
        data: 要分析的数据
    
    Returns:
        数据类型信息字符串
    """
    data_type = type(data).__name__
    
    if isinstance(data, dict):
        size = len(data)
        keys_info = f"keys: {list(data.keys())[:5]}" if size <= 5 else f"keys: {list(data.keys())[:5]}... (total: {size})"
        return f"Type: {data_type}, Size: {size}, {keys_info}"
    
    elif isinstance(data, (list, tuple)):
        size = len(data)
        item_type = type(data[0]).__name__ if size > 0 else "None"
        return f"Type: {data_type}, Length: {size}, Item Type: {item_type}"
    
    elif isinstance(data, np.ndarray):
        shape_str = "x".join(map(str, data.shape))
        dtype_str = str(data.dtype)
        return f"Type: {data_type}, Shape: {shape_str}, Dtype: {dtype_str}"
    
    elif isinstance(data, str):
        return f"Type: {data_type}, Length: {len(data)}"
    
    else:
        return f"Type: {data_type}"


def log_data(logger: logging.Logger, data: Any, title: str = "Data"):
    """
    记录数据结构（Dict/List）的格式化输出，使用 DATA 级别和绿色显示
    
    Args:
        logger: 日志记录器
        data: 要记录的数据（Dict、List 或其他）
        title: 数据标题
    
    Example:
        >>> logger = get_logger()
        >>> data = {"x": 10, "y": 20, "nested": {"a": 1, "b": [1, 2, 3]}}
        >>> log_data(logger, data, title="Position")
    """
    try:
        # 获取数据类型信息
        type_info = get_data_type_info(data)
        
        # 格式化数据内容
        formatted = format_data(data)
        
        # 构建完整消息（用于文件日志）
        full_message = f"{title} | {type_info}\n{formatted}"
        
        # 检查是否启用颜色（通过检查根日志记录器的处理器）
        root_logger = logging.getLogger()
        enable_colors = False
        console_handler = None
        
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and isinstance(handler.formatter, ColoredFormatter):
                enable_colors = True
                console_handler = handler
                break
        
        if enable_colors and console_handler:
            # 创建日志记录用于格式化标题行（使用 DATA 级别）
            formatter = console_handler.formatter
            record = logging.LogRecord(
                name=logger.name,
                level=DATA_LEVEL,
                pathname="",
                lineno=0,
                msg=f"{title} | {type_info}",
                args=(),
                exc_info=None
            )
            
            # 格式化标题和类型信息（使用 DATA 级别的绿色）
            header_formatted = formatter.format(record)
            
            # 数据内容使用独立的颜色（绿色），每行都添加颜色
            data_lines = formatted.split('\n')
            data_colored = '\n'.join(f"{Colors.GREEN}{line}{Colors.RESET}" for line in data_lines)
            
            # 输出到控制台
            print(f"{header_formatted}\n{data_colored}")
            
            # 记录到文件（无颜色，使用完整消息，使用 DATA 级别）
            # 只发送给文件处理器，避免控制台重复输出
            for handler in root_logger.handlers:
                if isinstance(handler, RotatingFileHandler):
                    file_record = logging.LogRecord(
                        name=logger.name,
                        level=DATA_LEVEL,
                        pathname="",
                        lineno=0,
                        msg=full_message,
                        args=(),
                        exc_info=None
                    )
                    handler.emit(file_record)
        else:
            # 如果未启用颜色，正常记录（使用 DATA 级别）
            logger.log(DATA_LEVEL, full_message)
            
    except Exception as e:
        logger.exception("Failed to log data: %s", e)


def log_image_summary(logger, cam_name: str, img_bgr: np.ndarray):
    """Log concise information about an image: shape, dtype, stats and encoded JPEG size."""
    try:
        h, w = img_bgr.shape[:2]
        dtype = str(img_bgr.dtype)
        minv = float(img_bgr.min()) if img_bgr.size else None
        maxv = float(img_bgr.max()) if img_bgr.size else None
        meanv = float(img_bgr.mean()) if img_bgr.size else None
        channels = img_bgr.shape[2] if img_bgr.ndim == 3 else 1

        # try to encode to JPEG to get approximate size
        ok, enc = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        jpg_size = int(enc.nbytes) if ok and enc is not None else None

        logger.debug(
            "[%s] image summary: %dx%d channels=%s dtype=%s min=%s max=%s mean=%.2f jpg_bytes=%s",
            cam_name, w, h, channels, dtype, minv, maxv, meanv if meanv is not None else 0.0, jpg_size,
        )
    except Exception as e:
        logger.exception("Failed to summarize image for %s: %s", cam_name, e)
