"""Member 1：音频读取与统一预处理。

统一约定（供 SVM / RF / MLP / Demo 共用）：
    - 采样率 22050 Hz
    - 单声道
    - 长度统一为 4 秒（不足补零，超出截断）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import numpy as np

from config import DURATION, MONO, SAMPLE_RATE


def pad_or_trim(y: np.ndarray, sr: int, duration: float = DURATION) -> np.ndarray:
    """把波形补齐或截断到固定时长。"""
    target_length = int(round(sr * duration))
    return librosa.util.fix_length(np.asarray(y, dtype=np.float32), size=target_length)


def load_audio(path, sr: int = SAMPLE_RATE, mono: bool = MONO,
               duration: float = DURATION) -> tuple:
    """读取音频并完成重采样、声道统一、定长处理。

    Returns:
        (waveform, sample_rate)
    """
    y, sr_out = librosa.load(str(path), sr=sr, mono=mono)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:                     # 未做单声道转换时的兜底处理
        y = librosa.to_mono(y)
    if duration is not None:
        y = pad_or_trim(y, sr_out, duration)
    return y, sr_out


def validate_waveform(y: np.ndarray) -> bool:
    """检查波形是否可用：有限值、非全零。"""
    if y.size == 0:
        return False
    if not np.all(np.isfinite(y)):
        return False
    return bool(np.any(y != 0))


def load_checked(path, sr: int = SAMPLE_RATE) -> tuple:
    """读取音频并校验，异常时抛出带文件名的错误，便于定位坏文件。"""
    y, sr_out = load_audio(path, sr=sr)
    if not validate_waveform(y):
        raise ValueError(f"音频为空或含非法值：{path}")
    return y, sr_out
