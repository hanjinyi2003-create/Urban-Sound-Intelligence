"""Member 5：上传音频的读取与统一预处理。

与训练阶段保持完全一致：22050 Hz、单声道、4 秒定长。
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import numpy as np
import soundfile as sf

from config import DURATION, SAMPLE_RATE
from preprocessing.audio_processing import pad_or_trim


def load_uploaded_audio(uploaded_file, sr: int = SAMPLE_RATE) -> dict:
    """读取 Streamlit 上传的音频文件（WAV）。

    Returns:
        dict: waveform（已重采样、单声道、定长）、原始采样率、原始时长、处理说明
    """
    raw = uploaded_file.read() if hasattr(uploaded_file, "read") else Path(uploaded_file).read_bytes()
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    data, original_sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError("音频文件为空")

    channels = data.shape[1]
    waveform = data.mean(axis=1) if channels > 1 else data[:, 0]
    original_duration = len(waveform) / original_sr

    if original_sr != sr:
        waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=sr)
    waveform = pad_or_trim(waveform.astype(np.float32), sr, DURATION)

    return {
        "waveform": waveform,
        "sample_rate": sr,
        "original_sr": int(original_sr),
        "channels": int(channels),
        "original_duration": round(original_duration, 2),
        "processed_duration": DURATION,
    }
