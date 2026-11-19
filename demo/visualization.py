"""Member 5：波形、频谱与预测结果的绘图函数。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import HOP_LENGTH, N_FFT, SAMPLE_RATE


def plot_waveform(waveform: np.ndarray, sr: int = SAMPLE_RATE):
    """绘制音频波形图。"""
    fig, ax = plt.subplots(figsize=(8, 2.6))
    times = np.arange(len(waveform)) / sr
    ax.plot(times, waveform, color="#4C72B0", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.set_xlim(0, times[-1] if times.size else 1)
    fig.tight_layout()
    return fig


def plot_spectrogram(waveform: np.ndarray, sr: int = SAMPLE_RATE):
    """绘制 Mel 频谱图。"""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(waveform, dtype=np.float32),
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=128,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    image = librosa.display.specshow(
        mel_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    fig.colorbar(image, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel Spectrogram")
    fig.tight_layout()
    return fig


def plot_probabilities(class_names, probabilities, predicted_index: int = None):
    """绘制各类别预测概率条形图（按概率降序）。"""
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(probabilities)[::-1]
    labels = [class_names[i] for i in order]
    values = probabilities[order]
    colors = [
        "#C44E52" if predicted_index is not None and i == predicted_index else "#4C72B0"
        for i in order
    ]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_title("Class Probabilities")
    for index, value in enumerate(values[::-1]):
        ax.text(value + 0.01, index, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    return fig
