"""Member 5：单条音频的预测逻辑。

与训练阶段复用同一个 mfcc_statistics 函数，保证特征定义一致。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import CLASS_NAMES, SAMPLE_RATE
from preprocessing.feature_extraction import mfcc_statistics


def predict_waveform(model, waveform: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """对一条已预处理的波形做预测。

    Returns:
        dict: 预测类别、置信度、各类别概率、特征维度
    """
    features = mfcc_statistics(waveform, sr).reshape(1, -1)

    predicted_id = int(model.predict(features)[0])
    result = {
        "class_id": predicted_id,
        "class_name": CLASS_NAMES[predicted_id],
        "n_features": int(features.shape[1]),
        "probabilities": None,
        "confidence": None,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = np.asarray(model.classes_, dtype=int)
        full = np.zeros(len(CLASS_NAMES), dtype=float)
        full[classes] = probabilities
        result["probabilities"] = full
        result["confidence"] = float(full[predicted_id])

    return result
