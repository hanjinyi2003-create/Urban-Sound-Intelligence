"""Member 5：已训练模型的加载与状态检查。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import streamlit as st

from config import MODELS_DIR

MODEL_FILES = {
    "SVM (RBF)": "svm.joblib",
    "Random Forest": "rf.joblib",
    "MLP": "mlp.joblib",
}


def available_models() -> dict:
    """返回 {显示名: 模型路径}，只包含磁盘上真实存在的模型。"""
    return {
        name: MODELS_DIR / filename
        for name, filename in MODEL_FILES.items()
        if (MODELS_DIR / filename).exists()
    }


@st.cache_resource(show_spinner=False)
def load_model(path_str: str):
    """缓存加载模型，避免每次交互都重新读盘。"""
    return joblib.load(path_str)


def model_metadata() -> dict:
    """读取各模型的参数文件，用于在页面上展示配置。"""
    import json

    metadata = {}
    for name, filename in MODEL_FILES.items():
        params_path = MODELS_DIR / filename.replace(".joblib", "_params.json")
        if params_path.exists():
            with open(params_path, encoding="utf-8") as handle:
                metadata[name] = json.load(handle)
    return metadata
