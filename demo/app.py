"""Member 5：Urban Sound Intelligence Streamlit Demo。

运行（在仓库根目录）：
    streamlit run demo/app.py

功能：上传 WAV -> 播放音频 -> 展示波形与 Mel 频谱 -> 选择模型 -> 输出预测类别与置信度。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import CLASS_NAMES, TEST_FOLD, TRAIN_FOLDS, VAL_FOLD
from demo.audio_loader import load_uploaded_audio
from demo.model_loader import available_models, load_model, model_metadata
from demo.prediction import predict_waveform
from demo.visualization import plot_probabilities, plot_spectrogram, plot_waveform

st.set_page_config(page_title="Urban Sound Intelligence", page_icon="🔊", layout="wide")


def render_sidebar(models: dict) -> str:
    st.sidebar.header("Model")
    selected = st.sidebar.radio("Choose a classifier", list(models.keys()))
    st.sidebar.caption(
        f"Evaluation split: Fold {list(TRAIN_FOLDS)} train / "
        f"Fold {VAL_FOLD} val / Fold {TEST_FOLD} test"
    )

    metadata = model_metadata()
    if selected in metadata:
        info = metadata[selected]
        with st.sidebar.expander("Model configuration", expanded=False):
            st.json(
                {
                    "best_params": info.get("best_params"),
                    "train_samples": info.get("train_samples"),
                    "n_features": info.get("n_features"),
                    "training_seconds": info.get("training_seconds"),
                }
            )
    return selected


def main() -> None:
    st.title("🔊 Urban Sound Intelligence")
    st.caption(
        "UrbanSound8K 十分类城市环境声音识别 · MFCC(20) 均值+标准差 · "
        "SVM / Random Forest / MLP"
    )

    models = available_models()
    if not models:
        st.error(
            "未在 saved_models/ 中找到已训练模型。请先按 README 完成训练：\n\n"
            "python -m models.svm_training\n\n"
            "python -m models.rf_training\n\n"
            "python -m models.mlp_training"
        )
        st.stop()

    selected_model_name = render_sidebar(models)
    model = load_model(str(models[selected_model_name]))

    uploaded = st.file_uploader("Upload a WAV file (≤ 4 s is recommended)", type=["wav"])
    if uploaded is None:
        st.info("上传一段城市环境声音（例如警笛、狗吠、钻孔）开始预测。")
        st.stop()

    try:
        audio = load_uploaded_audio(uploaded)
    except Exception as error:
        st.error(f"音频读取失败：{error}")
        st.stop()

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Input audio")
        st.audio(uploaded)
        st.write(
            f"Original: {audio['original_sr']} Hz, {audio['channels']} channel(s), "
            f"{audio['original_duration']} s  →  processed: {audio['sample_rate']} Hz, "
            f"mono, {audio['processed_duration']} s"
        )
        st.pyplot(plot_waveform(audio["waveform"], audio["sample_rate"]), clear_figure=True)

    with right:
        st.subheader("Prediction")
        result = predict_waveform(model, audio["waveform"], audio["sample_rate"])
        st.metric("Predicted class", result["class_name"].replace("_", " "))
        if result["confidence"] is not None:
            st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        st.caption(f"Model: {selected_model_name} | feature dimension: {result['n_features']}")

        if result["probabilities"] is not None:
            st.pyplot(
                plot_probabilities(
                    list(CLASS_NAMES), result["probabilities"], result["class_id"]
                ),
                clear_figure=True,
            )

    with st.expander("Mel spectrogram", expanded=False):
        st.pyplot(
            plot_spectrogram(audio["waveform"], audio["sample_rate"]), clear_figure=True
        )

    if result["probabilities"] is not None:
        top = (
            pd.DataFrame(
                {
                    "class": list(CLASS_NAMES),
                    "probability": result["probabilities"],
                }
            )
            .sort_values("probability", ascending=False)
            .head(5)
            .reset_index(drop=True)
        )
        top["probability"] = (top["probability"] * 100).round(2)
        st.subheader("Top-5 probabilities (%)")
        st.dataframe(top, use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "注意：模型仅在干净音频（Fold 1-9）上训练，强背景噪声下置信度可能明显下降；"
        "演示结果不构成对真实城市部署性能的保证。"
    )
    plt.close("all")


if __name__ == "__main__":
    main()
