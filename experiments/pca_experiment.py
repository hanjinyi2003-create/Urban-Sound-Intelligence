"""Member 2：PCA 降维实验（P1）。

对比三种特征设置下 SVM 的维度、训练耗时与分类表现：
    - 原始 40 维（无 PCA）
    - PCA 保留 20 维
    - PCA 保留 95% 方差

PCA 与 StandardScaler 都只在训练数据上 fit，验证/测试数据只做 transform。

运行：
    python -m experiments.pca_experiment
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    ensure_dirs,
)
from experiments.evaluation import evaluate_model
from preprocessing.data_split import load_split


def load_svm_params() -> dict:
    path = MODELS_DIR / "svm_params.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在，请先运行 python -m models.svm_training 完成 SVM 调参与训练"
        )
    with open(path, encoding="utf-8") as handle:
        params = json.load(handle)["best_params"]
    gamma = params["gamma"]
    gamma = float(gamma) if gamma not in ("scale", "auto") else gamma
    return {"C": float(params["C"]), "gamma": gamma}


def build_variant(n_components, params: dict) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if n_components is not None:
        steps.append(
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE))
        )
    steps.append(
        (
            "svm",
            SVC(
                C=params["C"],
                gamma=params["gamma"],
                kernel="rbf",
                random_state=RANDOM_STATE,
            ),
        )
    )
    return Pipeline(steps)


def run_variant(label, n_components, params, split) -> dict:
    model = build_variant(n_components, params)

    started = time.perf_counter()
    model.fit(
        np.vstack([split["X_train"], split["X_val"]]),
        np.concatenate([split["y_train"], split["y_val"]]),
    )
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    metrics = evaluate_model(model, split["X_test"], split["y_test"])
    predict_seconds = time.perf_counter() - started

    dimensions = (
        int(model.named_steps["pca"].n_components_)
        if "pca" in model.named_steps
        else int(split["X_train"].shape[1])
    )
    explained = (
        float(np.sum(model.named_steps["pca"].explained_variance_ratio_))
        if "pca" in model.named_steps
        else 1.0
    )

    record = {
        "variant": label,
        "n_components": dimensions,
        "explained_variance": round(explained, 4),
        "accuracy": round(metrics["accuracy"], 4),
        "macro_precision": round(metrics["macro_precision"], 4),
        "macro_recall": round(metrics["macro_recall"], 4),
        "macro_f1": round(metrics["macro_f1"], 4),
        "train_seconds": round(train_seconds, 2),
        "predict_seconds": round(predict_seconds, 3),
        "predict_ms_per_sample": round(predict_seconds / len(split["y_test"]) * 1000, 3),
    }
    print(
        f"{label:<22} dims={dimensions:<3} "
        f"Macro-F1={record['macro_f1']:.4f} "
        f"train={record['train_seconds']:.1f}s"
    )
    return record


def plot_results(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    labels = [f"{row.variant}\n({row.n_components} dims)" for row in table.itertuples()]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    axes[0].bar(labels, table["macro_f1"], color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Macro-F1 (Fold 10)")
    axes[0].set_title("SVM performance with / without PCA")
    for x, value in enumerate(table["macro_f1"]):
        axes[0].text(x, value + 0.01, f"{value:.3f}", ha="center", fontsize=9)

    axes[1].bar(labels, table["train_seconds"], color=colors)
    axes[1].set_ylabel("Training time (s)")
    axes[1].set_title("Training cost vs. feature dimension")
    for x, value in enumerate(table["train_seconds"]):
        axes[1].text(x, value + 0.5, f"{value:.1f}s", ha="center", fontsize=9)

    for axis in axes:
        axis.tick_params(axis="x", labelsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    params = load_svm_params()
    split = load_split()
    print(f"使用 SVM 参数：{params}\n")

    records = [
        run_variant("Raw 40-dim", None, params, split),
        run_variant("PCA 20 components", 20, params, split),
        run_variant("PCA 95% variance", 0.95, params, split),
    ]

    table = pd.DataFrame(records)
    table.to_csv(TABLES_DIR / "pca_results.csv", index=False)
    with open(TABLES_DIR / "pca_results.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    plot_results(table)

    baseline = table.loc[0]
    print(
        f"\nPCA 20 维相对原始 40 维：Macro-F1 变化 "
        f"{table.loc[1, 'macro_f1'] - baseline['macro_f1']:+.4f}，"
        f"训练时间变化 {table.loc[1, 'train_seconds'] - baseline['train_seconds']:+.1f}s"
    )
    print(f"已保存：{TABLES_DIR / 'pca_results.csv'}")
    print(f"已保存：{FIGURES_DIR / 'pca_comparison.png'}")


if __name__ == "__main__":
    main()
