"""统一评价模块：Accuracy、Macro-Precision / Recall / F1、Confusion Matrix。

三个模型必须通过在相同测试集（Fold 10）与相同阈值下调用本模块，
保证指标口径完全一致。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import CLASS_NAMES, FIGURES_DIR, METRICS_DIR, TABLES_DIR, ensure_dirs


def evaluate_model(model, X, y) -> dict:
    """计算统一指标，返回可直接写入 JSON 的字典。"""
    predictions = model.predict(X)
    labels = list(range(len(CLASS_NAMES)))
    return {
        "n_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_precision": float(
            precision_score(y, predictions, average="macro", zero_division=0, labels=labels)
        ),
        "macro_recall": float(
            recall_score(y, predictions, average="macro", zero_division=0, labels=labels)
        ),
        "macro_f1": float(
            f1_score(y, predictions, average="macro", zero_division=0, labels=labels)
        ),
        "confusion_matrix": confusion_matrix(y, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            y,
            predictions,
            labels=labels,
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
    }


def save_metrics(metrics: dict, filename: str) -> Path:
    ensure_dirs()
    path = METRICS_DIR / filename
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return path


def plot_confusion_matrix(matrix, title: str, filename: str) -> Path:
    """绘制混淆矩阵（含数值与行归一化颜色）。"""
    ensure_dirs()
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised rate")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            if value == 0:
                continue
            color = "white" if normalized[row, col] > 0.55 else "black"
            ax.text(col, row, value, ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def format_metrics(metrics: dict) -> str:
    return (
        f"Accuracy {metrics['accuracy']:.4f} | "
        f"Macro-P {metrics['macro_precision']:.4f} | "
        f"Macro-R {metrics['macro_recall']:.4f} | "
        f"Macro-F1 {metrics['macro_f1']:.4f}"
    )


MODEL_METRIC_FILES = (
    ("SVM (RBF)", "svm_metrics.json"),
    ("Random Forest", "rf_metrics.json"),
    ("MLP", "mlp_metrics.json"),
)


def build_comparison_table() -> pd.DataFrame:
    """汇总三个模型在同一测试集（Fold 10）上的指标。"""
    ensure_dirs()
    rows = []
    for model_name, filename in MODEL_METRIC_FILES:
        path = METRICS_DIR / filename
        if not path.exists():
            print(f"跳过 {model_name}：未找到 {path}")
            continue
        with open(path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        rows.append(
            {
                "model": model_name,
                "accuracy": round(metrics["accuracy"], 4),
                "macro_precision": round(metrics["macro_precision"], 4),
                "macro_recall": round(metrics["macro_recall"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "train_seconds": metrics.get("train_seconds", ""),
            }
        )
    if not rows:
        raise FileNotFoundError("没有可用的模型指标文件，请先完成三个模型的训练。")

    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    plot_model_comparison(table)
    print(table.to_string(index=False))
    print(f"\n已保存：{TABLES_DIR / 'model_comparison.csv'}")
    print(f"已保存：{FIGURES_DIR / 'model_comparison.png'}")
    return table


def plot_model_comparison(table: pd.DataFrame) -> Path:
    """绘制 Accuracy 与 Macro-F1 的横向对比柱状图。"""
    ensure_dirs()
    positions = np.arange(len(table))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(positions - width / 2, table["accuracy"], width, label="Accuracy", color="#4C72B0")
    ax.bar(positions + width / 2, table["macro_f1"], width, label="Macro-F1", color="#DD8452")
    ax.set_xticks(positions)
    ax.set_xticklabels(table["model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the same test set (Fold 10)")
    ax.legend()
    for index, row in table.iterrows():
        ax.text(index - width / 2, row["accuracy"] + 0.01, f"{row['accuracy']:.3f}",
                ha="center", fontsize=8)
        ax.text(index + width / 2, row["macro_f1"] + 0.01, f"{row['macro_f1']:.3f}",
                ha="center", fontsize=8)
    fig.tight_layout()
    path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    build_comparison_table()


if __name__ == "__main__":
    main()
