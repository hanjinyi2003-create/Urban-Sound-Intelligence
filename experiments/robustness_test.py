"""Member 4：噪声鲁棒性实验。

所有模型都用干净数据（Fold 1-9）训练，只在含噪测试集（Fold 10 + 白噪声）上评价，
研究的是「未针对噪声训练时的鲁棒性」。

运行前需完成：
    python -m experiments.noise_generator --snr 20 10

运行：
    python -m experiments.robustness_test
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, MODELS_DIR, TABLES_DIR
from experiments.evaluation import evaluate_model
from experiments.noise_generator import NOISE_DIR, load_noisy_test_features
from preprocessing.data_split import load_split

MODEL_FILES = (
    ("SVM (RBF)", "svm.joblib"),
    ("Random Forest", "rf.joblib"),
    ("MLP", "mlp.joblib"),
)

COLORS = {"SVM (RBF)": "#4C72B0", "Random Forest": "#DD8452", "MLP": "#55A868"}
MARKERS = {"SVM (RBF)": "o", "Random Forest": "s", "MLP": "^"}


def discovered_snr_levels() -> list:
    """扫描已生成的含噪测试特征文件，返回按 SNR 从高到低排序的条件列表。"""
    pattern = re.compile(r"test_(.+)dB\.npz$")
    levels = []
    if NOISE_DIR.exists():
        for path in NOISE_DIR.glob("test_*dB.npz"):
            match = pattern.search(path.name)
            if match:
                levels.append(float(match.group(1)))
    return sorted(levels, reverse=True)


def main() -> None:
    split = load_split()
    levels = discovered_snr_levels()
    if not levels:
        raise FileNotFoundError(
            "未找到含噪测试特征，请先运行 "
            "python -m experiments.noise_generator --snr 20 10"
        )

    conditions = [("Clean", None, split["X_test"], split["y_test"])]
    for snr in levels:
        data = load_noisy_test_features(snr)
        conditions.append((f"{snr:g} dB", snr, data["X"], data["y"]))

    rows = []
    for model_name, filename in MODEL_FILES:
        path = MODELS_DIR / filename
        if not path.exists():
            print(f"跳过 {model_name}：{path} 不存在，请先完成该模型训练")
            continue
        model = joblib.load(path)
        for label, snr, X, y in conditions:
            metrics = evaluate_model(model, X, y)
            rows.append(
                {
                    "model": model_name,
                    "condition": label,
                    "snr_db": "" if snr is None else snr,
                    "accuracy": round(metrics["accuracy"], 4),
                    "macro_precision": round(metrics["macro_precision"], 4),
                    "macro_recall": round(metrics["macro_recall"], 4),
                    "macro_f1": round(metrics["macro_f1"], 4),
                    "n_samples": metrics["n_samples"],
                }
            )
            print(
                f"{model_name:<14} {label:<8} "
                f"Accuracy={metrics['accuracy']:.4f} Macro-F1={metrics['macro_f1']:.4f}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "robustness_results.csv", index=False)

    pivot = table.pivot(index="model", columns="condition", values="macro_f1")
    order = [label for label, _, _, _ in conditions]
    pivot = pivot.reindex(columns=order)
    pivot.to_csv(TABLES_DIR / "robustness_macro_f1_pivot.csv")
    print("\nMacro-F1 汇总：")
    print(pivot.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in pivot.index:
        ax.plot(
            order,
            pivot.loc[model_name].values,
            marker=MARKERS.get(model_name, "o"),
            color=COLORS.get(model_name),
            linewidth=2,
            label=model_name,
        )
    ax.set_xlabel("Test condition")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Robustness to white noise (trained on clean audio only)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_macro_f1.png", dpi=150)
    plt.close(fig)

    if len(order) > 1:
        print("\nMacro-F1 下降幅度（Clean -> 最强噪声）：")
        for model_name in pivot.index:
            drop = pivot.loc[model_name, order[0]] - pivot.loc[model_name, order[-1]]
            print(f"  {model_name:<14} -{drop:.4f}")

    print(f"\n已保存：{TABLES_DIR / 'robustness_results.csv'}")
    print(f"已保存：{FIGURES_DIR / 'robustness_macro_f1.png'}")


if __name__ == "__main__":
    main()
