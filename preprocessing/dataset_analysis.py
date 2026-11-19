"""Member 1：数据集检查与类别分布统计。

运行：
    python -m preprocessing.dataset_analysis

输出：
    results/figures/class_distribution.png
    results/tables/class_distribution.csv
    results/tables/fold_class_counts.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    AUDIO_DIR,
    CLASS_NAMES,
    FIGURES_DIR,
    TABLES_DIR,
    TEST_FOLD,
    TRAIN_FOLDS,
    VAL_FOLD,
    ensure_dirs,
    metadata_path,
)


def load_metadata() -> pd.DataFrame:
    """读取官方元数据，并补充每条音频的绝对路径与文件存在性。"""
    meta = pd.read_csv(metadata_path())
    meta["audio_path"] = meta.apply(
        lambda row: AUDIO_DIR / f"fold{int(row['fold'])}" / row["slice_file_name"],
        axis=1,
    )
    meta["exists"] = meta["audio_path"].apply(lambda path: Path(path).is_file())
    return meta


def analyze(meta: pd.DataFrame) -> None:
    total = len(meta)
    missing = meta.loc[~meta["exists"]]
    print(f"元数据记录数：{total}")
    print(f"音频文件缺失数：{len(missing)}")
    if not missing.empty:
        print(missing[["slice_file_name", "fold"]].head(20).to_string(index=False))

    counts = (
        meta.groupby(["classID", "class"])
        .size()
        .reset_index(name="count")
        .sort_values("classID")
    )
    counts["ratio"] = (counts["count"] / total).round(4)
    print("\n各类别样本数量：")
    print(counts.to_string(index=False))
    print(f"\n最多类别 {counts['count'].max()} 条，最少类别 {counts['count'].min()} 条，"
          f"不平衡比 {counts['count'].max() / counts['count'].min():.2f}")

    fold_table = (
        meta.pivot_table(
            index="fold", columns="class", values="slice_file_name", aggfunc="count", fill_value=0
        )
        .reindex(columns=list(CLASS_NAMES), fill_value=0)
    )
    fold_table["total"] = fold_table.sum(axis=1)
    print("\n各 Fold × 类别 数量：")
    print(fold_table.to_string())

    # 官方划分核对
    used = fold_table.loc[list(TRAIN_FOLDS) + [VAL_FOLD, TEST_FOLD], "total"].sum()
    print(
        f"\n划分方式：Fold {list(TRAIN_FOLDS)} 训练 / Fold {VAL_FOLD} 验证 / "
        f"Fold {TEST_FOLD} 测试，覆盖 {used}/{total} 条样本"
    )

    counts.to_csv(TABLES_DIR / "class_distribution.csv", index=False)
    fold_table.to_csv(TABLES_DIR / "fold_class_counts.csv")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts["class"], counts["count"], color="#4C72B0")
    ax.set_title("UrbanSound8K Class Distribution (n = %d)" % total)
    ax.set_ylabel("Number of clips")
    ax.set_xlabel("Class")
    ax.tick_params(axis="x", rotation=30)
    for x, value in enumerate(counts["count"]):
        ax.text(x, value + 8, str(value), ha="center", fontsize=9)
    ax.set_ylim(0, counts["count"].max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)
    print(f"\n已保存类别分布图：{FIGURES_DIR / 'class_distribution.png'}")


def main() -> None:
    ensure_dirs()
    meta = load_metadata()
    analyze(meta)
    print("\n提示：类别并不完全均衡，因此后续评价以 Macro-F1 为主要指标。")


if __name__ == "__main__":
    main()
