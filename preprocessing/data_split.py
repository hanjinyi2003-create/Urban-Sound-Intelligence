"""Member 1：按 UrbanSound8K 官方 Fold 划分训练 / 验证 / 测试集。

划分规则（严禁随机打乱）：
    Fold 1-8 训练，Fold 9 验证（用于选择超参数），Fold 10 测试（只在最终评价时使用）。

运行：
    python -m preprocessing.data_split

输出：
    data/processed/X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy
    data/processed/test_files.csv       测试集音频清单（噪声实验需要重新读取原始波形）
    data/processed/split_info.json      划分说明与样本统计
    results/tables/split_summary.csv
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config import (
    CLASS_NAMES,
    PROCESSED_DIR,
    TABLES_DIR,
    TEST_FOLD,
    TRAIN_FOLDS,
    VAL_FOLD,
    ensure_dirs,
)
from preprocessing.feature_extraction import load_features


def split_by_fold(data: dict) -> dict:
    """只按官方 Fold 切分，保持原始样本顺序，不做任何随机打乱。"""
    fold = data["fold"]

    train_mask = np.isin(fold, TRAIN_FOLDS)
    val_mask = fold == VAL_FOLD
    test_mask = fold == TEST_FOLD

    splits = {
        "X_train": data["X"][train_mask],
        "y_train": data["y"][train_mask],
        "X_val": data["X"][val_mask],
        "y_val": data["y"][val_mask],
        "X_test": data["X"][test_mask],
        "y_test": data["y"][test_mask],
    }

    test_frame = pd.DataFrame(
        {
            "slice_file_name": data["slice_file_name"][test_mask],
            "classID": data["y"][test_mask],
            "class": data["class_name"][test_mask],
            "fold": fold[test_mask],
            "path": data["path"][test_mask],
        }
    )
    splits["test_files"] = test_frame

    # 数据泄漏自检：三份数据的音频文件名不能有交集
    names_train = set(data["slice_file_name"][train_mask])
    names_val = set(data["slice_file_name"][val_mask])
    names_test = set(test_frame["slice_file_name"])
    assert not (names_train & names_val), "训练集与验证集存在重复文件"
    assert not (names_train & names_test), "训练集与测试集存在重复文件"
    assert not (names_val & names_test), "验证集与测试集存在重复文件"

    return splits


def save_split(splits: dict) -> None:
    for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        np.save(PROCESSED_DIR / f"{key}.npy", splits[key])
    splits["test_files"].to_csv(PROCESSED_DIR / "test_files.csv", index=False)

    info = {
        "split_policy": "UrbanSound8K official folds (no random shuffling)",
        "train_folds": list(TRAIN_FOLDS),
        "val_fold": VAL_FOLD,
        "test_fold": TEST_FOLD,
        "n_train": int(len(splits["y_train"])),
        "n_val": int(len(splits["y_val"])),
        "n_test": int(len(splits["y_test"])),
        "n_features": int(splits["X_train"].shape[1]),
        "class_names": list(CLASS_NAMES),
        "leakage_check": "passed (train/val/test share no audio file)",
    }
    with open(PROCESSED_DIR / "split_info.json", "w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)

    rows = []
    for split_name, labels in (
        ("train", splits["y_train"]),
        ("val", splits["y_val"]),
        ("test", splits["y_test"]),
    ):
        counts = np.bincount(labels, minlength=len(CLASS_NAMES))
        for class_id, class_name in enumerate(CLASS_NAMES):
            rows.append(
                {
                    "split": split_name,
                    "classID": class_id,
                    "class": class_name,
                    "count": int(counts[class_id]),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_DIR / "split_summary.csv", index=False)

    print("划分完成（官方 Fold，未随机打乱）：")
    print(f"  Train: Fold {list(TRAIN_FOLDS)} -> {info['n_train']} 条")
    print(f"  Val  : Fold {VAL_FOLD} -> {info['n_val']} 条")
    print(f"  Test : Fold {TEST_FOLD} -> {info['n_test']} 条")
    print("数据泄漏检查：通过（三份数据无重复音频文件）")
    print(summary.pivot(index="class", columns="split", values="count")
          .reindex(list(CLASS_NAMES)).to_string())


def load_split() -> dict:
    """加载已保存的划分数组，供所有模型与实验使用。"""
    required = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
    paths = {key: PROCESSED_DIR / f"{key}.npy" for key in required}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少划分文件：{missing}。请先运行 python -m preprocessing.data_split"
        )
    return {key: np.load(path) for key, path in paths.items()}


def main() -> None:
    ensure_dirs()
    data = load_features()
    splits = split_by_fold(data)
    save_split(splits)


if __name__ == "__main__":
    main()
