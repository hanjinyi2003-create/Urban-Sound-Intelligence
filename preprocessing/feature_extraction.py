"""Member 1：MFCC 特征提取，产出三个模型共用的统一特征文件。

特征定义：20 个 MFCC 系数的均值 + 20 个 MFCC 系数的标准差 = 40 维。

运行：
    python -m preprocessing.feature_extraction

输出：
    data/processed/features.npz        全部样本特征（X, y, fold, 文件名, 类别名, 路径）
    results/tables/feature_extraction_summary.csv
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    AUDIO_DIR,
    DURATION,
    HOP_LENGTH,
    N_FFT,
    N_FEATURES,
    N_MFCC,
    PROCESSED_DIR,
    SAMPLE_RATE,
    TABLES_DIR,
    ensure_dirs,
    metadata_path,
)
from preprocessing.audio_processing import load_checked


def mfcc_statistics(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """计算单条波形的 40 维 MFCC 统计特征。

    该函数同时被 Demo 复用，必须保证与训练时完全一致。
    """
    mfcc = librosa.feature.mfcc(
        y=np.asarray(y, dtype=np.float32),
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    return features.astype(np.float32)


def extract_features(paths, labels=None, folds=None, names=None):
    """批量提取特征，跳过读取失败的文件并记录警告。"""
    features, kept_labels, kept_folds, kept_names, kept_paths = [], [], [], [], []
    failed = []

    for index, path in enumerate(tqdm(paths, desc="Extracting MFCC", ncols=80)):
        try:
            y, sr = load_checked(path)
            features.append(mfcc_statistics(y, sr))
        except Exception as error:  # 坏文件不应中断整个流程
            failed.append((str(path), str(error)))
            continue
        if labels is not None:
            kept_labels.append(labels[index])
        if folds is not None:
            kept_folds.append(folds[index])
        if names is not None:
            kept_names.append(names[index])
        kept_paths.append(str(path))

    if failed:
        print(f"警告：{len(failed)} 个文件提取失败，已跳过。示例：{failed[:3]}")

    X = np.vstack(features) if features else np.empty((0, N_FEATURES), dtype=np.float32)
    return X, kept_labels, kept_folds, kept_names, kept_paths, failed


def build_dataset() -> dict:
    """从原始音频构建全量特征矩阵并保存。"""
    meta = pd.read_csv(metadata_path())
    paths = [
        AUDIO_DIR / f"fold{int(row.fold)}" / row.slice_file_name
        for row in meta.itertuples()
    ]
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} 个音频文件不存在，示例：{missing[:3]}。请检查数据集是否完整解压。"
        )

    started = time.time()
    X, labels, folds, names, kept_paths, failed = extract_features(
        paths,
        labels=meta["classID"].astype(int).tolist(),
        folds=meta["fold"].astype(int).tolist(),
        names=meta["slice_file_name"].tolist(),
    )
    elapsed = time.time() - started

    class_names = meta.set_index("slice_file_name").loc[names, "class"].tolist()
    payload = {
        "X": X,
        "y": np.asarray(labels, dtype=np.int64),
        "fold": np.asarray(folds, dtype=np.int64),
        "slice_file_name": np.asarray(names, dtype=str),
        "class_name": np.asarray(class_names, dtype=str),
        "path": np.asarray(kept_paths, dtype=str),
    }
    np.savez_compressed(PROCESSED_DIR / "features.npz", **payload)

    config_record = {
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION,
        "n_mfcc": N_MFCC,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_features": N_FEATURES,
        "feature_order": "20 x MFCC mean, then 20 x MFCC std",
        "samples": int(X.shape[0]),
        "failed_files": len(failed),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(PROCESSED_DIR / "feature_config.json", "w", encoding="utf-8") as handle:
        json.dump(config_record, handle, ensure_ascii=False, indent=2)

    summary = pd.DataFrame(
        {
            "feature": [f"mfcc{i:02d}_mean" for i in range(N_MFCC)]
            + [f"mfcc{i:02d}_std" for i in range(N_MFCC)],
            "mean": X.mean(axis=0).round(4),
            "std": X.std(axis=0).round(4),
        }
    )
    summary.to_csv(TABLES_DIR / "feature_summary.csv", index=False)

    print(f"\n特征矩阵：{X.shape}，用时 {elapsed / 60:.1f} 分钟")
    print(f"已保存：{PROCESSED_DIR / 'features.npz'}")
    print(f"失败文件：{len(failed)}")
    return payload


def load_features() -> dict:
    """读取已保存的特征文件。"""
    path = PROCESSED_DIR / "features.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在，请先运行 python -m preprocessing.feature_extraction"
        )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main() -> None:
    ensure_dirs()
    build_dataset()


if __name__ == "__main__":
    main()
