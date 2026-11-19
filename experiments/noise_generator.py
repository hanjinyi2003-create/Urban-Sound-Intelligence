"""Member 4：白噪声生成与含噪测试特征构建。

关键约定：
    - 噪声加在原始音频波形上，再重新提取 MFCC；不对 MFCC 特征直接加噪。
    - 每条测试音频的噪声种子由「文件名 + SNR」确定，因此三个模型、多次运行
      看到的含噪信号完全一致，保证公平比较。
    - SNR 定义：snr_db = 10 * log10(P_signal / P_noise)

运行：
    python -m experiments.noise_generator                # 生成 20 dB 与 10 dB
    python -m experiments.noise_generator --snr 20 10 0  # 追加 0 dB 扩展

输出：
    data/processed/noise/test_{snr}dB.npz   含噪测试特征 (X, y)
    results/tables/noise_conditions.csv     各噪声条件下的实际 SNR 核验值
"""

import argparse
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import NOISE_SNR_LEVELS, PROCESSED_DIR, TABLES_DIR, ensure_dirs
from preprocessing.audio_processing import load_checked
from preprocessing.feature_extraction import mfcc_statistics

NOISE_DIR = PROCESSED_DIR / "noise"


def snr_seed(file_name: str, snr_db: float) -> int:
    """稳定的逐文件种子，保证含噪信号可复现。"""
    return zlib.crc32(f"{file_name}|{snr_db:g}".encode("utf-8"))


def add_white_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """按指定 SNR 在波形叠加高斯白噪声。"""
    y = np.asarray(y, dtype=np.float32)
    signal_power = float(np.mean(np.square(y)))
    if signal_power <= 0:
        return y.copy()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=y.shape).astype(np.float32)
    return (y + noise).astype(np.float32)


def measure_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """核验实际达成的 SNR（dB）。"""
    noise = noisy - clean
    clean_power = float(np.mean(np.square(clean)))
    noise_power = float(np.mean(np.square(noise)))
    if noise_power <= 0:
        return float("inf")
    return 10.0 * np.log10(clean_power / noise_power)


def build_noisy_test_features(snr_db: float) -> dict:
    """对测试集（Fold 10）每条音频加噪并重新提取 MFCC。"""
    manifest = PROCESSED_DIR / "test_files.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} 不存在，请先运行 python -m preprocessing.data_split"
        )

    test_files = pd.read_csv(manifest)
    NOISE_DIR.mkdir(parents=True, exist_ok=True)

    features, labels, snr_checks = [], [], []
    for row in tqdm(test_files.itertuples(), total=len(test_files),
                    desc=f"SNR {snr_db:g} dB", ncols=80):
        clean, sr = load_checked(row.path)
        rng = np.random.default_rng(snr_seed(row.slice_file_name, snr_db))
        noisy = add_white_noise(clean, snr_db, rng)

        features.append(mfcc_statistics(noisy, sr))
        labels.append(int(row.classID))
        if len(snr_checks) < 50:                     # 前 50 条采样核验，避免额外开销
            snr_checks.append(measure_snr(clean, noisy))

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    output = NOISE_DIR / f"test_{snr_db:g}dB.npz"
    np.savez_compressed(output, X=X, y=y, snr_db=np.float64(snr_db))

    achieved = float(np.mean(snr_checks))
    print(
        f"已生成 {output.name}：{X.shape}，"
        f"目标 SNR {snr_db:g} dB，实测平均 {achieved:.2f} dB"
    )
    return {
        "condition": f"{snr_db:g} dB",
        "target_snr_db": float(snr_db),
        "measured_snr_db": round(achieved, 2),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
    }


def load_noisy_test_features(snr_db: float) -> dict:
    path = NOISE_DIR / f"test_{snr_db:g}dB.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在，请先运行 python -m experiments.noise_generator --snr {snr_db:g}"
        )
    with np.load(path, allow_pickle=False) as data:
        return {"X": data["X"], "y": data["y"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成含噪测试集 MFCC 特征")
    parser.add_argument(
        "--snr",
        nargs="+",
        type=float,
        default=list(NOISE_SNR_LEVELS),
        help="噪声条件（dB），默认 20 10",
    )
    args = parser.parse_args()

    ensure_dirs()
    records = [build_noisy_test_features(snr) for snr in args.snr]
    pd.DataFrame(records).to_csv(TABLES_DIR / "noise_conditions.csv", index=False)
    print(f"\n已保存：{TABLES_DIR / 'noise_conditions.csv'}")


if __name__ == "__main__":
    main()
