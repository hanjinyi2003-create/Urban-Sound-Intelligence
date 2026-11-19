"""项目统一配置：路径、音频参数、特征参数与官方 Fold 划分。

所有模块（preprocessing / models / experiments / demo）都从这里读取同一套常量，
保证特征维度、类别编码和训练测试划分完全一致。
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 路径
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "UrbanSound8K"
AUDIO_DIR = DATASET_DIR / "audio"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# 数据集解压后元数据可能位于 metadata/ 子目录，也可能直接在根目录
METADATA_CANDIDATES = (
    DATASET_DIR / "metadata" / "UrbanSound8K.csv",
    DATASET_DIR / "UrbanSound8K.csv",
)

# ---------------------------------------------------------------- 音频参数
SAMPLE_RATE = 22050
MONO = True
DURATION = 4.0          # UrbanSound8K 中所有片段均不超过 4 秒，统一补齐/截断到 4 秒

# ---------------------------------------------------------------- 特征参数
N_MFCC = 20
N_FFT = 2048
HOP_LENGTH = 512
N_FEATURES = N_MFCC * 2  # 20 个系数的均值 + 20 个系数的标准差

# ---------------------------------------------------------------- 官方 Fold 划分
# UrbanSound8K 已按原始录音来源切分为 10 折，必须整折使用，禁止随机打乱，
# 否则同一段原始录音的切片会同时出现在训练集和测试集，造成数据泄漏。
TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
VAL_FOLD = 9
TEST_FOLD = 10

# ---------------------------------------------------------------- 类别
CLASS_NAMES = (
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
)

# ---------------------------------------------------------------- 实验参数
RANDOM_STATE = 42
NOISE_SNR_LEVELS = (20, 10)   # P0：必须完成 Clean / 20 dB / 10 dB
OPTIONAL_SNR_LEVELS = (0,)    # P1：可选扩展


def metadata_path() -> Path:
    """返回存在的 UrbanSound8K 元数据 CSV 路径。"""
    for candidate in METADATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "未找到 UrbanSound8K.csv，请先下载数据集并解压到 "
        f"{DATASET_DIR}（应包含 audio/ 与 metadata/ 两个目录）。"
    )


def ensure_dirs() -> None:
    """创建所有输出目录，重复调用安全。"""
    for directory in (PROCESSED_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def feature_names() -> list:
    """40 维特征列名，顺序与 feature_extraction.mfcc_statistics 一致。"""
    names = [f"mfcc{i:02d}_mean" for i in range(N_MFCC)]
    names += [f"mfcc{i:02d}_std" for i in range(N_MFCC)]
    return names
