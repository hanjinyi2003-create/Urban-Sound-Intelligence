"""Member 4：MLP（Scikit-learn MLPClassifier）模型定义与超参数搜索。

网络输入为 40 维 MFCC 统计特征，标准化放在 Pipeline 内，
确保 Demo 推理时使用与训练时完全一致的缩放参数。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE

DEFAULT_GRID = {
    "hidden_layer_sizes": [(128,), (256, 128), (512, 256)],
    "alpha": [1e-4, 1e-3],
}


def build_mlp(
    hidden_layer_sizes=(256, 128),
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    max_iter: int = 300,
) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    solver="adam",
                    batch_size=256,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=15,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def tune_mlp(X_train, y_train, X_val, y_val, grid: dict = None) -> tuple:
    """网格搜索，返回最佳参数与全部候选的验证集成绩。

    注意：MLP 内部的 early_stopping 只在传入的训练数据内部再切一份，
    不涉及验证集 Fold 9 与测试集 Fold 10，不会造成跨折泄漏。
    """
    grid = grid or DEFAULT_GRID
    records = []
    best = None

    total = len(grid["hidden_layer_sizes"]) * len(grid["alpha"])
    index = 0
    for hidden_layer_sizes in grid["hidden_layer_sizes"]:
        for alpha in grid["alpha"]:
            index += 1
            started = time.perf_counter()
            model = build_mlp(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)
            model.fit(X_train, y_train)
            train_seconds = time.perf_counter() - started

            started = time.perf_counter()
            predictions = model.predict(X_val)
            predict_seconds = time.perf_counter() - started

            macro_f1 = float(
                f1_score(y_val, predictions, average="macro", zero_division=0)
            )
            record = {
                "hidden_layer_sizes": str(hidden_layer_sizes),
                "alpha": alpha,
                "val_macro_f1": round(macro_f1, 4),
                "train_seconds": round(train_seconds, 2),
                "predict_seconds": round(predict_seconds, 3),
            }
            records.append(record)
            print(
                f"[{index}/{total}] hidden={str(hidden_layer_sizes):<12} "
                f"alpha={alpha:<6} val Macro-F1={macro_f1:.4f} ({train_seconds:.1f}s)"
            )
            if best is None or macro_f1 > best["val_macro_f1"]:
                best = record

    return best, pd.DataFrame(records)
