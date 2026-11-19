"""Member 2：SVM（RBF Kernel）模型定义与超参数搜索。

搜索策略：在 Fold 1-8 上训练，在 Fold 9 上按 Macro-F1 选择参数，
测试集 Fold 10 全程不参与选择，避免信息泄漏。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import RANDOM_STATE

DEFAULT_GRID = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", 0.001, 0.01, 0.1],
}


def build_svm(C: float = 10.0, gamma="scale", probability: bool = False) -> Pipeline:
    """StandardScaler + RBF-SVM。

    标准化必须放在 Pipeline 内，确保 fit 只发生在训练数据上。
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=C,
                    gamma=gamma,
                    kernel="rbf",
                    probability=probability,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def tune_svm(X_train, y_train, X_val, y_val, grid: dict = None) -> tuple:
    """网格搜索，返回最佳参数与全部候选的验证集成绩。"""
    grid = grid or DEFAULT_GRID
    records = []
    best = None

    total = len(grid["C"]) * len(grid["gamma"])
    index = 0
    for C in grid["C"]:
        for gamma in grid["gamma"]:
            index += 1
            started = time.perf_counter()
            model = build_svm(C=C, gamma=gamma, probability=False)
            model.fit(X_train, y_train)
            train_seconds = time.perf_counter() - started

            started = time.perf_counter()
            predictions = model.predict(X_val)
            predict_seconds = time.perf_counter() - started

            macro_f1 = float(
                f1_score(y_val, predictions, average="macro", zero_division=0)
            )
            record = {
                "C": C,
                "gamma": str(gamma),
                "val_macro_f1": round(macro_f1, 4),
                "train_seconds": round(train_seconds, 2),
                "predict_seconds": round(predict_seconds, 3),
            }
            records.append(record)
            print(
                f"[{index}/{total}] C={C:<6} gamma={str(gamma):<6} "
                f"val Macro-F1={macro_f1:.4f} ({train_seconds:.1f}s)"
            )
            if best is None or macro_f1 > best["val_macro_f1"]:
                best = record

    return best, pd.DataFrame(records)
