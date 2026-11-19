"""Member 3：Random Forest 模型定义与超参数搜索。

搜索策略与 SVM 完全一致：Fold 1-8 训练，Fold 9 验证选参，Fold 10 只做最终评价。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from config import RANDOM_STATE

DEFAULT_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 20, 40],
}


def build_rf(
    n_estimators: int = 300,
    max_depth=None,
    min_samples_leaf: int = 1,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def tune_rf(X_train, y_train, X_val, y_val, grid: dict = None) -> tuple:
    """网格搜索，返回最佳参数与全部候选的验证集成绩。"""
    grid = grid or DEFAULT_GRID
    records = []
    best = None

    total = len(grid["n_estimators"]) * len(grid["max_depth"])
    index = 0
    for n_estimators in grid["n_estimators"]:
        for max_depth in grid["max_depth"]:
            index += 1
            started = time.perf_counter()
            model = build_rf(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X_train, y_train)
            train_seconds = time.perf_counter() - started

            started = time.perf_counter()
            predictions = model.predict(X_val)
            predict_seconds = time.perf_counter() - started

            macro_f1 = float(
                f1_score(y_val, predictions, average="macro", zero_division=0)
            )
            record = {
                "n_estimators": n_estimators,
                "max_depth": str(max_depth),
                "val_macro_f1": round(macro_f1, 4),
                "train_seconds": round(train_seconds, 2),
                "predict_seconds": round(predict_seconds, 3),
            }
            records.append(record)
            print(
                f"[{index}/{total}] n_estimators={n_estimators:<4} "
                f"max_depth={str(max_depth):<5} val Macro-F1={macro_f1:.4f} "
                f"({train_seconds:.1f}s)"
            )
            if best is None or macro_f1 > best["val_macro_f1"]:
                best = record

    return best, pd.DataFrame(records)
