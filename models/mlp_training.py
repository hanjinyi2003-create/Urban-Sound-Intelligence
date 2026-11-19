"""Member 4：MLP 训练主流程。

运行：
    python -m models.mlp_training
"""

import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np

from config import MODELS_DIR, TABLES_DIR, ensure_dirs
from experiments.evaluation import (
    evaluate_model,
    format_metrics,
    plot_confusion_matrix,
    save_metrics,
)
from models.mlp_model import DEFAULT_GRID, build_mlp, tune_mlp
from preprocessing.data_split import load_split

MODEL_PATH = MODELS_DIR / "mlp.joblib"
PARAMS_PATH = MODELS_DIR / "mlp_params.json"


def main() -> None:
    ensure_dirs()
    split = load_split()
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    X_test, y_test = split["X_test"], split["y_test"]
    print(f"Train {X_train.shape} | Val {X_val.shape} | Test {X_test.shape}")

    print("\n[1/3] 在验证集（Fold 9）上搜索超参数")
    best, tuning_table = tune_mlp(X_train, y_train, X_val, y_val, DEFAULT_GRID)
    tuning_table.sort_values("val_macro_f1", ascending=False).to_csv(
        TABLES_DIR / "mlp_tuning.csv", index=False
    )
    print(
        f"最佳参数：hidden_layer_sizes={best['hidden_layer_sizes']}, "
        f"alpha={best['alpha']}，Val Macro-F1={best['val_macro_f1']}"
    )

    print("\n[2/3] 使用 Train+Val 重新拟合最终模型")
    hidden_layer_sizes = ast.literal_eval(best["hidden_layer_sizes"])

    started = time.perf_counter()
    final_model = build_mlp(
        hidden_layer_sizes=hidden_layer_sizes, alpha=best["alpha"], max_iter=500
    )
    final_model.fit(
        np.vstack([X_train, X_val]), np.concatenate([y_train, y_val])
    )
    training_seconds = time.perf_counter() - started

    joblib.dump(final_model, MODEL_PATH)
    with open(PARAMS_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": "MLPClassifier",
                "best_params": {
                    "hidden_layer_sizes": list(hidden_layer_sizes),
                    "alpha": best["alpha"],
                },
                "selection_set": "Fold 9 validation",
                "final_fit_set": "Fold 1-8 + Fold 9",
                "train_samples": int(len(y_train) + len(y_val)),
                "n_features": int(X_train.shape[1]),
                "epochs_run": int(final_model.named_steps["mlp"].n_iter_),
                "training_seconds": round(training_seconds, 2),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"模型已保存：{MODEL_PATH}（训练耗时 {training_seconds:.1f}s）")

    print("\n[3/3] 在测试集（Fold 10）上评价")
    metrics = evaluate_model(final_model, X_test, y_test)
    metrics["model"] = "MLP"
    metrics["train_seconds"] = round(training_seconds, 2)
    save_metrics(metrics, "mlp_metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        "MLP - Confusion Matrix (Fold 10)",
        "mlp_confusion_matrix.png",
    )
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
