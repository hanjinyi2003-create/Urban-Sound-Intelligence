"""Member 2：读取已保存的 SVM 模型，在测试集上复核评价。

运行：
    python -m models.svm_evaluation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from config import MODELS_DIR
from experiments.evaluation import (
    evaluate_model,
    format_metrics,
    plot_confusion_matrix,
    save_metrics,
)
from preprocessing.data_split import load_split


def main() -> None:
    model_path = MODELS_DIR / "svm.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} 不存在，请先运行 python -m models.svm_training")

    model = joblib.load(model_path)
    split = load_split()
    metrics = evaluate_model(model, split["X_test"], split["y_test"])
    metrics["model"] = "SVM (RBF)"

    save_metrics(metrics, "svm_metrics.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        "SVM (RBF) - Confusion Matrix (Fold 10)",
        "svm_confusion_matrix.png",
    )

    report = metrics["classification_report"]
    print(format_metrics(metrics))
    print("\n各类别 F1：")
    for class_name, values in report.items():
        if isinstance(values, dict) and "f1-score" in values and class_name not in (
            "macro avg",
            "weighted avg",
        ):
            print(f"  {class_name:<20} F1={values['f1-score']:.3f}")


if __name__ == "__main__":
    main()
