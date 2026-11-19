"""Member 3：Random Forest Feature Importance 分析与作图。

需要说明的局限：
    - 重要性反映的是「该特征降低不纯度的程度」，不等价于因果贡献；
    - 相关性强的 MFCC 特征会相互分摊重要性，单看排名可能低估整体频带的作用；
    - 训练集内的划分依然沿用官方 Fold，特征重要性只能在训练范围内解释。

运行：
    python -m models.feature_importance
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, MODELS_DIR, TABLES_DIR, feature_names


def main() -> None:
    model_path = MODELS_DIR / "rf.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} 不存在，请先运行 python -m models.rf_training")

    model = joblib.load(model_path)
    names = feature_names()
    importances = model.feature_importances_

    table = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    table["rank"] = table.index + 1
    table.to_csv(TABLES_DIR / "rf_feature_importance.csv", index=False)

    mean_importance = {
        "MFCC mean (20 dims)": float(table[table.feature.str.endswith("mean")]["importance"].sum()),
        "MFCC std (20 dims)": float(table[table.feature.str.endswith("std")]["importance"].sum()),
    }

    print("Top 10 重要特征：")
    print(table.head(10).to_string(index=False))
    print("\n按统计量汇总：")
    for key, value in mean_importance.items():
        print(f"  {key:<22} {value:.4f}")

    top = table.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#DD8452" if name.endswith("std") else "#4C72B0" for name in top["feature"]]
    ax.barh(top["feature"], top["importance"], color=colors)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest - Top 20 MFCC Features")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rf_feature_importance.png", dpi=150)
    plt.close(fig)

    print(f"\n已保存：{TABLES_DIR / 'rf_feature_importance.csv'}")
    print(f"已保存：{FIGURES_DIR / 'rf_feature_importance.png'}")


if __name__ == "__main__":
    main()
