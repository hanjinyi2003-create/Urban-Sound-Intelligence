# Urban Sound Intelligence

CDS701 Group Project —— **城市环境声音智能识别与复杂噪声环境下的鲁棒性研究**

基于 UrbanSound8K 的十分类城市环境声音识别，提取 MFCC 统计特征，比较 **SVM / Random Forest / MLP**，
并研究白噪声条件下的鲁棒性、PCA 降维的影响，最终提供 Streamlit 上传演示。

| 研究问题 | 对应模块 |
|---|---|
| RQ1 哪个模型在城市声音分类上表现最好 | `models/svm_training.py`、`models/rf_training.py`、`models/mlp_training.py` |
| RQ2 背景噪声如何影响各模型性能 | `experiments/noise_generator.py`、`experiments/robustness_test.py` |
| RQ3 PCA 能否在保持性能的同时降维 | `experiments/pca_experiment.py` |

---

## 1. 数据划分（不可更改的核心约束）

UrbanSound8K 的 10 折是**按原始录音来源**预先划分的（同一段录音的切片只会出现在同一折）。
因此本项目严格使用官方 Fold，**任何随机打乱（`train_test_split` 不带 fold 约束、`shuffle=True` 切分）都会造成数据泄漏**。

| 用途 | Fold | 说明 |
|---|---|---|
| 训练 | 1–8 | 拟合模型参数 |
| 验证 | 9 | 选择超参数（C/gamma、树参数、MLP 结构） |
| 测试 | 10 | 只在最终评价时使用 |

- 标准化 `StandardScaler`、`PCA` 一律只用训练数据 `fit`，再 `transform` 验证/测试数据（已封装在 Pipeline 中）。
- 超参数选择只看 Fold 9 的 Macro-F1，测试集不参与任何选择。
- `preprocessing/data_split.py` 内置了泄漏自检：若三份数据出现同名音频文件会直接报错。

## 2. 环境准备

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> 推荐 Python 3.10–3.12。若想免安装，可直接使用 `Urban_Sound_Intelligence_Colab.ipynb`。

## 3. 数据集

**主数据集 UrbanSound8K**（8,732 条，≤4 秒，10 类，约 6 GB）
<https://zenodo.org/records/1203745>

下载 `UrbanSound8K.tar.gz` 后解压，使目录结构为：

```
data/UrbanSound8K/
├── audio/
│   ├── fold1/ ... fold10/
└── metadata/UrbanSound8K.csv
```

**可选对照数据集**
- ESC-50（2,000 条 5 秒、50 类，约 600 MB）：<https://github.com/karolpiczak/ESC-50>，适合先验证流程。
- SONYC-UST（纽约城市传感器多标签标注，约 13 GB）：<https://zenodo.org/records/3693077>，扩展方向，不在本次 P0 范围内。

> 原始音频与提取出的特征文件已在 `.gitignore` 中排除，不进入版本库。

## 4. 运行顺序

```bash
# Member 1：数据检查、MFCC 特征、官方 Fold 划分
python -m preprocessing.dataset_analysis
python -m preprocessing.feature_extraction      # ~40 分钟（CPU）
python -m preprocessing.data_split

# Member 2：SVM 调参 + 训练 + 评价，PCA 实验
python -m models.svm_training
python -m models.svm_evaluation
python -m experiments.pca_experiment

# Member 3：Random Forest 调参 + 训练 + 评价，特征重要性
python -m models.rf_training
python -m models.rf_evaluation
python -m models.feature_importance

# Member 4：MLP 训练，含噪测试集，鲁棒性实验
python -m models.mlp_training
python -m experiments.noise_generator --snr 20 10     # 可选扩展：再加 0
python -m experiments.robustness_test

# 汇总三模型对比表与图
python -m experiments.evaluation

# Member 5：Demo
streamlit run demo/app.py
```

## 5. 特征定义

- 采样率 22050 Hz，单声道，统一 4 秒（不足补零、超出截断）。
- 每帧 MFCC 20 个系数 → 对时间轴取均值与标准差 → **40 维**特征向量。
- 特征顺序：`mfcc00_mean … mfcc19_mean, mfcc00_std … mfcc19_std`（见 `config.feature_names()`）。

## 6. 输出文件

| 路径 | 内容 |
|---|---|
| `data/processed/features.npz` | 全部样本的 40 维特征、标签、fold、文件路径 |
| `data/processed/X_*.npy, y_*.npy` | 官方 Fold 划分后的训练/验证/测试矩阵 |
| `data/processed/split_info.json` | 划分说明与样本数（含泄漏检查结论） |
| `data/processed/test_files.csv` | Fold 10 音频清单，噪声实验据此重新读取波形 |
| `saved_models/*.joblib` | 三个训练完成的模型（Pipeline 内含标准化） |
| `saved_models/*_params.json` | 最佳超参数、训练耗时等可复核记录 |
| `results/metrics/*.json` | 各模型完整指标与分类报告 |
| `results/figures/*.png` | 类别分布、混淆矩阵、PCA 对比、特征重要性、鲁棒性曲线、模型对比 |
| `results/tables/*.csv` | 调参记录、划分统计、PCA 结果、鲁棒性结果、模型对比表 |

`results/` 中的内容必须由实际运行生成。仓库中不预置任何编造的准确率或 Macro-F1。

## 7. 噪声实验口径

- 噪声添加在**原始波形**上，再重新提取 MFCC，不对特征直接加噪。
- SNR 定义：`SNR_dB = 10·log10(P_signal / P_noise)`。
- 每条音频的噪声随机种子由 `文件名 + SNR` 决定（`zlib.crc32`），因此三个模型、多次运行看到的含噪信号完全一致。
- 三个模型都在干净数据（Fold 1–9）上训练，只在含噪测试集上评价，研究的是「未做噪声增强时的鲁棒性」。
- 实测 SNR 会写入 `results/tables/noise_conditions.csv` 以便核验。

## 8. Streamlit Demo

```bash
streamlit run demo/app.py
```

功能：上传 WAV → 播放音频 → 波形与 Mel 频谱 → 选择 SVM / RF / MLP → 显示预测类别、置信度、Top-5 概率。
现场演示建议准备录屏或截图作为备用方案。

## 9. Colab

打开 `Urban_Sound_Intelligence_Colab.ipynb`，按顺序执行即可完成「下载数据 → 特征 → 训练 → 评价 → 鲁棒性 → 打包模型」。
Notebook 顶部可填写仓库地址自动 `git clone`，也可以直接把本项目文件夹上传到 `/content` 后运行。

## 10. 成员分工

| 成员 | 职责 | 主要文件 |
|---|---|---|
| Member 1 | Dataset & Audio Processing | `preprocessing/*` |
| Member 2 | SVM & PCA | `models/svm_*.py`、`experiments/pca_experiment.py` |
| Member 3 | Random Forest | `models/rf_*.py`、`models/feature_importance.py` |
| Member 4 | MLP & Noise Robustness | `models/mlp_*.py`、`experiments/noise_generator.py`、`experiments/robustness_test.py` |
| Member 5 | Integration & Demo | `demo/*`、`experiments/evaluation.py` |

## 11. 引用

> Salamon, J., Jacoby, C., & Bello, J. P. (2014). **A Dataset and Taxonomy for Urban Sound Research.**
> In *Proceedings of the 22nd ACM International Conference on Multimedia* (pp. 1041–1044).
> Dataset: <https://urbansounddataset.weebly.com/urbansound8k.html> · <https://zenodo.org/records/1203745>

MFCC 与重采样使用 librosa；模型与评价使用 scikit-learn。如这些库的文档或示例代码被直接借用，需在 PPT 中注明。

## 12. 生成式 AI 使用披露

本项目在代码框架搭建、文档整理与调试辅助中使用了生成式 AI 工具（Trae IDE 内置助手）。
所有实验数值均需由团队成员在本地或 Colab 实际运行产生，AI 不参与、也不提供任何结果数值。
提交的 PPT 需按课程要求披露该使用情况与具体用途。

## 13. 已知局限

- MFCC 统计特征丢弃了时间结构，对时长/顺序敏感的声音区分能力有限。
- 训练集类别不均衡，Macro-F1 比 Accuracy 更能反映真实表现。
- 白噪声不等于真实城市背景噪声，结论只说明「对白噪声的敏感性」。
- 仅用 Fold 10 做最终测试，单次划分存在一定方差；如需更稳的结论可补充 10 折交叉验证（仍必须整折轮换）。
