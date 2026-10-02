# CCS Reference

[English](README.md)

本仓库发布医生复核标签合入后的 CCS 最终开发路线，并沿用 `supdub/vision-ai` 研究原型的组织方式，将配置、可安装源码、测试、聚合数据、结果、表格、图片及论文材料分开保存。

![五随机种子内部验证结果](figures/metrics_overview.png)

## 最终版本

最终路线包括冻结的 ConvNeXt-Tiny 图像编码器、正则化前囊膜浑浊四分类头、WTW 校准囊口面积头、确定性收缩分级及最终 CCS 诊断规则。完成重复登记清理及 23 张边界图像盲法复核后，患者级无交叉开发队列为训练集 777 张、内部验证集 177 张。

五次训练使用相同固定划分，随机种子为 `20260927`、`20260928`、`20260929`、`20260930`、`20261001`。

| 任务 | 准确率 | 平衡准确率 | Macro-F1 | Macro ROC-AUC |
|---|---:|---:|---:|---:|
| 前囊膜浑浊四分类 | 88.14 +/- 1.87% | 85.22 +/- 2.12% | 0.8554 +/- 0.0214 | 0.9743 +/- 0.0047 |
| 囊口收缩三分类 | 96.38 +/- 0.31% | 76.82 +/- 2.09% | 0.7954 +/- 0.0190 | 未计算 |
| 最终 CCS 诊断三分类 | 97.97 +/- 0.31% | 82.82 +/- 0.11% | 0.7831 +/- 0.0127 | 未计算 |

数值为五个训练种子的均值 +/- 样本标准差。完整机器可读结果见 [`results/multiseed_mean.json`](results/multiseed_mean.json)。

## 仓库结构

```text
configs/                 最终路线冻结配置
data/aggregate/          不含个体身份的队列与划分统计
docs/                    Methods 证据记录与聚合结果报告
figures/                 论文用聚合图片
results/                 五随机种子指标及逐次训练历史
src/ccs_model/           数据准备、训练和评估源码
tables/                  论文用 CSV 表格
tests/                   架构、指标和阈值测试
```

## 安装与测试

建议使用 Python 3.12。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest
```

最终配置与全部随机种子见 [`configs/ccs_reviewed_multiseed_v1.yaml`](configs/ccs_reviewed_multiseed_v1.yaml)。临床输入和模型权重按照数据治理要求单独管理，公开边界及申请方式见 [`DATA_ACCESS.md`](DATA_ACCESS.md)。

## 局限性

本版本报告 177 张图像的开发阶段内部验证，锁定独立测试集未访问。复核边界病例被固定分入训练集，因此最终划分不同于早期开发划分。收缩少数类别及 CCS 阳性类别在验证集中的样本量较少。随机种子间标准差反映优化稳定性，不等同于患者级 95% 置信区间。
