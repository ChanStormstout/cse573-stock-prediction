# Four-hour finite new-information experiment

All existing periods are exposed exploratory backtests. The 60% objective is not a guaranteed outcome. No new independent holdout or human extraction review is claimed.

## BA, separately by stock and period

| Method | AAPL outer monthly | AMZN outer monthly | AAPL development | AMZN development | AAPL later | AMZN later |
|---|---:|---:|---:|---:|---:|---:|
| F0 | 50.93% | 60.75% | 50.46% | 48.33% | 51.89% | 49.79% |
| R1 | 52.83% | 60.73% | 56.21% | 51.11% | 53.87% | 50.85% |
| F1 | 47.86% | 61.20% | 57.94% | 55.66% | 51.74% | 54.36% |
| F2 | 50.41% | 56.08% | 54.13% | 46.29% | 56.71% | 54.27% |
| F6 | 52.25% | 60.48% | 50.35% | 52.50% | 49.39% | 51.16% |
| F1_new | 47.86% | 60.49% | 57.94% | 56.68% | 51.74% | 53.82% |
| F2_new | 50.93% | 56.38% | 55.12% | 49.17% | 55.14% | 56.79% |
| H0 | 56.20% | 64.95% | 54.61% | 51.21% | 52.82% | 50.29% |
| H1 | 56.58% | 60.76% | 55.86% | 55.01% | 52.92% | 48.72% |
| H2 | 54.15% | 62.16% | 53.40% | 54.55% | 51.77% | 48.67% |
| P_cross_hgb | 59.53% | 42.16% | 51.42% | 49.54% | 49.84% | 50.74% |
| P_cross_lr | 56.88% | 57.06% | 50.56% | 56.77% | 54.84% | 49.20% |
| P_cross_tabpfn | 50.36% | 49.87% | 53.17% | 52.32% | 50.17% | 54.35% |
| P_own_hgb | 60.30% | 44.05% | 56.09% | 54.92% | 46.00% | 51.38% |
| P_own_lr | 55.04% | 60.87% | 54.23% | 53.15% | 53.19% | 50.26% |
| P_own_tabpfn | 54.15% | 53.77% | 55.25% | 57.42% | 45.65% | 52.28% |
| T_A1 | 52.47% | 62.89% | 51.90% | 50.93% | 54.34% | 46.65% |
| T_A1_interaction | 43.92% | 58.31% | 54.44% | 50.65% | 54.53% | 49.01% |
| T_mix | 51.15% | 59.02% | 50.43% | 52.32% | 50.20% | 51.72% |
| T_original | 52.37% | 60.14% | 49.70% | 51.67% | 52.12% | 48.36% |
| T_original_interaction | 50.54% | 62.17% | 48.07% | 48.24% | 50.16% | 47.83% |

## Meaning and limitations

P: own versus cross-stock past prices, with LR, shallow boosting and fixed synthetic TabPFN. T: same protected target-company text and CLS pooling, original versus January–February adapted A1; LR versus rank-two interaction. H: original four-hour news, separate preceding-session news, and already-realized reaction. F1_new/F2_new select C on actual issued probabilities, including new price fallback; they are new controls, not corrections to historical scores.

Outer means weight June, July and August equally. Pooled train_forward_oof includes March–August and is separately reported. C is global across stocks; no hindsight per-stock architecture selection. Brier is a separate diagnostic, and positive temperature never changes direction.

## Advancement

[
  {
    "method": "P_cross_lr",
    "base": "P_own_lr",
    "weaker_gain": 0.018413775766716878,
    "macro_gain": -0.00984504448379131,
    "AAPL_gain": 0.018413775766716878,
    "AMZN_gain": -0.0381038647342995,
    "positive_months": 1,
    "passed": false
  },
  {
    "method": "P_cross_hgb",
    "base": "P_own_hgb",
    "weaker_gain": -0.0188614026320173,
    "macro_gain": -0.013265782356972011,
    "AAPL_gain": -0.0076701620819267236,
    "AMZN_gain": -0.0188614026320173,
    "positive_months": 1,
    "passed": false
  },
  {
    "method": "P_cross_tabpfn",
    "base": "P_own_tabpfn",
    "weaker_gain": -0.03897009828419129,
    "macro_gain": -0.03843251416799198,
    "AAPL_gain": -0.037894930051792675,
    "AMZN_gain": -0.03897009828419129,
    "positive_months": 0,
    "passed": false
  },
  {
    "method": "P_own_hgb",
    "base": "P_own_lr",
    "weaker_gain": -0.10994773956601872,
    "macro_gain": -0.05780719105600787,
    "AAPL_gain": 0.05258485160445936,
    "AMZN_gain": -0.1681992337164751,
    "positive_months": 0,
    "passed": false
  },
  {
    "method": "P_own_tabpfn",
    "base": "P_own_lr",
    "weaker_gain": -0.012735928804732621,
    "macro_gain": -0.03996373310447893,
    "AAPL_gain": -0.008940043253768848,
    "AMZN_gain": -0.070987422955189,
    "positive_months": 0,
    "passed": false
  },
  {
    "method": "P_cross_hgb",
    "base": "P_cross_lr",
    "weaker_gain": -0.1472229179647529,
    "macro_gain": -0.061227928929188574,
    "AAPL_gain": 0.026500913755815758,
    "AMZN_gain": -0.1489567716141929,
    "positive_months": 0,
    "passed": false
  },
  {
    "method": "P_cross_tabpfn",
    "base": "P_cross_lr",
    "weaker_gain": -0.07011980285564079,
    "macro_gain": -0.0685512027886796,
    "AAPL_gain": -0.0652487490722784,
    "AMZN_gain": -0.0718536565050808,
    "positive_months": 0,
    "passed": false
  },
  {
    "method": "T_A1",
    "base": "T_original",
    "weaker_gain": 0.0009736088167460188,
    "macro_gain": 0.014224727113686997,
    "AAPL_gain": 0.0009736088167460188,
    "AMZN_gain": 0.027475845410627975,
    "positive_months": 3,
    "passed": false
  },
  {
    "method": "T_original_interaction",
    "base": "T_original",
    "weaker_gain": -0.018345942365550205,
    "macro_gain": 0.000982367814392926,
    "AAPL_gain": -0.018345942365550205,
    "AMZN_gain": 0.020310677994336057,
    "positive_months": 2,
    "passed": false
  },
  {
    "method": "T_A1_interaction",
    "base": "T_A1",
    "weaker_gain": -0.08546210506994817,
    "macro_gain": -0.06564771920164073,
    "AAPL_gain": -0.08546210506994817,
    "AMZN_gain": -0.04583333333333328,
    "positive_months": 0,
    "passed": false
  },
  {
    "method": "T_mix",
    "base": "F2_new",
    "weaker_gain": 0.002124781536546161,
    "macro_gain": 0.014262040943185617,
    "AAPL_gain": 0.002124781536546161,
    "AMZN_gain": 0.026399300349825072,
    "positive_months": 2,
    "passed": false
  },
  {
    "method": "H1",
    "base": "H0",
    "weaker_gain": 0.0038714916165896174,
    "macro_gain": -0.019044389124238947,
    "AAPL_gain": 0.0038714916165896174,
    "AMZN_gain": -0.04196026986506751,
    "positive_months": 1,
    "passed": false
  },
  {
    "method": "H2",
    "base": "H1",
    "weaker_gain": -0.024289345367776827,
    "macro_gain": -0.005139841766014008,
    "AAPL_gain": -0.024289345367776827,
    "AMZN_gain": 0.01400966183574881,
    "positive_months": 1,
    "passed": false
  }
]

Blends: none; no contrast passed the registered gate.. Blend architecture was selected on June–August; any earlier blend figure is post-selection descriptive, not new outer validation. Final blend weights use past stock OOF only.

## Training and provenance

Actual LR/boosting/bilinear fitting is recorded in training_evidence.json. A1 encoders are frozen existing checkpoints, not retrained; original and A1 share token hashes. TabPFN conditions its fixed prior on past samples and does not update its foundation weights. Private model files and conditioning bundles are reloaded to verify predictions. Preliminary runs before upstream fingerprint hardening are preserved privately and excluded from reported results.

The Alpaca 2018 sample requires authentication (HTTP401). No external SPY/QQQ features are included and no coverage assumption is made.

Raw news, price features, model weights and environment files remain private. Joint-date paired 1/5-day block intervals are descriptive, not selection-adjusted significance tests. Exact coverage and fixed prediction cases are supplied; they do not prove causal explanations.

## 中文结论

**这轮已经实际完成，但没有找到两股稳定超过60%的方法；没有机制通过预先规定的晋级线，因此最终没有强行组合。**

1. **跨股票信息与非线性：**跨股LR在六月至八月两股月均BA为56.88%/57.06%，却没有保持到后续AMZN（49.20%）。浅层树在AAPL训练期可达60.30%，AMZN仅44.05%。TabPFN同样未显示两股一致优势。
2. **A1稠密迁移：**同样正文、同样价格和线性头，A1相对原始FinBERT的外层月均BA为AAPL+0.10pp、AMZN+2.75pp；较弱股票的改善不足1pp。A1后续为54.34%/46.65%。不能把抽取能力的进步直接等同于市场预测能力。
3. **交互：**增加价格×正文的rank2交互没有通过对照；三个种子的分数和波动全部报告，没有挑最好种子。
4. **历史新闻：**AMZN开发/后续原无新闻的53/95个窗口全部找到合格过去新闻。但H1后续AMZN只有48.72%，H2为48.67%；覆盖改善没有转化为稳定方向增量。
5. **8+8组合：**修正为与F2_new完全相同的价格输入后，外层较弱股BA仅增加0.21pp，未达到1pp晋级线。早期混入R1价格的无效归因版本及其blend已保存为私有调试记录，不进入最终报告。
6. **F2_new：**实际发出概率的全局选参得到后续55.14%/56.79%，但AMZN开发为49.17%。这是局部结果，不能称为跨时期稳定提升。旧F0/F1/F2保留，历史分数没有被改写。

### 观察、解释与下一步

已观察到：训练期较高成绩未跨期保持、加入历史新闻能填补覆盖、A1的少量抽取适配不足以保证稠密股价表示有效。合理解释包括信息冗余、当前窗口和旧信息的关系不足、样本/市场条件变化；本轮不能确定它们各自的因果贡献，更不能证明不存在可预测信号。

下一项尚未完成的独立信息分支是合格SPY/QQQ市场数据。Alpaca固定2018日期试取返回401，需要账户后核验真实历史权限和完整性；本轮没有使用这部分数据，没有付费，也不以注册账户为已完成实验。现有方案不再扩网格追分。

所有结果仍是已经暴露的历史回测。课程主baseline继续F0；旧F1是统一全文参照，旧F2是现代语义参照。这里没有用后续期成绩给两只股票各自选择不同架构。
