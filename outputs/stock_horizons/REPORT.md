# 预测周期比较：实际训练结果

**结论：本轮没有发现两只股票共同、跨时期稳定受益的更长周期；不据此更换项目主目标。** 四小时值得保留为辅助任务，日频暂不扩大搜索。所有分数都是已暴露历史回测，不是新独立测试。

## 实际执行

- 18 个最终 L2 LR 模型，162 个训练期候选折拟合；三个周期、两股、三种输入。所有最终模型重载概率一致。
- 原 1 小时 3,233 行的样本、标签、价格输入、新闻标题逐项一致；旧实验文件未修改。
- 六月—八月选 C，训练变换仅拟合过去；九十月与十一月以后分开报告。
- 4h 每小时滚动，目标重叠；区间按共同交易日配对，提供1日/5日块敏感性。

## 样本与新闻覆盖

| symbol   | horizon   | split      |    n |   days |   news_ratio |
|:---------|:----------|:-----------|-----:|-------:|-------------:|
| AAPL     | 1h        | test       |  364 |     62 |       1.0000 |
| AAPL     | 1h        | train      |  996 |    168 |       0.9930 |
| AAPL     | 1h        | validation |  252 |     42 |       0.9921 |
| AAPL     | 4h        | test       |  178 |     60 |       1.0000 |
| AAPL     | 4h        | train      |  499 |    167 |       0.9920 |
| AAPL     | 4h        | validation |  126 |     42 |       0.9841 |
| AAPL     | day       | test       |   61 |     61 |       1.0000 |
| AAPL     | day       | train      |  167 |    167 |       0.9880 |
| AAPL     | day       | validation |   42 |     42 |       1.0000 |
| AMZN     | 1h        | test       |  365 |     62 |       0.5342 |
| AMZN     | 1h        | train      | 1004 |    168 |       0.5667 |
| AMZN     | 1h        | validation |  252 |     42 |       0.6667 |
| AMZN     | 4h        | test       |  179 |     60 |       0.4693 |
| AMZN     | 4h        | train      |  499 |    167 |       0.5170 |
| AMZN     | 4h        | validation |  126 |     42 |       0.5794 |
| AMZN     | day       | test       |   61 |     61 |       0.3934 |
| AMZN     | day       | train      |  167 |    167 |       0.4192 |
| AMZN     | day       | validation |   42 |     42 |       0.4524 |

日频包含短交易日；4h没有足够盘中时长的短交易日会排除。三者平盘分别排除，所以共同开盘面板只保留两股三种标签均合格的日期。AMZN日频原测试仅39.34%窗口有新闻，AAPL则100%；不能说所有日频都缺新闻。

## 全部匹配模型：开发与原测试

| symbol   | horizon   | method      | period     |   n |     ba |     mcc |   brier |   pred_up | constant   |
|:---------|:----------|:------------|:-----------|----:|-------:|--------:|--------:|----------:|:-----------|
| AAPL     | 1h        | price       | test       | 364 | 0.4423 | -0.1336 |  0.2653 |    0.7555 | False      |
| AAPL     | 1h        | price       | validation | 252 | 0.5120 |  0.0275 |  0.2517 |    0.7460 | False      |
| AAPL     | 1h        | price_count | test       | 364 | 0.4434 | -0.1279 |  0.2763 |    0.7363 | False      |
| AAPL     | 1h        | price_count | validation | 252 | 0.4917 | -0.0199 |  0.2611 |    0.7738 | False      |
| AAPL     | 1h        | price_text  | test       | 364 | 0.4665 | -0.0767 |  0.2766 |    0.7473 | False      |
| AAPL     | 1h        | price_text  | validation | 252 | 0.4951 | -0.0125 |  0.2590 |    0.8095 | False      |
| AAPL     | 4h        | price       | test       | 178 | 0.5292 |  0.0726 |  0.2665 |    0.7978 | False      |
| AAPL     | 4h        | price       | validation | 126 | 0.5132 |  0.0385 |  0.2573 |    0.8651 | False      |
| AAPL     | 4h        | price_count | test       | 178 | 0.5289 |  0.0706 |  0.2738 |    0.7865 | False      |
| AAPL     | 4h        | price_count | validation | 126 | 0.4899 | -0.0304 |  0.2774 |    0.8730 | False      |
| AAPL     | 4h        | price_text  | test       | 178 | 0.5302 |  0.0829 |  0.2712 |    0.8427 | False      |
| AAPL     | 4h        | price_text  | validation | 126 | 0.5157 |  0.0515 |  0.2730 |    0.8968 | False      |
| AAPL     | day       | price       | test       |  61 | 0.4935 | -0.0144 |  0.2875 |    0.7213 | False      |
| AAPL     | day       | price       | validation |  42 | 0.4386 | -0.1494 |  0.2503 |    0.7857 | False      |
| AAPL     | day       | price_count | test       |  61 | 0.4578 | -0.0908 |  0.3146 |    0.6885 | False      |
| AAPL     | day       | price_count | validation |  42 | 0.4841 | -0.0426 |  0.2557 |    0.8333 | False      |
| AAPL     | day       | price_text  | test       |  61 | 0.4535 | -0.1253 |  0.2956 |    0.8361 | False      |
| AAPL     | day       | price_text  | validation |  42 | 0.5500 |  0.2345 |  0.2520 |    0.9524 | False      |
| AMZN     | 1h        | price       | test       | 365 | 0.5066 |  0.0134 |  0.2580 |    0.6000 | False      |
| AMZN     | 1h        | price       | validation | 252 | 0.4887 | -0.0227 |  0.2536 |    0.5357 | False      |
| AMZN     | 1h        | price_count | test       | 365 | 0.5093 |  0.0190 |  0.2561 |    0.5918 | False      |
| AMZN     | 1h        | price_count | validation | 252 | 0.4926 | -0.0149 |  0.2545 |    0.5317 | False      |
| AMZN     | 1h        | price_text  | test       | 365 | 0.5286 |  0.0576 |  0.2638 |    0.5507 | False      |
| AMZN     | 1h        | price_text  | validation | 252 | 0.5239 |  0.0479 |  0.2585 |    0.5079 | False      |
| AMZN     | 4h        | price       | test       | 179 | 0.4620 | -0.0792 |  0.2677 |    0.6425 | False      |
| AMZN     | 4h        | price       | validation | 126 | 0.5009 |  0.0019 |  0.2620 |    0.6111 | False      |
| AMZN     | 4h        | price_count | test       | 179 | 0.4620 | -0.0792 |  0.2798 |    0.6425 | False      |
| AMZN     | 4h        | price_count | validation | 126 | 0.5306 |  0.0608 |  0.2659 |    0.5952 | False      |
| AMZN     | 4h        | price_text  | test       | 179 | 0.5121 |  0.0254 |  0.2747 |    0.6480 | False      |
| AMZN     | 4h        | price_text  | validation | 126 | 0.4972 | -0.0055 |  0.2798 |    0.5952 | False      |
| AMZN     | day       | price       | test       |  61 | 0.4742 | -0.0522 |  0.2464 |    0.5738 | False      |
| AMZN     | day       | price       | validation |  42 | 0.4423 | -0.1132 |  0.2566 |    0.5714 | False      |
| AMZN     | day       | price_count | test       |  61 | 0.5081 |  0.0161 |  0.2981 |    0.5082 | False      |
| AMZN     | day       | price_count | validation |  42 | 0.5264 |  0.0523 |  0.2709 |    0.4048 | False      |
| AMZN     | day       | price_text  | test       |  61 | 0.4898 | -0.0211 |  0.2851 |    0.6230 | False      |
| AMZN     | day       | price_text  | validation |  42 | 0.5192 |  0.0374 |  0.2803 |    0.4762 | False      |

## 价格＋数量＋标题文本：周期比较

| symbol   | horizon   | period     |   n |     ba |   brier |
|:---------|:----------|:-----------|----:|-------:|--------:|
| AAPL     | 1h        | test       | 364 | 0.4665 |  0.2766 |
| AAPL     | 1h        | validation | 252 | 0.4951 |  0.2590 |
| AAPL     | 4h        | test       | 178 | 0.5302 |  0.2712 |
| AAPL     | 4h        | validation | 126 | 0.5157 |  0.2730 |
| AAPL     | day       | test       |  61 | 0.4535 |  0.2956 |
| AAPL     | day       | validation |  42 | 0.5500 |  0.2520 |
| AMZN     | 1h        | test       | 365 | 0.5286 |  0.2638 |
| AMZN     | 1h        | validation | 252 | 0.5239 |  0.2585 |
| AMZN     | 4h        | test       | 179 | 0.5121 |  0.2747 |
| AMZN     | 4h        | validation | 126 | 0.4972 |  0.2798 |
| AMZN     | day       | test       |  61 | 0.4898 |  0.2851 |
| AMZN     | day       | validation |  42 | 0.5192 |  0.2803 |

### 观察与解释

- AAPL 文本：原测试1h 46.65%、4h 53.02%、day 45.35%；4h价格自身已52.92%，文本只增加约0.10个百分点。不能归因为新闻理解改善。
- AMZN 文本：原测试1h 52.86%、4h 51.21%、day 48.98%；4h相对自己的价格基线46.20%有增量，但九十月49.72%低于价格50.09%。
- 日频每股训练仅167个日期，原测试61个；AAPL日频文本九十月55.00%而后来45.35%，再次显示只挑单段最高分的风险。
- 全样本原测试中，六个“价格＋文本”模型Brier全部高于固定0.5概率的0.25。方向分数与概率质量必须分开判断。

## 共同开盘日期面板

| symbol   | horizon   | method      | split      |   n |     ba |     mcc |   brier |   up_rate |   pred_up | constant   |
|:---------|:----------|:------------|:-----------|----:|-------:|--------:|--------:|----------:|----------:|:-----------|
| AAPL     | 1h        | price       | test       |  58 | 0.4773 | -0.1694 |  0.2784 |    0.3793 |    0.9828 | False      |
| AAPL     | 1h        | price       | validation |  42 | 0.4583 | -0.1936 |  0.2536 |    0.5714 |    0.9524 | False      |
| AAPL     | 1h        | price_count | test       |  58 | 0.4773 | -0.1694 |  0.2968 |    0.3793 |    0.9828 | False      |
| AAPL     | 1h        | price_count | validation |  42 | 0.5000 |  0.0000 |  0.2579 |    0.5714 |    1.0000 | True       |
| AAPL     | 1h        | price_text  | test       |  58 | 0.5000 |  0.0000 |  0.2998 |    0.3793 |    1.0000 | True       |
| AAPL     | 1h        | price_text  | validation |  42 | 0.5000 |  0.0000 |  0.2514 |    0.5714 |    1.0000 | True       |
| AAPL     | 4h        | price       | test       |  58 | 0.5321 |  0.1268 |  0.2722 |    0.4828 |    0.9310 | False      |
| AAPL     | 4h        | price       | validation |  42 | 0.5000 |  0.0000 |  0.2616 |    0.5238 |    1.0000 | True       |
| AAPL     | 4h        | price_count | test       |  58 | 0.5321 |  0.1268 |  0.2848 |    0.4828 |    0.9310 | False      |
| AAPL     | 4h        | price_count | validation |  42 | 0.5000 |  0.0000 |  0.2722 |    0.5238 |    1.0000 | True       |
| AAPL     | 4h        | price_text  | test       |  58 | 0.5167 |  0.1280 |  0.2844 |    0.4828 |    0.9828 | False      |
| AAPL     | 4h        | price_text  | validation |  42 | 0.5000 |  0.0000 |  0.2660 |    0.5238 |    1.0000 | True       |
| AAPL     | day       | price       | test       |  58 | 0.4905 | -0.0213 |  0.2831 |    0.4828 |    0.7241 | False      |
| AAPL     | day       | price       | validation |  42 | 0.4386 | -0.1494 |  0.2503 |    0.5238 |    0.7857 | False      |
| AAPL     | day       | price_count | test       |  58 | 0.4714 | -0.0608 |  0.3043 |    0.4828 |    0.6724 | False      |
| AAPL     | day       | price_count | validation |  42 | 0.4841 | -0.0426 |  0.2557 |    0.5238 |    0.8333 | False      |
| AAPL     | day       | price_text  | test       |  58 | 0.4595 | -0.1071 |  0.2888 |    0.4828 |    0.8276 | False      |
| AAPL     | day       | price_text  | validation |  42 | 0.5500 |  0.2345 |  0.2520 |    0.5238 |    0.9524 | False      |
| AMZN     | 1h        | price       | test       |  58 | 0.5107 |  0.0273 |  0.2686 |    0.4828 |    0.8103 | False      |
| AMZN     | 1h        | price       | validation |  42 | 0.4771 | -0.0535 |  0.2553 |    0.4524 |    0.7619 | False      |
| AMZN     | 1h        | price_count | test       |  58 | 0.5452 |  0.1153 |  0.2636 |    0.4828 |    0.8103 | False      |
| AMZN     | 1h        | price_count | validation |  42 | 0.4725 | -0.0605 |  0.2523 |    0.4524 |    0.7143 | False      |
| AMZN     | 1h        | price_text  | test       |  58 | 0.5452 |  0.1153 |  0.2647 |    0.4828 |    0.8103 | False      |
| AMZN     | 1h        | price_text  | validation |  42 | 0.5160 |  0.0338 |  0.2539 |    0.4524 |    0.6667 | False      |
| AMZN     | 4h        | price       | test       |  58 | 0.5445 |  0.1010 |  0.2488 |    0.5517 |    0.7414 | False      |
| AMZN     | 4h        | price       | validation |  42 | 0.4027 | -0.2202 |  0.2954 |    0.4524 |    0.7381 | False      |
| AMZN     | 4h        | price_count | test       |  58 | 0.5793 |  0.1802 |  0.2487 |    0.5517 |    0.7414 | False      |
| AMZN     | 4h        | price_count | validation |  42 | 0.5206 |  0.0454 |  0.2957 |    0.4524 |    0.7143 | False      |
| AMZN     | 4h        | price_text  | test       |  58 | 0.5829 |  0.1812 |  0.2396 |    0.5517 |    0.7069 | False      |
| AMZN     | 4h        | price_text  | validation |  42 | 0.4245 | -0.1664 |  0.3036 |    0.4524 |    0.7143 | False      |
| AMZN     | day       | price       | test       |  58 | 0.4409 | -0.1206 |  0.2485 |    0.5345 |    0.6034 | False      |
| AMZN     | day       | price       | validation |  42 | 0.4423 | -0.1132 |  0.2566 |    0.3810 |    0.5714 | False      |
| AMZN     | day       | price_count | test       |  58 | 0.4988 | -0.0024 |  0.3003 |    0.5345 |    0.5172 | False      |
| AMZN     | day       | price_count | validation |  42 | 0.5264 |  0.0523 |  0.2709 |    0.3810 |    0.4048 | False      |
| AMZN     | day       | price_text  | test       |  58 | 0.4916 | -0.0172 |  0.2854 |    0.5345 |    0.6207 | False      |
| AMZN     | day       | price_text  | validation |  42 | 0.5192 |  0.0374 |  0.2803 |    0.3810 |    0.4762 | False      |

AMZN 4h文本在共同开盘原测试58个日期上BA58.29%、Brier0.2396，但同面板九十月BA42.45%、Brier0.3036；新闻数量对照原测试已57.93%。不能将58.29%单独宣传为稳定文本优势。这个面板未重训开盘专用模型，训练期样本数量仍有差异，因此不是纯预测距离因果实验。

## 新闻相对价格：配对不确定性

| symbol   | horizon   | split      | method      |   block_days |   ba_low |   ba_high |   brier_low |   brier_high |
|:---------|:----------|:-----------|:------------|-------------:|---------:|----------:|------------:|-------------:|
| AAPL     | 1h        | test       | price_count |            1 |  -0.0264 |    0.0294 |      0.0067 |       0.0154 |
| AAPL     | 1h        | test       | price_count |            5 |  -0.0220 |    0.0189 |      0.0072 |       0.0156 |
| AAPL     | 1h        | test       | price_text  |            1 |  -0.0026 |    0.0508 |      0.0068 |       0.0159 |
| AAPL     | 1h        | test       | price_text  |            5 |  -0.0089 |    0.0506 |      0.0085 |       0.0159 |
| AAPL     | 1h        | validation | price_count |            1 |  -0.0626 |    0.0188 |      0.0013 |       0.0184 |
| AAPL     | 1h        | validation | price_count |            5 |  -0.0554 |    0.0179 |      0.0022 |       0.0180 |
| AAPL     | 1h        | validation | price_text  |            1 |  -0.0706 |    0.0324 |     -0.0008 |       0.0159 |
| AAPL     | 1h        | validation | price_text  |            5 |  -0.0709 |    0.0326 |     -0.0002 |       0.0158 |
| AAPL     | 4h        | test       | price_count |            1 |  -0.0275 |    0.0250 |     -0.0002 |       0.0157 |
| AAPL     | 4h        | test       | price_count |            5 |  -0.0275 |    0.0232 |      0.0013 |       0.0147 |
| AAPL     | 4h        | test       | price_text  |            1 |  -0.0440 |    0.0437 |     -0.0027 |       0.0117 |
| AAPL     | 4h        | test       | price_text  |            5 |  -0.0362 |    0.0318 |     -0.0009 |       0.0112 |
| AAPL     | 4h        | validation | price_count |            1 |  -0.0787 |    0.0313 |      0.0045 |       0.0368 |
| AAPL     | 4h        | validation | price_count |            5 |  -0.0625 |    0.0403 |      0.0122 |       0.0345 |
| AAPL     | 4h        | validation | price_text  |            1 |  -0.0492 |    0.0585 |      0.0043 |       0.0284 |
| AAPL     | 4h        | validation | price_text  |            5 |  -0.0349 |    0.0697 |      0.0093 |       0.0266 |
| AAPL     | day       | test       | price_count |            1 |  -0.1291 |    0.0550 |      0.0025 |       0.0532 |
| AAPL     | day       | test       | price_count |            5 |  -0.1390 |    0.0500 |      0.0100 |       0.0456 |
| AAPL     | day       | test       | price_text  |            1 |  -0.1172 |    0.0389 |     -0.0040 |       0.0190 |
| AAPL     | day       | test       | price_text  |            5 |  -0.1217 |    0.0343 |     -0.0036 |       0.0195 |
| AAPL     | day       | validation | price_count |            1 |  -0.0833 |    0.1875 |     -0.0208 |       0.0345 |
| AAPL     | day       | validation | price_count |            5 |  -0.0601 |    0.1394 |     -0.0214 |       0.0326 |
| AAPL     | day       | validation | price_text  |            1 |  -0.0139 |    0.2381 |     -0.0148 |       0.0187 |
| AAPL     | day       | validation | price_text  |            5 |   0.0159 |    0.1977 |     -0.0114 |       0.0182 |
| AMZN     | 1h        | test       | price_count |            1 |  -0.0226 |    0.0302 |     -0.0048 |       0.0011 |
| AMZN     | 1h        | test       | price_count |            5 |  -0.0270 |    0.0283 |     -0.0046 |       0.0008 |
| AMZN     | 1h        | test       | price_text  |            1 |  -0.0085 |    0.0535 |     -0.0006 |       0.0125 |
| AMZN     | 1h        | test       | price_text  |            5 |  -0.0126 |    0.0418 |      0.0003 |       0.0114 |
| AMZN     | 1h        | validation | price_count |            1 |  -0.0339 |    0.0415 |     -0.0034 |       0.0055 |
| AMZN     | 1h        | validation | price_count |            5 |  -0.0355 |    0.0451 |     -0.0032 |       0.0044 |
| AMZN     | 1h        | validation | price_text  |            1 |  -0.0213 |    0.0910 |     -0.0060 |       0.0153 |
| AMZN     | 1h        | validation | price_text  |            5 |  -0.0076 |    0.0861 |     -0.0045 |       0.0137 |
| AMZN     | 4h        | test       | price_count |            1 |  -0.0431 |    0.0445 |      0.0029 |       0.0211 |
| AMZN     | 4h        | test       | price_count |            5 |  -0.0309 |    0.0388 |      0.0025 |       0.0208 |
| AMZN     | 4h        | test       | price_text  |            1 |  -0.0116 |    0.1143 |     -0.0114 |       0.0226 |
| AMZN     | 4h        | test       | price_text  |            5 |  -0.0081 |    0.1134 |     -0.0146 |       0.0241 |
| AMZN     | 4h        | validation | price_count |            1 |  -0.0414 |    0.1025 |     -0.0082 |       0.0162 |
| AMZN     | 4h        | validation | price_count |            5 |  -0.0224 |    0.1247 |     -0.0105 |       0.0168 |
| AMZN     | 4h        | validation | price_text  |            1 |  -0.1061 |    0.1039 |     -0.0016 |       0.0403 |
| AMZN     | 4h        | validation | price_text  |            5 |  -0.0788 |    0.0973 |      0.0007 |       0.0335 |
| AMZN     | day       | test       | price_count |            1 |  -0.0839 |    0.1617 |      0.0054 |       0.1002 |
| AMZN     | day       | test       | price_count |            5 |  -0.1045 |    0.1719 |      0.0040 |       0.0918 |
| AMZN     | day       | test       | price_text  |            1 |  -0.0959 |    0.1296 |     -0.0063 |       0.0847 |
| AMZN     | day       | test       | price_text  |            5 |  -0.1169 |    0.1348 |     -0.0134 |       0.0813 |
| AMZN     | day       | validation | price_count |            1 |  -0.0444 |    0.2251 |     -0.0357 |       0.0624 |
| AMZN     | day       | validation | price_count |            5 |   0.0201 |    0.2143 |     -0.0414 |       0.0594 |
| AMZN     | day       | validation | price_text  |            1 |  -0.0333 |    0.1964 |     -0.0273 |       0.0796 |
| AMZN     | day       | validation | price_text  |            5 |  -0.0084 |    0.1963 |     -0.0291 |       0.0686 |

区间未校正探索和模型选择；区间跨0不能证明没有作用。部分月样本很小，完整逐月值在 metrics.csv，二月尾部不完整月份不得用于单月稳定性宣传。

## 建议与下一步

1. 保留1h主线与4h辅助对照；本轮不宣布最佳周期。改变项目主目标需要更稳定的月份证据。
2. 优先执行Pro提出的严格时间交叉拟合/数量校准匹配对照；不要因为AAPL 4h较高就同时叠加复杂模型。
3. 如继续日频，应另立24/72小时新闻回看实验，先检查覆盖和事件时效。当前4小时输入对照没有排除其他日频输入设计，但不应再无限搜索。
4. 获取真正新时期后，才有条件确认周期选择的泛化表现。

## 文件与复现

- run.py：数据构造、训练、全候选保存、bootstrap。
- finalize.py：模型与输入再核验、动态表格与报告。报告观察段针对本次v1，换run需重新解读。
- runs/v1/protocol.json：事前协议；coverage.csv：全部候选窗口及排除原因。
- runs/v1/models.json：每个模型训练键、预测键、训练指标、哈希。
- runs/v1/predictions.csv / metrics.csv / common_open_metrics.csv / increments.csv：逐样本、逐月与对照增量。
- runs/v1/reference_metrics.csv：固定0.5和训练先验参考。
- 默认拒绝覆盖，完整命令见 README.md。