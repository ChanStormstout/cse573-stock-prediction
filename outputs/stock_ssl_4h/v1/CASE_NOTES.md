# 固定案例诊断

复用上一轮按SHA256预先选定的案例键，不按本轮收益挑选。下表是预测变化，不是市场变化的因果解释。

| 窗口 | 标签 | B正确 | RAW正确 | SSL正确 |
|---|---:|---|---|---|
| AAPL|2018-10-25 14:30:00+00:00 | 1 | False | True | True |
| AAPL|2018-09-21 14:30:00+00:00 | 1 | True | True | True |
| AAPL|2018-09-04 14:30:00+00:00 | 1 | False | False | False |
| AAPL|2018-10-04 14:30:00+00:00 | 0 | False | False | False |
| AAPL|2018-11-16 15:30:00+00:00 | 0 | False | False | False |
| AAPL|2018-12-26 16:30:00+00:00 | 1 | True | True | True |
| AAPL|2018-08-21 13:30:00+00:00 | 1 | True | True | True |
| AAPL|2018-04-24 14:30:00+00:00 | 0 | False | False | False |
| AAPL|2018-07-10 13:30:00+00:00 | 1 | True | True | True |
| AAPL|2018-04-06 13:30:00+00:00 | 0 | False | False | False |
| AMZN|2018-09-13 14:30:00+00:00 | 0 | False | False | False |
| AMZN|2018-10-01 15:30:00+00:00 | 0 | True | True | True |
| AMZN|2018-10-02 13:30:00+00:00 | 0 | False | False | False |
| AMZN|2018-10-10 14:30:00+00:00 | 0 | False | False | True |
| AMZN|2019-01-14 14:30:00+00:00 | 1 | True | True | True |
| AMZN|2018-11-09 16:30:00+00:00 | 1 | True | False | True |
| AMZN|2018-11-12 15:30:00+00:00 | 0 | False | False | False |
| AMZN|2018-11-30 14:30:00+00:00 | 1 | True | True | True |
| AMZN|2018-03-09 14:30:00+00:00 | 1 | True | False | True |
| AMZN|2018-04-05 15:30:00+00:00 | 1 | True | True | True |
| AMZN|2018-07-16 14:30:00+00:00 | 1 | True | True | True |
| AMZN|2018-04-02 15:30:00+00:00 | 0 | False | False | False |

AMZN后续原无新闻窗口固定95个；SSL相对旧F2净多对4个，但不能把这些已经观察到的窗口用于选择开关。完整共同错误/新增错误计数见error_transitions.csv。

输入异常与市场预测错误分开：缺少完整48根输入时严格回退B；具备完整输入且预测错，只能证明该预测错，不能自动归因于某种价格形态。
