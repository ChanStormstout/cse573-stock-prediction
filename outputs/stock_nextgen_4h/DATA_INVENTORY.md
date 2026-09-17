# 数据盘点与外部数据可行性

本文件在本轮模型评分前生成。结论是：AAPL／AMZN 五分钟 OHLC、交易日历和新闻正文可用；市场 ETF、宏观公布表、盈利预期／实际值在本地缺失。

## 已有输入

| 项目 | 状态 | 可用于什么 | 限制 |
|---|---|---|---|
| AAPL 5分钟 OHLC | 可用，38,634 行 | R0/R1/R2、标签复核 | 第七列语义未知 |
| AMZN 5分钟 OHLC | 可用，30,283 行 | R0/R1/R2、标签复核 | 第七列语义未知 |
| XNYS 日历 | 可用，273 日 | 开收盘和隔夜边界 | 不是宏观事件表 |
| 新闻索引和正文 | 可用，78,055 条 | P0--P3 | 语料偏向 AAPL，时间字段存在已知异常 |
| SPY/QQQ/XLK/XLY 分钟线 | 缺失 | 无 | M1--M3 停止 |
| FOMC/CPI/就业公布及 consensus | 缺失 | 无 | 不能构造 surprise |
| 盈利 consensus/actuals | 缺失 | 无 | 不能构造 earnings surprise |

## 第七列是否为成交量

课程文件没有定义第七列。它是整数且为正，只能记录为 `activity`；这些现象不能证明它是成交量。因此本轮不构造 volume、VWAP、成交量冲击或订单流特征。

## 新闻时间、来源与重复

新闻来自 2,175 个站点；精确哈希重复 3,251 条，规范化正文哈希重复 3,625 条。AAPL 标记 78,021 条，AMZN 标记 18,118 条，后者全部同时带有 AAPL 标记，说明语料覆盖不对称。

原始字段中有 11,651 条记录的抓取时间早于标注发布时间。流水线的 `available_utc` 明确定义为可用发布时间和抓取时间中的较晚者，所以派生后的 `available_utc < published_utc` 为 0 条；这只是保守截止规则，不证明源时间元数据无异常。

## 数据角色

| 角色 | 本轮内容 |
|---|---|
| 监督标签 | 由目标区间48根五分钟线重新计算的四小时UP/DOWN |
| 普通输入 | 截止前AAPL/AMZN OHLC、新闻标题/正文/来源/时间、交易日边界 |
| 外部元数据 | XNYS日历；外部供应商文档只用于可行性判断 |
| 暂定LLM标注 | 本轮预测没有使用；旧事件标注不进入P0--P3 |
| 完全缺失 | 市场/行业分钟线、宏观公布与consensus、盈利consensus/actuals、独立复核事件标签 |

## 外部分钟数据门槛

官方资料显示 Alpha Vantage 的历史月度 intraday 属于 premium；Polygon 的长期分钟聚合需要相应付费访问；Twelve Data 虽有 intraday 接口，但本轮没有确认足够的免费 2018 历史权限和可复现／再分发条件。没有下载或混入来源不明的数据。

因此只执行 M0=R1。M1--M3、market PCA、HMM 和 graph 分支按预注册停止，不把日频 ETF 数据混入四小时任务。

## 任务数据

固定保留 1,607 个四小时窗口：AAPL|test=178，AAPL|train=499，AAPL|validation=126，AMZN|test=179，AMZN|train=499，AMZN|validation=126。

所有 September 2018 之后的标签均已在历史探索中暴露，本轮仍称探索性历史回放。

## External feasibility sources

- [Alpha Vantage official documentation](https://www.alphavantage.co/documentation/)
- [Polygon minute aggregate documentation](https://polygon.io/docs/flat-files/stocks/minute-aggregates/2009/03)
- [Twelve Data historical data guidance](https://support.twelvedata.com/en/articles/5214728-getting-historical-data)
