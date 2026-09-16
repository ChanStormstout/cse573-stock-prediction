# 免费新增数据可行性调查 · 2026-09-15

结论：本轮未获得并验收可直接对齐的新增“完整新闻正文＋小时价格”数据包，因此没有建立新的60共同交易日保留期。不是断言不存在免费来源。

| 来源 | 官方证据 | 本轮判断 |
|---|---|---|
| Alpaca / Benzinga | [历史新闻](https://docs.alpaca.markets/us/docs/historical-news-data)称覆盖至2015；[接口](https://docs.alpaca.markets/us/reference/news-3)支持正文（如有）、起止时间、股票和分页，需要API认证 | 有潜力。未提供/接入免费账户凭证，未验证具体免费权限、修订历史、全文可用率与课堂再分发许可；不能宣布已取得合格数据 |
| Alpaca 价格 | [官方历史数据说明](https://alpaca.markets/learn/fetch-historical-data)允许符合条件账户使用IEX历史股票数据 | 单交易所来源需与新闻时点、缺线、公司行动核对；本轮未下载，不能把官方可用性描述当作验收 |
| Alpha Vantage | [TIME_SERIES_INTRADAY 官方文档](https://www.alphavantage.co/documentation/)明确标为Premium，支持1/5/15/30/60分钟 | 不符合本轮免费要求，未购买 |
| GDELT | [数据说明](https://gdeltproject.org/data.html)提供事件、元数据、链接与部分上下文；[使用说明](https://gdeltproject.org/about.html)要求引用GDELT | 不能由元数据直接推定拥有全部历史新闻正文及全文再分发权；还缺配套合格小时价格，因此本轮不建立新实验数据分支 |

## 如以后接入候选数据

1. 先核实许可、正文覆盖、created/updated时间含义、UTC时区、可用时间和修订记录。
2. 对齐两股共同交易日及连续小时价格，记录来源与公司行动处理。
3. 在读取预测标签作设计之前，以日期锁定最后至少60个共同交易日；写入哈希清单。
4. 开发数据和新保留期严格分开；日频数据另立任务。

本调查仅使用公开官方文档，没有调用付费API、注册账户或接触密钥。没有新增数据并不阻止现有历史回测完成。
