# 免费新增数据可行性：本轮调查

目标是能可靠对齐历史正文与盘中价格的新增时期，而非只找规模更大的日频数据。此次限时调查未建立合格新分支，也未锁定新保留期；没有购买数据或调用付费 API。结论不等于免费来源不存在。

| 来源 | 已核对内容 | 本轮判断 |
|---|---|---|
| Alpha Vantage | 官方文档将延长历史盘中数据功能标为 premium | 不作为已获免费历史分钟价格。见 [文档](https://www.alphavantage.co/documentation/) |
| Alpaca | 历史新闻文档说明可追溯至2015年，接口有正文选项；股票行情的 feed/权限需分别核验 | 值得继续核验的来源，但本轮没有账户权限、IEX/SIP同口径历史覆盖及使用许可的完整证据，未下载并宣称合格。见 [新闻](https://docs.alpaca.markets/us/docs/historical-news-data)、[新闻接口](https://docs.alpaca.markets/us/reference/news-3)、[行情 FAQ](https://docs.alpaca.markets/us/docs/market-data-faq) |
| FinMultiTime | 原论文摘要提到分钟粒度，但表2的中美价格是日频，新闻为分钟级时间戳；数据来源段也是日频OHLCV | 不能将新闻分钟时间戳当成四小时价格标签。见 [原论文](https://arxiv.org/html/2506.05019v1) |
| FNSPID | 作者仓库提供财经新闻与股价数据说明 | 本轮未验证与正文匹配的完整历史盘中价格，不接入四小时主任务。见 [作者仓库](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) |
| GDELT | 提供新闻事件/索引类数据 | 本轮未验证完整公司正文、首次可用性与分钟行情的配对包，不作为已经解决时间对齐的来源。见 [官方数据页](https://gdeltproject.org/data.html) |

若今后确认一个免费来源可用，应先记录许可、feed、时区、可用时间和缺失规则，再锁定最后至少60个共同交易日。不能先看新标签挑策略，再称其为新保留期。当前继续使用原历史回放，数据缺口没有被代码假装补齐。
