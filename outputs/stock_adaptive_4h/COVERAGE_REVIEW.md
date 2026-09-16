# AMZN 覆盖与事实抽取检查

## 统计范围

`audit.py` 按原 1,607 窗口的可用时间扫描原始正文；原分类样本与原入选文章集合保持固定。最终有效审计为 `coverage/v3/`，v1 为时间解析失败，v2 为收紧规则前的候选审计；两者保留。

原始 78,055 行 → 基本可用性筛选 59,813 行 → 原窗口时点范围内读取 14,105 篇 → 两公司候选对 17,844 → 暂定事件字段 2,665。数量不是人工验证的正确事件数。

AMZN 后续 179 窗口：原有新闻 84；严格规则候选可达到 99，来自 21 篇新候选，增加 15 个潜在窗口。宽规则 v2 的 141 窗口/90 篇不能用作正式覆盖改善结果。训练严格候选 258→286；开发73→73。

## 助手开发审阅与规则变化

实际先阅读 v2 复核候选前 8 篇 AAPL、前 8 篇 AMZN。这些阅读用于修改规则，不能再把它们称为独立检查集。

| 股票 | 标题/内容标识 | 判断与规则含义 |
|---|---|---|
| AAPL | Meet Intel's Record Quarter | Apple troubles 引导/背景，不足以认定新 Apple 事件 |
| AAPL | Amazon stock tops 2000 | Apple 市场表现比较，不能按直接公司行动计数 |
| AAPL | Hot Dividend Stocks | Apple 宣布派息有对象明确的事实；日期和新旧另查 |
| AAPL | Corning / Gorilla Glass | 历史使用与未来预期，不能当新公告 |
| AAPL | S&P futures / Buffett-Tepper Q4 | 季度持仓披露和过去行情，披露时间需区别事件发生时间 |
| AAPL | Wall Street / Fed / Tesla employees | Apple 招聘 Tesla 员工是直接主张；严格规则可能仍漏，召回有限 |
| AAPL | You searched for UK / fonecable | 聚合/导航页，不应当独立公司新闻 |
| AAPL | Mobile payments / Alipay | Apple 同意在门店接受支付宝是明确公司事实 |
| AMZN | Microsoft 1 trillion | Amazon 是市值比较对象，排除直接事件标签 |
| AMZN | FAANG next Berkshire | 交易员意见，不是公司做了什么变化 |
| AMZN | Nightly Business Report May 7 | Telsey initiated Amazon Outperform，是标题漏掉的评级事实 |
| AMZN | Apple / D.A. Davidson | 评级动作对象是 Apple，Amazon 仅用于比较，原宽规则误判 |
| AMZN | tech-stock money flows | 排名/交易资金讨论，不能认定公司新公告 |
| AMZN | Trump / Google / AWS | Amazon 过去涨幅和 AWS 背景，不是本次直接事件 |
| AMZN | money finances / Echo holidays | Echo 假日销售纪录是公司事实，但可能很旧 |
| AMZN | closed-end funds | 九月债务权益比只是背景，不是本次公司行动 |

据此去除泛泛的 is/has/stock 等“动作”条件，明确目标公司与评级动作的联系，并拒绝市值比较误标。加入三项覆盖回归测试。

随后又查看 v3 前 8 篇 AMZN 严格候选（部分和上表重复，不宣称 24 篇独立文章）：

- Nightly Business Report：Telsey 初始评级，直接事实。
- Echo 假日纪录：直接历史事实，不能只按抓取新到达认定新事件。
- Major chip flaws：Amazon 关于云机器保护/更新的声明，直接事实。
- FAAMNG Q1：去年收购 Whole Foods，历史背景。
- 7 Biggest Price Target Changes For Friday：Credit Suisse 的 Amazon 1750→1800，直接目标价调整。
- 3 FAANG Stocks in the News Today：Amazon/Marriott 合作，直接事实。
- Tech War：四月 AWS Blockchain Templates，在五月文章中复述。
- Wall Street Breakfast / Powell：Amazon/Berkshire/JPM 医疗合作回顾，首次披露未知。

结论是“标题过滤确有漏项，同时规则候选仍有新旧混杂”，不是“全部候选已合格”。也不能从现有语料推断同期 AMZN 新闻已完整覆盖。

## 质量合同

输出含 company、kind、action、old/new value/rating、unit、period、actor、字符证据、发布时间/可用时间、历史/未知/冲突标记及哈希。所有 `prediction_accepted=false`。数值单位冲突、多个目标混杂、机构不确定等情况保留未知，不猜首次披露。

按过去信息进行近似事件分组：标准化文本重复或近期标题相似度；仅是可复核启发式，不宣称已验证真实事件身份。分组后再分开发和检查，避免同组转载跨边。证据关系可从 `events.jsonl` 直接读取；没有训练 GNN。

`independent_check.csv` 有65行：AAPL评级30、目标价30、指引1；AMZN目标价4。独立复核字段全部为空。`quality_gate` 每次调用都核对样本集合、证据/提案内容哈希、公司/事件类型/组和复核字段；助手身份不视为独立复核。每拟使用类型至少30个非重复组，关键字段正确率至少90%，另报漏抽、不确定和覆盖。当前 `accepted_types=[]`。

数量不足的类型不能用删除字段、合并不同任务或助手标签宣布完整验收。即使将来某类型通过抽取质量，也要在匹配样本的预测实验中检验是否有增量。现在可以完成其他训练、报告与演示，正式关系输入仍等待独立检查。
