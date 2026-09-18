# 全语料 reaction 数据审计

- 原始索引 78,055 行；英文 70,732；足够长文本 70,601。
- 目标关联单位是 `(article_key, target_symbol)`，不是单独的 article key，所以一篇同时涉及 AAPL 与 AMZN 的文章可以生成两个候选单位。
- 候选 89,958 对经过全文/标题规范化后得到 85,402 个 all-time canonical groups；Jan–Aug 2018 保留 51,485 对。
- AAPL 70,591 个候选对、AMZN 19,367 个候选对；AAPL 绝大部分是 Tier 0，AMZN 仍有较多 Tier 2。Tier 是规则关联层级，不等于人工确认。
- available time 使用 `max(published_utc, crawled_utc)`。30/60/120/240 分钟反应必须在同一 regular session、连续五分钟 bar 内完成；跨夜不计入同一反应。
- 文章正文、标题和证据句只保存在私有 `work/stock-data/reaction_features_4h/v1`，公开 CSV 不含正文。
- `reaction_gate.json` 记录了完整计数和两股 gate；它不证明新闻覆盖代表真实市场全部新闻，也不证明反应是因果。
- 独立人工抽取/关联检查尚未完成，后续若要部署 target-specific graph 或完整 AR2，必须先补充独立检查。
