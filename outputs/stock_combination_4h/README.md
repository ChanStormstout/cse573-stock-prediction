# 有限组合、归因与课程交付

- [预注册](PRE_REGISTRATION.md)
- [实验报告](v1/REPORT.md)
- [课程报告主稿](../../docs/COURSE_REPORT_4H.md)
- [离线回放](v1/demo.html)

## 复现

从仓库根目录运行（需要本地私有数据、既有模型和finbert-env环境）：

```sh
work/stock-data/finbert-env/bin/python outputs/stock_combination_4h/run.py
work/stock-data/finbert-env/bin/python outputs/stock_combination_4h/deliver.py
work/stock-data/finbert-env/bin/python outputs/stock_combination_4h/verify.py
```

run.py拒绝覆盖既有私有v1目录。另跑实验必须先预注册并更换公开和私有运行目录，不能删除旧结果来绕过保护。deliver.py只重建报告和回放；verify.py重算已保存预测的指标，不训练模型。环境依赖沿用stock_paper_methods_4h，权重与原文不在Git。

## 演示

直接用浏览器打开v1/demo.html即可，无外部资源、无需模型或网络。也可在仓库根目录启动：

```sh
python3 -m http.server 8773 --bind 127.0.0.1
```

访问 http://127.0.0.1:8773/outputs/stock_combination_4h/v1/demo.html 。此演示只回放609个保存预测，不提供实时推理。标签默认视觉隐藏，不是安全隔离的盲测。

浏览器已核验AAPL/AMZN切换、无新闻概率一致、揭示标签、换时期后重新隐藏标签及页面布局。54行汇总指标独立重算通过；详见v1/verification.json。
