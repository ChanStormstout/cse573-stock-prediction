# 每次更新流程

1. 保留历史实验，新实验使用独立输出目录。
2. 更新 `outputs/PROJECT_LOG.md`、`docs/CURRENT_STATUS.md`；若机制状态改变，同步对应报告。
3. 新结果先检查是否含原文/敏感数据，再把路径加到 `scripts/repository_results.txt`。
4. 执行 `python3 scripts/refresh_repository.py`，更新结果清单和本地资产哈希。
5. 执行 `python3 scripts/check_repository.py` 并运行改动所需的实际测试。
6. `git add` 本次相关文件，检查 `git diff --cached --stat` 和差异，创建说明目的的 commit，再 `git push origin main`。
7. 核对本地 HEAD 与远端 main 一致，将仓库网址和 commit SHA 给 ChatGPT；要求重新读取最新文件。

不要使用 `git add -f` 绕过数据忽略规则。不要把历史分数更新成未经运行的新数字。推送代码不等于 ChatGPT 已读取或重新分析。
