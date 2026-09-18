# 复现层级

## 1. 无数据的代码检查

```sh
python3 scripts/check_repository.py
```

仅标准库，解析纳入 Git 的 Python 文件并检查文件范围/大小；不训练、不证明实验分数。

## 2. 有原项目资产的实验与 demo

使用[当前四小时 README](../outputs/stock_nextgen_4h/README.md)的统一入口和[环境版本](../outputs/stock_nextgen_4h/environment.lock.txt)。运行前先合法恢复 `docs/LOCAL_ARTIFACTS.json` 所列本地资产并核验来源。编辑 `config.json` 使用全新的版本目录；所有生产阶段默认拒绝覆盖。

原代码和文档保留原本绝对路径/历史指纹。这是实验档案，还不是安装即用的通用 Python 包；克隆到其他路径时必须处理合同迁移。旧报告指向未上传资产的链接属于本地证据引用，不能当作 GitHub 可访问链接。

实际回归测试命令（需要已有依赖）：

```sh
${PYTHON} outputs/stock_nextgen_4h/test_contract.py -v
```

随后按 README 运行 `run_all.py`。验证保存权重、全量原始标签与本地Qwen推理仍需要私有资产；不能只凭语法检查宣称完成这些核验。早期 `stock_adaptive_4h` 的环境与命令仍保留在其原目录，仅用于复现对应历史实验。


### Latest finite goal60 round

See `outputs/stock_goal60_4h/README.md`. Run `work/stock-data/finbert-env/bin/python outputs/stock_goal60_4h/run_all.py` with the private source artifacts available. The separate TabPFN runtime versions and exact checkpoint revision are in `v1/environment.json`. `replay_model.py T_A1_interaction` verifies an independently reloaded private model without retraining. Raw data and weights are intentionally not distributed.
