# 复现层级

## 1. 无数据的代码检查

```sh
python3 scripts/check_repository.py
```

仅标准库，解析纳入 Git 的 Python 文件并检查文件范围/大小；不训练、不证明实验分数。

## 2. 有原项目资产的实验与 demo

使用[四小时 README](../outputs/stock_adaptive_4h/README.md)的命令和[环境版本](../outputs/stock_adaptive_4h/environment.lock.txt)。运行前先恢复 `docs/LOCAL_ARTIFACTS.json` 对应资产并核验来源。不要覆盖历史目录，使用新的 run 目录。

原代码和文档保留原本绝对路径/历史指纹。这是实验档案，还不是安装即用的通用 Python 包；克隆到其他路径时必须处理合同迁移。旧报告指向未上传资产的链接属于本地证据引用，不能当作 GitHub 可访问链接。

实际回归测试命令（需要已有依赖）：

```sh
work/stock-data/finbert-env/bin/python -m unittest discover -s outputs/stock_adaptive_4h -p 'test_*.py' -v
```

验证保存权重与全量原始标签还需要本地资产；不能只凭语法检查宣称完成这些核验。
