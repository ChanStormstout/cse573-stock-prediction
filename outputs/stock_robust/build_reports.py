from pathlib import Path
import json,numpy as np
O=Path(__file__).resolve().parents[2]/'outputs'
def read(p):return json.loads((O/p).read_text())
L=read('stock_robust/results/linear/results.json');T=read('stock_robust/results/temporal/results.json');old=read('stock_temporal/results/results.json');C=read('stock_calibration/results/results.json');E=read('stock_text_regularization/results/results.json')
names={'price':'价格 LR','text':'标题 TF-IDF LR','combined':'价格＋标题 TF-IDF','price_count':'价格＋新闻数量','finbert_only':'FinBERT 情绪＋数量','price_finbert':'价格＋FinBERT 情绪','paper_stem_title':'词干标题＋卡方＋L1','paper_stem_body':'词干全文＋卡方＋L1'}
lines=['# E09：统一基线、全文词特征、参数共享与训练期早停','', '2026-09-15，实际实验完成。所有表格由保存的 JSON 结果生成。','',
'## 结论','',
'全文词特征在 AAPL 开发集取得更明显提升：BA 59.44%，相同预处理标题为 52.72%，相差 +6.72 个百分点。AMZN 全文为 47.59%，比标题更差。共享参数与早停没有形成跨股稳定的方向提升；早停明显改善概率误差。','',
'这是开发集发现。全文 AAPL 训练期向前验证均值只有 51.12%，低于标题 55.60%；此前 Sep–Oct 已用于方案设计，不能把本轮分数或重采样区间当作独立最终结论。','',
'## 固定的数据和评价','',
'- AAPL 996/252、AMZN 1004/252 个训练/开发窗口；两股共 2,000 个训练窗口，504 个开发窗口。没有筛掉无新闻窗口。',
'- 2018 年 1–8 月训练；在六月、七月、八月三个向前折内选择 C 或权重衰减，所有词表、卡方选择、缩放只在对应训练部分拟合。9–10 月只在选定配置后评分。',
'- 目标为一个完整常规交易小时 C>O，收益数值存储为 C/O−1；预测截止为目标开始前五分钟。历史 log-return 设计文字与简单收益符号相同，但本报告按实际代码描述。',
'- BA 是上涨召回率和下跌召回率的平均；50% 为恒定方向参照。Brier 是预测概率与 0/1 结果的均方差，越低越好；恒定 0.5 概率为 0.25。MCC 越大越好。',
'- 最终测试 2018 年 11 月至 2019 年 2 月初保持未评价。','',
'## 1. 统一选参后的基线','',
'| 模型 | AAPL BA | AMZN BA | AAPL Brier | AMZN Brier | AAPL MCC | AMZN MCC |','|---|---:|---:|---:|---:|---:|---:|']
for kind,label in names.items():
 a=L['stocks']['AAPL'][kind]['validation']['overall'];b=L['stocks']['AMZN'][kind]['validation']['overall'];lines.append(f'| {label} | {a["balanced_accuracy"]:.2%} | {b["balanced_accuracy"]:.2%} | {a["brier"]:.4f} | {b["brier"]:.4f} | {a["mcc"]:.3f} | {b["mcc"]:.3f} |')
lines+=['','这是对 Review 第一项的实际处理：不沿用早期“同一个验证集挑 C 再报分”的成绩作为公平对照。价格与价格＋FinBERT 的预测还与 E04 版本逐项复现一致。','',
'## 2. 全文实验的逻辑、观察与限制','',
'**改了什么：** 标题和全文均使用 Snowball 英语词干化、停用词去除、二值 unigram；训练折内 min_df=3、最多 10,000 词、卡方前 500 词，再加同样的价格特征训练 L1 LR。全文保留原标题并对正文执行固定 HTML/URL/模板行清理。只改文字范围的对照比直接拿全文与旧 FinBERT 分数相比更能解释差异。','',
'这是对老师 Alostad & Davulcu 新闻分支的进一步借鉴。没有实现原文完整两阶段卡方流程或历史 Twitter 突增系统，不能称全文复现。','',
'| 股票/表示 | C | 训练 BA | 训练期 CV BA | 九月 BA | 十月 BA |','|---|---:|---:|---:|---:|---:|']
for s in ['AAPL','AMZN']:
 for k in ['paper_stem_title','paper_stem_body']:
  r=L['stocks'][s][k];m=r['validation']['months'];lines.append(f'| {s} / {names[k]} | {r["C"]:g} | {r["training"]["balanced_accuracy"]:.2%} | {r["training_cv_BA"]:.2%} | {m["2018-09"]["balanced_accuracy"]:.2%} | {m["2018-10"]["balanced_accuracy"]:.2%} |')
lines+=['','AAPL 全文在两个开发月份均超过标题，但训练/开发差距仍大。其原始 Brier 0.2868，说明方向分数较高与概率可靠性是两回事。','',
'对“全文−标题”做 2,000 次按共同交易日成对重采样：AAPL BA 差的描述性 95% 区间为 [+0.45,+13.20] 个百分点，AMZN 为 [−8.46,+3.71]；两股等权平均为 [−2.27,+6.91]。五个相邻交易日的移动块敏感性分析中，两股平均仍跨零。没有对反复探索做多重比较校正，也没有估计训练样本和选模型的全部不确定性。','',
'[全部 504 个窗口](results/all_504_body_comparisons.csv)、[32 张分层诊断卡](CASES_32.md)、[8 个重点案例复核](QUALITATIVE_REVIEW.md) 显示：AAPL 60 改对/44 改错，AMZN 33/38。作者、媒体名称、其他公司与旧财务背景仍会影响模型。不能把预测正确包装为理解或因果推断正确。','',
'## 3. 参数共享，以及它与图平滑的关系','',
'动机：每股只有约千个训练窗口，允许两股借用共同规律，同时限制每股修正。输入为同样的 22 个当前价格/情绪/数量特征；两股按日期一起切分。三种模式均用 pooled training scaler、相同 C 网格、正则化的股票截距。','',
'| 斜率模式 | AAPL BA | AMZN BA |','|---|---:|---:|']
for mode,label in [('separate','独立'),('shared','完全共享'),('partial','共享＋各股偏移')]:
 r=L['sharing'][mode];lines.append(f'| {label} | {r["stocks"]["AAPL"]["overall"]["balanced_accuracy"]:.2%} | {r["stocks"]["AMZN"]["overall"]["balanced_accuracy"]:.2%} |')
lines+=['','这些共享对照需要彼此比较：其缩放和截距正则方式与单股基线有区别，不能将全部差异归于参数共享。两股平均的描述性区间均跨零。','',
r'部分共享写成 $w_A=w_0+\delta_A,\ w_M=w_0+\delta_M$，使用相同 L2 惩罚。消去公共参数，得到：','',
r'$$\min_{w_0}\bigl(\|w_0\|^2+\|w_A-w_0\|^2+\|w_M-w_0\|^2\bigr)=\tfrac13\bigl(\|w_A\|^2+\|w_M\|^2+\|w_A-w_M\|^2\bigr).$$','',
'最后一项惩罚两个股票参数差太远，正是两节点图的最简单平滑正则思想。本轮已经检验了这种轻量关系约束；结果没有明显支持它。它没有图神经网络的消息传递，也不是因果图或完整 CausalStock。','',
'## 4. 神经网络早停：改善了什么','',
'在每个训练前缀内，留最后 20 个交易日监控 BCE，最多 40 轮，patience=5；选出轮数后重置种子并用完整前缀训练。外层验证月份和 Sep–Oct 没有用于选择轮数。三种子 573/574/575 全部报告。E08 六快照输入、时间间隔和掩码不变；除原参考宽度外补充 GRU hidden=4。','',
'| 股票 | 模型 | 训练 BA 均值 | 开发 BA 均值±标准差 | 开发 Brier | 最终训练轮数 |','|---|---|---:|---:|---:|---|']
for s in ['AAPL','AMZN']:
 for kind,label in [('mlp','旧 MLP，40 轮'),('gru','旧 GRU，40 轮')]:
  r=old['stocks'][s]['models'][kind];mean=np.mean([q['training']['balanced_accuracy'] for q in r['seeds'].values()]);lines.append(f'| {s} | {label} | {mean:.2%} | {r["seed_mean"]["balanced_accuracy"]*100:.2f}±{r["seed_std"]["balanced_accuracy"]*100:.2f}% | {r["seed_mean"]["brier"]:.4f} | 40,40,40 |')
 for kind,label in [('mlp_early_h8','早停 MLP'),('gru_early_h8','早停 GRU h8'),('gru_early_h4','早停 GRU h4')]:
  r=T['stocks'][s][kind];mean=np.mean([q['training']['balanced_accuracy'] for q in r['seeds'].values()]);epochs=','.join(str(q['epochs']) for q in r['seeds'].values());lines.append(f'| {s} | {label} | {mean:.2%} | {r["seed_mean"]["balanced_accuracy"]*100:.2f}±{r["seed_std"]["balanced_accuracy"]*100:.2f}% | {r["seed_mean"]["brier"]:.4f} | {epochs} |')
lines+=['','“±”是三个种子的样本标准差（百分点），不是置信区间；没有挑最好种子。Brier 的改善支持修正原来过度训练的问题，但 BA 仍接近 50%，也没有超过同信息量的六快照 LR。这不证明神经模型都无效；它说明这次训练控制修正尚不足以获得稳定方向收益。','',
'## 5. 检查、保存与下一步','',
'[协议](protocol.json) 在 E09 训练前固定。[check.py](check.py) 实际检查了所有保存权重、预测/指标、参数选择、早停轮数、训练缩放、源指纹及原对照复现，204 个历史产物保持原哈希。结果见 [check_result.txt](check_result.txt)。','',
'第一次准备步骤因时间字符串混用有/无小数秒而退出，发生在任何模型训练之前。该失败现场保存在 `attempt_01_input_parse/`，修复为 mixed 时间解析后在干净目录准备。没有删除失败实验，也没有覆盖旧结果。','',
'后续已另立协议执行 [E09-C 温度校准](../stock_calibration/README.md)、[E11 L1/L2 对照](../stock_text_regularization/README.md)，FinBERT 微调见 [E10](../stock_finetune/REPORT.md)。这些后续设计受 E09 观察影响，属于自适应开发探索。','',
'复核命令（项目根目录）：','',
'```bash','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_robust/check.py','```','',
'现有结果目录不能覆盖；新训练需要先建立新的版本目录和缓存位置。运行源代码、训练网格、权重、逐样本概率和协议均保存在本目录，数据/缓存来源由 prepared.json 锁定。']
(O/'stock_robust/REPORT.md').write_text('\n'.join(lines)+'\n')
(O/'stock_robust/README.md').write_text('# E09 实验入口\n\n详见 [完整报告](REPORT.md)、[协议](protocol.json)、[检查结果](check_result.txt)、[案例复核](QUALITATIVE_REVIEW.md)。\n\n训练脚本为 prepare.py → run_linear.py / run_temporal.py；结果目录受到防覆盖保护。检查已有模型使用 check.py；新训练需另建版本并先固定输出/缓存位置。\n')

lines=['# E09-C：训练期样本外温度校准','','2026-09-15，实际完成。此实验在看到 E09 全文 AAPL 的高 BA 与高 Brier 后设计，训练前保存协议；不冒充在 E09 之前预先声明。','','## 为什么做、怎样做','',
'方向可能判断正确，却报得过于自信。温度校准只改变概率强弱：例如 T>1 将 .90 拉向 .50，不改变上涨/下跌方向。', '',r'$$p_T=\sigma(z/T),\quad T>0,$$','',
'其中 z 是原 LR 的 logit。T 用训练期样本外预测的 BCE 拟合，范围预设为 [.25,10]。T>1 降低置信度，T<1 增强置信度；没有根据开发集选择范围、温度或是否应用。','',
'严格生成样本外校准数据：六月/七月/八月各自再用此前三个月向前验证选 C，然后只用该月之前的数据训练，预测该月。合并这些样本外 logit 拟合 T。最后应用于 E09 用全 Jan–Aug 拟合的原标题/全文模型，不改变其权重。内层早期 AMZN 词表小于 500 时，SelectKBest 保留全部可用词，这是预期行为。','',
'该过程借鉴 [Guo 等，2017](https://proceedings.mlr.press/v70/guo17a.html) 的温度校准；本项目另外使用嵌套时间划分。样本外各折的 C 可以不同，故这里校准的是训练流程，不能保证温度对最终拟合或后续月份完全可迁移。','',
'## 全部四组结果','','| 股票 | 输入 | T | BA（不变） | Brier：原始→校准 |','|---|---|---:|---:|---:|']
for s in ['AAPL','AMZN']:
 for kind,label in [('paper_stem_title','标题'),('paper_stem_body','全文')]:
  r=C['stocks'][s][kind];a=r['uncalibrated']['overall'];b=r['calibrated']['overall'];lines.append(f'| {s} | {label} | {r["temperature"]:.3f} | {b["balanced_accuracy"]:.2%} | {a["brier"]:.4f} → {b["brier"]:.4f} |')
lines+=['','AAPL 全文 Brier 降到 0.2470、BA 保持 59.44%。BA 提升来自全文表示，不是校准。AMZN 两种输入的 Brier 都变差，说明训练期校准关系也可能无法迁移到开发月份；不能只展示有利一股。T 是否改善分数必须评估，不能默认。','',
'## 检查与复现','','[check.py](check.py) 重训了 12 个外层样本外模型并重现 logit，核验内层 C 选择记录、时间边界、重新优化 T、最终概率、全部指标和方向判断完全不变；[检查结果](check_result.txt) 已通过。原 E09 权重与全部旧结果保持不变；最终测试未评价。','','```bash','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_calibration/check.py','```','',
'[协议](protocol.json)、[结果](results/results.json)、各股 nested_oof/inner_selection CSV、最终逐窗口概率均保存。不要事后只对 AMZN 关闭校准再把它称为本实验的预先规定系统。']
(O/'stock_calibration/README.md').write_text('\n'.join(lines)+'\n')

lines=['# E11：相同文本特征，比较 L1 与 L2','','2026-09-15，实际完成。看到 E09 的训练/开发差距和来源词贡献后，登记了这个有限的自适应对照；没有扩大 C 网格或改验证样本。','','## 设计与直觉','','L1 倾向把一些系数置零；L2 倾向把各系数整体缩小。全文模型存在大系数和很多潜在杂质词，因此检验换一种约束能否更好泛化。L2 并不保证更少过拟合，尤其在多余特征很多时。','','输入、词干化、二值词、训练折卡方筛选、价格缩放和 liblinear solver 全部与 E09 一致，只改惩罚为 L2。标题/全文两种表示、两只股票都运行；每组依旧只有 C=.01/.1/1，训练期三个月向前验证选参。','','## 结果','','| 股票/输入 | L1 BA | L2 BA | L1 Brier | L2 Brier | L2 训练 BA | L2 选择 C |','|---|---:|---:|---:|---:|---:|---:|']
for s in ['AAPL','AMZN']:
 for kind,label in [('paper_stem_title','标题'),('paper_stem_body','全文')]:
  a=L['stocks'][s][kind]['validation']['overall'];r=E['stocks'][s][kind];b=r['validation']['overall'];lines.append(f'| {s}/{label} | {a["balanced_accuracy"]:.2%} | {b["balanced_accuracy"]:.2%} | {a["brier"]:.4f} | {b["brier"]:.4f} | {r["training"]["balanced_accuracy"]:.2%} | {r["C"]:g} |')
lines+=['','四组 L2 的 BA 都没有超过相应 L1。AAPL 全文 Brier 也更差；AMZN 的概率误差有所降低，但方向更差。这个有限网格没有支持替换 L1。不能推论所有 L2 超参数都无效，也不能将相同 C 解释为相同有效模型复杂度。','','## 验证与决定','','[check.py](check.py) 验证最终词特征集合与 E09 对应 L1 完全相同、选 C 记录一致、缩放均值来自训练、保存权重重现预测与全部指标；按共同交易日做了 2,000 次描述性重采样。结果见 [检查](check_result.txt)、[原始指标](results/results.json)。最终测试未评价。','','停止这条分支，保留负结果。不要在当前开发分数基础上继续细分 C 来追逐更高成绩。','','```bash','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 work/stock-data/finbert-env/bin/python outputs/stock_text_regularization/check.py','```']
(O/'stock_text_regularization/README.md').write_text('\n'.join(lines)+'\n')
print('Wrote E09, E09-C and E11 reports.')
