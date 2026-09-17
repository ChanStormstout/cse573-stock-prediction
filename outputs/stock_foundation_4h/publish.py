"""Render measured results; keep historical reports and demo intact."""
import json,re
import numpy as np,pandas as pd
from experiment import ROOT,HERE,PRIVATE,OUT,dump,inputs,sha

def table(m,methods):
 rows=['| 方法 | AAPL开发 BA / Brier | AAPL后续 | AMZN开发 | AMZN后续 |','|---|---|---|---|---|']
 for method in methods:
  cells=[]
  for symbol,phase in [('AAPL','development'),('AAPL','later'),('AMZN','development'),('AMZN','later')]:
   r=m[(m.method==method)&(m.symbol==symbol)&(m.phase==phase)].iloc[0];cells.append(f'{r.BA:.2%} / {r.Brier:.4f}')
  rows.append('| '+method+' | '+' | '.join(cells)+' |')
 return '\n'.join(rows)

def main():
 pd.read_csv(OUT/'chronos_alignment.csv').drop(columns=['anchor']).to_csv(OUT/'chronos_time_audit.csv',index=False)
 m=pd.read_csv(OUT/'all_metrics.csv');d=pd.read_csv(OUT/'all_predictions.csv');sel=json.loads((OUT/'selection.json').read_text());modern=json.loads((OUT/'modern_inference.json').read_text());chrono=json.loads((OUT/'chronos_inference.json').read_text())
 texts=['# 固定正则归因、Fin-ModernBERT与Chronos-2四小时实验','',
 '**已实际运行全部三个有限实验。AMZN 57.32%的原方向增益可由较强正则化在文章等权模型中重现；Fin-ModernBERT改善AMZN但损害AAPL开发期；Chronos-2未显示稳定方向优势。两个新模型均未通过训练期门槛，因此不做额外融合或微调。**','',
 '## 实际执行与协议','',
 '用户授权本轮在上轮停止后继续有限探索，先保存PRE_REGISTRATION.md；没有改写上轮门槛。保留1,607窗口，233 warmup、765前向OOF、252 development、357 later。时间截止仍是目标开始前5分钟，目标仍是四小时开盘到收盘方向。既有评价期全部暴露。','',
 f'- 新训练208个LR分类器：固定C归因56个、匹配旧FinBERT38个、Modern38个、Chronos与等历史长度对照76个。记录每次参数、边界、耗时、权重哈希和重载误差。',
 f'- Fin-ModernBERT：冻结{modern["parameters"]:,}参数，在M5 MPS上编码5,078标题，{modern["seconds"]:.2f}秒；无梯度、无编码器训练。',
 f'- Chronos-2：冻结{chrono["parameters"]:,}参数，CPU处理1,607个双变量历史窗口，{chrono["seconds"]:.2f}秒；全部满足输入条件，未因失败删除样本。',
 '- LR拟合耗时在记录中不包含前面的PCA/特征准备；不能把短拟合耗时称为整个模型重训耗时。没有使用Sol、付费API或GPU服务器。',
 '- 原F2经同一训练入口重现，最大概率差1.73e-15。全部208个模型重载通过，Modern首批重载向量差0，Chronos单任务重载预测差0。','',
 '## 1. 原57.32%的来源：固定C四组对照','',
 'A01/A1=文章等权＋相同元信息，C分别0.01/1；E01/E1=近似转载组等权＋相同元信息，C分别0.01/1。C越小，正则化越强。两种聚合共享原始文章集合、PCA16和选定价格特征。','',table(m,['A01','A1','E01','E1']),'',
 'AMZN后续A01与E01都为57.32%，179个方向判断逐个完全相同；Brier分别0.248103/0.248316。在C=1时，文章等权54.74%，事件聚合54.21%。因此本数据上，原N1M相对N0M的后续方向增益不需要事件聚合就能重现，不能继续称为去重独立贡献。',
 '', '但这不等于“强正则已稳定提高方向”。训练六个月平均，文章分支C从1降至0.01，AAPL/AMZN BA分别下降0.66/0.69个百分点，同时Brier改善0.0542/0.0512。强正则更稳定地改善概率尺度，方向增量仍随时期改变。固定C=0.01的事件相对文章聚合，训练月均BA仅AAPL−0.04、AMZN+0.53个百分点；未形成明显共同收益。','',
 '## 2. 两个新模型与既有方法','',table(m,['F0','R1','F1','F2','F6','MODERN','HISTORY_LR','CHRONOS_LR']),'',
 'F0=标题baseline；F1=全文；F2=原冻结FinBERT；MODERN=只替换为冻结Fin-ModernBERT；HISTORY_LR=相同512根五分钟开盘/收盘历史的线性分类器；CHRONOS_LR=Chronos预测形成3个特征，再经训练期LR学习方向概率。两种Chronos相关分类器都有独立训练期选C，不能把Chronos LR称为零样本概率输出。','',
 '### Fin-ModernBERT：这是编码器替换实验','',
 '精确使用原F2的相同标题与文章集合、attention-mask平均池化、256 token预算、文章等权、训练内PCA16和下游LR。没有新增目标公司前缀、正文、事件抽取或微调。这是为了分离编码器变化；尚未验证长全文或任务适配版Fin-ModernBERT。5,078篇标题在新tokenizer中均未达到截断上限。',
 '', 'Modern相对F2：AMZN开发46.29%→54.82%、后续54.27%→56.03%；AAPL开发54.13%→45.51%、后续56.71%→54.52%。单看AMZN有正向观察，但不能依据这些已暴露分数给两股分配不同模型，更不能声称普遍替代F2。','',
 '### Chronos：预测目标与时间处理','',
 '只输入截止前已完成的regular-session五分钟open/close序列，最多512根。用交易步序列移除隔夜非交易区间，但保留缺失交易bar为NaN；这不是均匀日历时间建模。截止点在开盘前时，目标开盘/末尾对应未来第1/48步；其他窗口第2/49步，避免偷看未知开盘或错误跳过五分钟间隔。',
 '', '两个变量在同一任务内共享信息，不同股票/窗口cross_learning=False，避免批次内跨时间信息。仅用预测开盘/末收盘中位差、两端10%–90%分位宽度三个特征；没有假设两个边际分位数构成联合分布，也没有把它们直接算成上涨概率。',
 '', '原始中位差方向（无概率映射）的BA：AAPL开发52.99%、后续48.12%；AMZN开发52.32%、后续49.31%。完整记录见chronos_raw_direction.csv，不为原始中位差编造Brier。',
 '', '加入LR后，AAPL开发全部预测上涨（BA50%），后续178个中177个预测上涨（BA50.55%）；AMZN后续BA47.01%。概率更温和不等于能分清方向。相同长历史的LR也不稳定且AAPL Brier很差，所以“比长历史LR概率好”不足以证明Chronos有用。','',
 '## 3. 训练期统一决策','',
 '| 新方法 / 匹配基线 | 月均BA差 | 月均Brier差 | 正增量月份 | AAPL BA差 | AMZN BA差 | 晋级 |','|---|---:|---:|---:|---:|---:|---|']
 for g in sel['gates']:texts.append(f'| {g["method"]}/{g["baseline"]} | {100*g["BA_delta"]:+.2f} pp | {g["Brier_delta"]:+.4f} | {g["positive_months"]}/{g["months"]} | {100*g["stocks"]["AAPL"]["BA"]:+.2f} pp | {100*g["stocks"]["AMZN"]["BA"]:+.2f} pp | 未通过 |')
 texts+=['','Modern宏平均方向略降、AAPL BA下降3.56个百分点且Brier恶化0.0270。Chronos相对同历史LR宏平均BA下降0.64个百分点，虽然Brier改善。两者均未达到门槛；fusion_selection.json为空，实际没有运行融合或继续扩大网格。每股C可以从自身过去训练选取，但模型结构不根据每股后续成绩挑选。','',
 '## 4. 观察、解释与限制','',
 '- **观察：** 强正则文章模型重现AMZN聚合方向成绩；Modern改善AMZN而损害AAPL；Chronos中位预测与下游概率分类器均不稳定。',
 '- **解释：** 先前部分局部提升可由收缩减少过拟合解释；预训练特征是否有用取决于具体语料和股票，模型更新并非自动更好。Chronos下游接近恒定预测，提示3个预测特征的区分能力有限。',
 '- **尚未证明：** 股票本身完全不可预测；Modern长正文无用；所有Chronos输入设计/微调均无用。这里只排查预注册的具体低成本配置。',
 '- Model card声明Modern金融预训练包含FNSPID等语料；Chronos也在历史真实序列上预训练。预训练与2018样本的重叠无法排除，因此只能称回顾性课程实验；时序下游切分不能消除此风险。',
 '- 1日/5日块配对区间已保存；两新方法相对各自对照的后续BA区间均跨0。它们不是经过反复探索选择校正的显著性检验。',
 '- 独立人工事件抽取验收仍未完成；没有增加新保留时期。','',
 '## 5. 工程检查与恢复','',
 '72行BA/MCC/Brier独立重算、全部标签/键、训练时间边界、过去选参、无新闻回退、checkpoint重载、Chronos目标第48/49步通过。将截止后尚未完成的五分钟gap bar改为100倍后，输入完全不变。cache失配被拒绝。',
 '', '启动Chronos时发现缺少accelerate，安装到独立runtime后解决。后续发现本地缓存键由Pandas产生object dtype，NumPy默认安全加载拒绝；已将52个缓存的键迁移为Unicode，原始文件与执行源码均归档，预测值逐元素未变。cache_repair.json记录前后哈希，恢复读取复测通过。原运行证据保留原源码哈希，不能将修复说成未出现过。',
 '', '## 6. 交付与复现','',
 '[运行说明](../README.md)、[协议](../PRE_REGISTRATION.md)、[完整指标](all_metrics.csv)、[逐月](all_monthly.csv)、[配对区间](paired_intervals.csv)、[案例](CASE_NOTES.md)、[离线demo](demo.html)。旧报告/结果保留，新报告不追溯修改旧选择。',
 '', '官方来源：[Fin-ModernBERT模型卡](https://huggingface.co/clapAI/Fin-ModernBERT)、[Chronos-2模型卡](https://huggingface.co/amazon/chronos-2)、[官方实现](https://github.com/amazon-science/chronos-forecasting)。两者模型卡均标Apache-2.0；原课程新闻的分发权限没有改变。']
 (OUT/'REPORT.md').write_text('\n'.join(texts)+'\n')
 cases=json.loads((OUT/'cases.json').read_text());notes=['# 固定规则抽样的案例核对','', '按预注册类别内样本键SHA选取，共16槽位，重复窗口不算新的独立案例。均是已暴露后续期描述性案例，不用于调参。新闻只核对原始标题与元信息，不冒充完整正文分析或独立人工验收。','', '| 股票/模型 | 类别 | 窗口UTC | 旧→新上涨概率 | 实际 |','|---|---|---|---|---|']
 for c in cases:notes.append(f'| {c["symbol"]}/{c["method"]} | {c["category"]} | {c["key"].split("|")[1]} | {c["base_probability"]:.4f}→{c["probability"]:.4f} | {"上涨" if c["label"] else "下跌"} |')
 notes+=['','## 新闻与数值核对','',
 '- AAPL 12月26日16:30：标题含技术股前景、iPhone市场份额、买入观点等，Modern从0.380改为0.528，真实上涨，修正成功。但不能从一次正确判定证明它读懂了哪些因果信息。',
 '- AAPL 1月14日15:30：估值买入观点与中国出口/供应商受挫等混合标题，Modern从0.478改为0.548，真实下跌，改错。不同语气和时间尺度仍可能混合。',
 '- AAPL 11月16日15:30：持仓模板、未来iPhone升级、反弹观点等；两个模型均预测上涨，真实下跌。现代编码器没有自动解决信息时效。',
 '- AMZN 1月8日14:30：三年股价观点与竞争新闻，Modern从0.506改为0.451，真实下跌。即使改对，也不能将多年观点直接解释成四小时信号。',
 '- AMZN 1月3日14:30：两条会员信息披露标题实为同一表达，区别是HTML撇号编码；Modern从0.181改为0.550，真实下跌。更换编码器不能替代重复/来源处理；这里没有事后更改过滤规则。',
 '- AMZN 1月14日14:30：无新闻，Modern与F2严格同为R1的0.644704，说明共同正确不是新编码器的贡献。',
 '- Chronos AAPL 12月26日16:30：原始预测中位差为−0.716%，但LR输出上涨0.563且碰巧改对。该成功不能称为Chronos价格中位预测正确。',
 '- Chronos AAPL 11月26日15:30：中位差−0.481%，LR仍输出上涨0.562、真实下跌；与AAPL接近恒定上涨的总体情况相符。',
 '- Chronos AMZN 11月9日16:30：预测中位差−1.932%，上涨概率0.349，真实上涨，改错。价格任务失败与新闻抽取错误不是一回事。','',
 '完整改对/改错/共同错误计数见transitions.csv。上述内容是观察，标题混杂、时间尺度错配、下游截距偏向只是可能解释，不是已经分解的市场因果贡献。']
 (OUT/'CASE_NOTES.md').write_text('\n'.join(notes)+'\n')
 # New standalone demo derives from prior audited interface, with new result payload.
 demo=d[d.phase.isin(['development','later'])].copy();source,*_=inputs();meta=source.set_index('key')
 for k in ['articles','clusters']:demo[k]=demo.key.map(meta[k])
 demo['start']=demo.key.str.split('|').str[1];demo['cutoff']=(pd.to_datetime(demo.start,utc=True)-pd.Timedelta('5min')).astype(str);demo['end']=(pd.to_datetime(demo.start,utc=True)+pd.Timedelta('4h')).astype(str)
 payload=demo.to_json(orient='records');html=(ROOT/'outputs/stock_combination_4h/v1/demo.html').read_text();a=html.index('const data=')+len('const data=');b=html.index(';const $=',a);html=html[:a]+payload+html[b:]
 methods=[['F0','标题baseline'],['R1','近期价格'],['F1','价格＋全文'],['F2','价格＋FinBERT'],['MODERN','价格＋Fin-ModernBERT'],['HISTORY_LR','512根历史＋LR'],['CHRONOS_LR','Chronos-2预测＋LR'],['A01','文章均值 C=0.01'],['E01','事件均值 C=0.01']]
 a=html.index('const methods=')+len('const methods=');b=html.index(';\nfunction choices',a);html=html[:a]+json.dumps(methods,ensure_ascii=False)+html[b:]
 html=html.replace('价格、新闻与预测信心','新编码器、时间序列与正则化').replace('无新闻：严格回退R1','无新闻：文本模型回退R1；Chronos仍使用价格').replace('温度使用对应模型过去OOF拟合；无新闻保持R1。','新编码器和Chronos冻结推理；下游分类器仅用过去数据训练。无新闻只有文本分支回退R1。').replace('href="metrics.csv"','href="all_metrics.csv"');(OUT/'demo.html').write_text(html)
 print('Wrote report, cases and new 609-window replay')
if __name__=='__main__':main()
