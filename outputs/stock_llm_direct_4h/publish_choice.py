"""Publish completed binary-choice experiment and candid partial JSON history."""
import argparse,csv,json,shutil
from pathlib import Path
from common import B,W,dump,rows,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--analysis',required=True,type=Path);ap.add_argument('--run',required=True,type=Path);ap.add_argument('--baseline',required=True,type=Path);ap.add_argument('--inputs',required=True,type=Path);a=ap.parse_args()
 summary=json.loads((a.run/'summary.json').read_text());assert summary['completed'] and summary['interface']=='binary_choice_logprobs'
 for src,dst in [('metrics.csv','METRICS.csv'),('transitions.csv','TRANSITIONS.csv'),('paired_intervals.csv','PAIRED_INTERVALS.csv'),('quality.json','QUALITY.json'),('verification.json','VERIFICATION.json'),('case_selection.json','CASE_SELECTION.json'),('coverage_summary.json','COVERAGE.json'),('predictions.csv','PREDICTIONS.csv')]:shutil.copy2(a.analysis/src,B/dst)
 shutil.copy2(a.baseline/'cv.csv',B/'BASELINE_CV.csv');shutil.copy2(a.baseline/'summary.json',B/'BASELINE_TRAINING.json')
 dump(B/'EXECUTION.json',dict(manifest=json.loads((a.run/'manifest.json').read_text()),summary=summary,input_check=json.loads((a.run/'input_check.json').read_text())))
 seal=json.loads((a.inputs/'manifest.json').read_text());dump(B/'INPUT_VERIFICATION.json',{k:seal[k] for k in ['n','split_counts','input_sha','features_sha','coverage_sha','historical_price_and_label_parity','protocol_sha']})
 history={}
 for name in ['smoke_v1','run_v1','batch_benchmark_v1','batch_v1','choice_smoke_v1','choice_smoke_v2']:
  dr=W/'direct_4h'/name;stats={}
  for path in dr.glob('*.jsonl'):
   rr=rows(path);stats[path.stem]=dict(n=len(rr),seconds=sum(r['seconds'] for r in rr),forecast_file_sha=sha(path),valid=sum(r['valid'] for r in rr))
  history[name]=dict(branches=stats,complete=(dr/'summary.json').exists(),interruption=json.loads((dr/'interruption.json').read_text()) if (dr/'interruption.json').exists() else None)
 shutil.copy2(W/'direct_4h/price_behavior_diagnostic.json',B/'PRICE_BEHAVIOR.json')
 shutil.copy2(W/'direct_4h/json_interface_diagnostic.json',B/'JSON_INTERFACE_DIAGNOSTIC.json')
 shutil.copy2(W/'direct_4h/json_price_results.json',B/'JSON_PRICE_RESULTS.json')
 shutil.copy2(W/'direct_4h/news_absence_diagnostic.json',B/'NEWS_ABSENCE.json')
 history['batch_parity']=json.loads((W/'direct_4h/batch_benchmark_v1/parity.json').read_text());dump(B/'ENGINE_HISTORY.json',history)
 scores=list(csv.DictReader((B/'METRICS.csv').open()));names={'legacy_title':'原价格＋标题baseline','legacy_body':'旧价格＋全文','legacy_semantic':'旧价格＋FinBERT','legacy_integrated':'旧完整组合','matched_price':'相同历史价格LR','matched_news':'相同新闻片段LR','matched_joint':'相同价格＋新闻片段LR','llm_price':'LLM看价格','llm_news':'LLM看新闻','llm_joint':'LLM看价格＋新闻'}
 def fmt(x):return f'{float(x)*100:.2f}%'
 def table(period):
  out=['| 方法 | AAPL BA | AMZN BA | AAPL Brier | AMZN Brier |','|---|---:|---:|---:|---:|']
  for name,title in names.items():
   rr=[next(r for r in scores if r['method']==name and r['symbol']==s and r['period']==period and r['scope']=='all') for s in ['AAPL','AMZN']]
   out.append(f"| {title} | {fmt(rr[0]['BA'])} | {fmt(rr[1]['BA'])} | {float(rr[0]['Brier']):.4f} | {float(rr[1]['Brier']):.4f} |")
  return '\n'.join(out)
 quality=json.loads((B/'QUALITY.json').read_text());q=['| 输入 | 完成窗口 | 最小/中位选择概率质量 | 原自由首token不是UP/DOWN | 分钟 |','|---|---:|---:|---:|---:|']
 for v,x in quality.items():q.append(f"| {v} | {x['n']} | {x['min_choice_mass']:.4f} / {x['median_choice_mass']:.4f} | {x['unconstrained_off_label']} | {x['seconds']/60:.1f} |")
 prior_seconds=sum(v['seconds'] for h in history.values() if isinstance(h,dict) for v in h.get('branches',{}).values())
 primary_seconds=sum(v['seconds'] for v in quality.values())
 report=f'''# LLM直接预测四小时：二选一完整实验

## 本轮实际完成

**LLM直接读取价格/新闻并选择上涨或下跌，没有先抽事件再交给分类器。**
固定Qwen3.5-9B 4bit；三组各完成609窗口，共1,827个预测。没有LLM微调。
另外实际完成60次匹配LR拟合；六个最终模型重载概率一致。
准备全部1,607原窗口，预测开发252＋后续357；没有删掉无新闻或低信心窗口。
[当前协议](CHOICE_PROTOCOL.md)；[精确prompt与概率读取代码](choice.py)；[复现入口](README.md)。

## 结果怎么理解

本轮三个LLM方案中，后续时期新闻单独输入最好：AAPL55.65%、AMZN52.54%，
比原价格＋标题baseline高3.76、3.62个百分点；但Brier为0.3684/0.3354，
均比恒定50%概率的0.25更差，开发期和逐月表现也不一致。不能称已找到稳定、可信概率模型。
联合输入为50.08%/51.21%，没有胜过新闻单独输入；AAPL还低于课程baseline。
苹果从新闻到联合改变18次判断，4次改对、14次改错。

相同输入的LR三组、旧完整流水线也完整保留如下，不能只展示较弱参照。
配对区间跨0，且所有时期已暴露；当前证据支持继续诊断，不支持宣布独立泛化成功。

## 为什么更改输出接口？

原[JSON协议](PROTOCOL.md)要求模型自己写p_up及方向。价格组609已完成，新闻组96后停止，
联合组未执行；不能称该版完成完整三组对照。大量数值固定在.5，同时方向可能写DOWN。
这些问题在未将LLM输出与市场答案连接时已观察并登记[诊断](OUTPUT_DIAGNOSTIC.md)。
原记录、失败、成本都保留于[ENGINE_HISTORY.json](ENGINE_HISTORY.json)。
价格609条中582条原始概率为.5，27条JSON无法解析；新闻96条中72条为.5、2条无法解析，
其余已经出现其他概率，因此不能说JSON版所有新闻输入都只会回答50%。
[原接口统计](JSON_INTERFACE_DIAGNOSTIC.json)与[完整价格组原协议/原生方向结果](JSON_PRICE_RESULTS.json)均保留。

于是另立V2：同样输入，要求只选UP/DOWN，程序读取两个答案token的模型log概率，
在这两个允许选项内归一化。不是随机补概率，也不是人工按新闻写规则。
合成smoke还发现原生低精度归一化的概率质量可能略大于1；正式V2前改用float32
对全词表重新归一化再取两选项。它保留相对偏好，修正数值诊断；两版smoke均保存。
**这是受未标注输出启发的接口改进，不是事先从未改动的单一实验，也不是按测试分数选prompt。**

## 给模型的信息

- 原目标：信息截至区间开始前5分钟，预测随后四小时区间结束价相对开盘价的方向；未来开盘价不输入。
- 最近6个完整交易小时的收益/振幅、均值/波动、历史年龄与时段，全部由原始5分钟线重建核对。
- 原新闻候选集合中最近收到的最多6篇，标题最多240字符、正文最多600字符；给出版/可用时间和延迟。
- 不使用抽取成功门槛，不给未来收益、结果标签、已有模型预测或带答案例子。

[输入抽查](INPUT_AUDIT.md)发现字符截取可能截断句子、遗漏目标段落，原候选也有历史持仓、
宣传和重复内容。AAPL后续178窗口中151个受到六篇上限影响，只展示1,031/2,465次文章出现；
AMZN84/179有新闻，全部143次出现均纳入。出现次数不是独立文章数。
大部分早盘窗口的完整小时历史停留在前一交易日；盘中最新完整小时也距截止55分钟。
本轮保持这些信息限制以匹配原任务，不宣称输入已最佳化。

新LR对照使用同样价格源/新闻片段，六月至八月训练期选C；LLM有逐篇时间结构，LR用数值摘要，
内部表示并不相同。旧模型读取更多/不同文本，仅作原项目参照，不能把区别全归因于算法。
旧完整流水线按三月至八月选择C，本轮匹配LR按六月至八月；例如AAPL价格模型
旧C=1、新C=0.1，因此新的价格对照分数不必等于旧价格参照，两个结果均保留。

## 后续时期结果（已暴露）

AAPL178、AMZN179窗口。BA为两类召回率平均值，50%为恒定方向参照；Brier越低越好。
LLM概率是**两个答案token的条件偏好，尚未经市场标签校准**。

{table('test')}

## 开发时期结果

{table('validation')}

分月、MCC、预测上涨比例、恒定预测标记见[METRICS.csv](METRICS.csv)。
[方向改对/改错](TRANSITIONS.csv)包含价格→联合、新闻→联合及匹配对照，
另按有无新闻统计，BA贡献用全体类别分母，
不把删掉无新闻窗口后的子集成绩冒充整体提升。
没有挑两股各自后续最高分替换系统，没有用后续标签校准概率。

## 质量与成本

{chr(10).join(q)}

“选择概率质量”指原词表分布赋给UP和DOWN的总概率，小值意味着强制二选一排除了
模型原本偏好的其他token；不据此删除窗口。它不是抽取质量或分类准确率。
二选一输出不生成解释或证据编号；兼容字段evidence_valid不代表做了语义证据验收。
批处理行耗时按批次均摊，不是每条请求实测延迟。
[执行证据](EXECUTION.json)提供版本、哈希、长度、内存和耗时；
[真实LR训练](BASELINE_TRAINING.json)提供60次拟合与模型重载核验。
最终三组记录推理耗时合计{primary_seconds/60:.1f}分钟；额外串行、批处理检查、
JSON版及smoke约{prior_seconds/60:.1f}分钟。两者合计{(primary_seconds+prior_seconds)/60:.1f}分钟
为各作业记录耗时之和，不是端到端墙钟时间（少量基准曾并发，准备/加载也不全在此内）。

## 价格版行为诊断

后续时期，价格版方向与过去六小时平均收益符号的一致率为AAPL91.57%、AMZN95.53%，
概率与该均值的Spearman相关为0.868/0.824。行为很像近期趋势延续，
不能说已经学到跨时期的四小时预测规律。这是事后描述，不是因果特征归因或新选中的模型。
[详细统计](PRICE_BEHAVIOR.json)。

## 无新闻时的偏向

新闻单独输入时，AMZN后续95个无新闻窗口全部判DOWN，上涨概率约0.165–0.245；
这些窗口实际上涨比例约49.5%，Brier为0.3286。它表现出明显负向先验，
不能把这个子集上的行为归功于新闻理解；也不能仅凭此认定记住了历史结果。
[缺失新闻诊断](NEWS_ABSENCE.json)。

## 案例与不确定性

[配对区间](PAIRED_INTERVALS.csv)按交易日配对重采样，含1日/最多5日连续块，每项500次；
仅描述历史波动，不是探索选择校正后的显著性证据。联合组区间为预先计划；
看到新闻组方向成绩后补充了新闻组的描述性区间，明确属于事后分析。
[固定案例](CASE_SELECTION.json)按股票×时期×两模型对错分层后SHA选择，
对比联合LLM与匹配联合LR。阅读结论见[CASE_NOTES.md](CASE_NOTES.md)；全新闻和逐次模型输入留本地。
V2无文字理由，因此案例只能检验输入与预测关系，不能编造LLM内部推理过程。

## 解释边界

所有时期已暴露。现代LLM预训练可能包含这些旧新闻或结果，prompt无法证明排除了记忆。
本轮不等同新时期泛化验证。没有LLM权重训练、历史示例、thinking对照、更长历史或最新
未完成小时的价格补充；这些机制不能写成已验证。归一化答案偏好也不能冒充校准概率。
'''
 (B/'REPORT.md').write_text(report);print('Published completed choice experiment and partial-run history.')
if __name__=='__main__':main()
