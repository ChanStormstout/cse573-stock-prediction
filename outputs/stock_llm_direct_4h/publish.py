"""Publish inspected numerical summaries only, never source text or raw responses."""
import argparse,csv,json,shutil
from pathlib import Path
from common import B,rows,dump,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--analysis',required=True,type=Path);ap.add_argument('--run',required=True,type=Path);ap.add_argument('--baseline',required=True,type=Path);ap.add_argument('--inputs',required=True,type=Path);a=ap.parse_args()
 for src,dst in [('metrics.csv','METRICS.csv'),('native_direction_diagnostic.csv','NATIVE_DIRECTION.csv'),('paired_intervals.csv','PAIRED_INTERVALS.csv'),('quality.json','QUALITY.json'),('verification.json','VERIFICATION.json'),('case_selection.json','CASE_SELECTION.json'),('coverage_summary.json','COVERAGE.json'),('predictions.csv','PREDICTIONS.csv'),('native_predictions.csv','NATIVE_PREDICTIONS.csv')]:shutil.copy2(a.analysis/src,B/dst)
 shutil.copy2(a.baseline/'cv.csv',B/'BASELINE_CV.csv');shutil.copy2(a.baseline/'summary.json',B/'BASELINE_TRAINING.json')
 manifest=json.loads((a.run/'manifest.json').read_text());summary=json.loads((a.run/'summary.json').read_text());assert summary['completed']
 execution=dict(manifest=manifest,summary=summary,input_check=json.loads((a.run/'input_check.json').read_text()),note='No LLM fine-tuning. Row seconds share batch elapsed time when batching; not request latency. Additional serial/benchmark costs are documented separately.')
 dump(B/'EXECUTION.json',execution)
 seal=json.loads((a.inputs/'manifest.json').read_text());dump(B/'INPUT_VERIFICATION.json',{k:seal[k] for k in ['n','split_counts','input_sha','features_sha','coverage_sha','historical_price_and_label_parity','protocol_sha']})
 scores=list(csv.DictReader((B/'METRICS.csv').open()));native=list(csv.DictReader((B/'NATIVE_DIRECTION.csv').open()))
 names={'legacy_title':'原价格＋标题baseline','legacy_body':'旧价格＋全文','legacy_semantic':'旧价格＋FinBERT','legacy_integrated':'旧完整组合','matched_price':'相同历史价格LR','matched_news':'相同新闻片段LR','matched_joint':'相同价格＋新闻片段LR','llm_price':'LLM看价格','llm_news':'LLM看新闻','llm_joint':'LLM看价格＋新闻'}
 def fmt(x):return f'{float(x)*100:.2f}%'
 def table(period):
  out=['| 方法 | AAPL BA | AMZN BA | AAPL Brier | AMZN Brier |','|---|---:|---:|---:|---:|']
  for name,title in names.items():
   rr=[next(r for r in scores if r['method']==name and r['symbol']==s and r['period']==period and r['scope']=='all') for s in ['AAPL','AMZN']]
   out.append(f"| {title} | {fmt(rr[0]['BA'])} | {fmt(rr[1]['BA'])} | {float(rr[0]['Brier']):.4f} | {float(rr[1]['Brier']):.4f} |")
  return '\n'.join(out)
 nd=['| 输入 | 时期 | AAPL明确方向BA | AMZN明确方向BA |','|---|---|---:|---:|']
 for v in ['price','news','joint']:
  for period in ['validation','test']:
   rr=[next(r for r in native if r['method']=='llm_'+v and r['symbol']==s and r['period']==period) for s in ['AAPL','AMZN']]
   nd.append(f"| {names['llm_'+v]} | {period} | {fmt(rr[0]['BA'])} | {fmt(rr[1]['BA'])} |")
 quality=json.loads((B/'QUALITY.json').read_text());q=['| 输入 | 有效概率/方向 | p=0.5窗口 | 可解析JSON中证据ID未通过 | 分钟 |','|---|---:|---:|---:|---:|']
 for v,x in quality.items():q.append(f"| {v} | {x['n']-x['invalid']}/{x['n']} | {x['p_exact_half']} | {x['bad_citations']} | {x['seconds']/60:.1f} |")
 report=f'''# LLM直接预测四小时：完整三组实验

## 做了什么

LLM直接读取历史数据并给出上涨概率/方向，**没有先提取事件再交给分类器**。
准备全部1,607原窗口，推理覆盖609个非训练窗口：开发252，后续357。
价格、新闻、联合三组均完整运行；未因无新闻、抽取失败或预测信心低删样本。
固定Qwen3.5-9B 4bit，nonthinking、greedy，原生模板；没有LLM微调。
另外实际完成60次传统LR拟合，六个最终模型重载概率一致。
[准确prompt与输入字段](common.py)；[事前协议](PROTOCOL.md)；[复现入口](README.md)。

## 输入与公平性

价格：最近6个完成交易小时的收益、振幅、历史年龄及摘要；由5分钟价格重建并核对。
新闻：原候选集合中最近收到的最多6篇，标题最多240字符、正文最多600字符，
含发布时间/可用时间和延迟。原文片段有省略，覆盖详见[COVERAGE.json](COVERAGE.json)。
正文没有经过事件抽取的成功过滤，也没有用模型总结代替原文。
[训练期输入抽查](INPUT_AUDIT.md)已发现字符截取可能截断句子、遗漏目标段落，
以及重复宣传/旧持仓报道；本轮不据结果更换输入。
固定窗口起点前5分钟为截止；预测区间开盘至4小时结束，未来开盘价不输入。
三组最大输入token数见执行证据，没有token级静默截断。
新LR读取相同新闻片段及历史价格源；LLM拥有逐篇时间结构，LR使用时间摘要，
这是信息来源匹配，不能称两种算法拥有完全相同的内部表示。旧模型使用不同文本范围，
仅作原项目参照。价格窗口不含新增成交量、指数或更长历史。

## 原协议评分：后续时期（已暴露）

AAPL178、AMZN179窗口。BA是两类召回率平均值，Brier越低越好。
概率/方向矛盾或无合法概率的输出按事前规则回退p=.5，并以UP处理平局，全部保留。

{table('test')}

## 开发时期

{table('validation')}

## 模型明确给出的方向：单列诊断

推理初期、尚未连接涨跌答案时，发现模型会同时输出p=.5和DOWN。
这违反原定平局规则，原评分会将其标为失败回退UP。
为区分输出一致性与方向能力，另列原生UP/DOWN成绩；非法方向仍回退UP。
**没有选择两套评分中较好者作为结果；此表是事后登记的接口诊断。**
原生方向与数值概率可能互相矛盾，不能把二者拼成已校准的预测器。
[NATIVE_DIRECTION.csv](NATIVE_DIRECTION.csv)另给原始合法数值概率的Brier。

{chr(10).join(nd)}

## 质量与执行成本

{chr(10).join(q)}

p=.5统计含失败回退；模型给的概率不等同实际正确率。短解释只是生成文本，
证据编号存在也不证明理由正确或是实际推理过程。检查详见[QUALITY.json](QUALITY.json)。
[EXECUTION.json](EXECUTION.json)保存模型revision、指纹、长度与运行成本；
[BASELINE_TRAINING.json](BASELINE_TRAINING.json)保存真实LR拟合与选参证据。
若使用批处理，行耗时是整批耗时均摊，不是逐请求延迟；串行/工程基准的额外成本另记。

## 不确定性与案例

[配对区间](PAIRED_INTERVALS.csv)按交易日配对重采样，含1日/5日块，每项500次。
用于描述已暴露历史结果的波动，不能当作探索选择校正后的显著性检验。
[固定案例面板](CASE_SELECTION.json)按股票、时期、两模型对错分层后SHA选取，
比较联合LLM与匹配联合LR。实际阅读结论另见CASE_NOTES.md；完整原文和生成输出留本地。

## 限制

所有时期已参与项目探索；现代LLM可能在预训练见过旧新闻/后续结果，提示词无法消除
这项风险。本轮只检验这一冻结模型/提示/历史长度/新闻预算，不能据此判断所有LLM
方案。四个历史示例、更多历史、thinking、约束输出和直接涨跌微调尚未在此轮测试。
没有将抽取指标、训练拟合或合理解释冒充股票预测成绩；没有新独立保留期。
'''
 (B/'REPORT.md').write_text(report)
 print('Published numerical summaries only.')
if __name__=='__main__':main()
