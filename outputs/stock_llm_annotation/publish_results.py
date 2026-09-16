"""Build public aggregate report; never copy raw article text or model labels."""
import csv,json,shutil,sys
from pathlib import Path
from labels import rows,dump,ROOT
B=Path(__file__).resolve().parent
W=ROOT/'work/stock-data/annotation'
def main():
 runs=W/'runs';variants={'ordinary':'qlora_v2','gate':'qlora_gate_v1'}
 evidence={k:json.loads((runs/v/'summary.json').read_text()) for k,v in variants.items()}
 assert all(r['actual_training'] and r['frozen_base_unchanged'] and r['reload_greedy_equal'] for r in evidence.values())
 dump(B/'TRAINING_EVIDENCE.json',evidence)
 curves=[]
 for k,v in variants.items():
  for r in rows(runs/v/'training.jsonl'):curves.append({'variant':k,**{x:r[x] for x in ['micro_step','epoch','loss','loss_weight','seconds','optimizer_updates','peak_gb']}})
 with (B/'TRAINING_CURVES.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(curves[0]),lineterminator='\n');w.writeheader();w.writerows(curves)
 comparison=[]
 for variant in variants:
  with (W/f'comparison_{variant}/metrics.csv').open() as f:
   for r in csv.DictReader(f):comparison.append({'experiment':variant,**r})
 with (B/'METRICS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(comparison[0]),lineterminator='\n');w.writeheader();w.writerows(comparison)
 diag=json.loads((W/'loss_diagnostic.json').read_text());dump(B/'LOSS_DIAGNOSTIC.json',diag)
 dump(B/'CONTEXT_VERIFICATION.json',json.loads((W/'preflight_v3/summary.json').read_text()))
 metrics={k:json.loads((runs/v/'metrics.json').read_text()) for k,v in {'rules':'rules_v3','frozen':'frozen_v3','ordinary':'tuned_v3','gate':'tuned_gate_v3'}.items()}
 selected={k:next(x for x in e['development'] if x['epoch']==e['selected_epoch']) for k,e in evidence.items()}
 def pct(v):return '—' if v is None else f'{v*100:.1f}%'
 table=[]
 for k,label in [('rules','原规则'),('frozen','冻结 Qwen'),('ordinary','扩充数据 QLoRA'),('gate','QLoRA＋事件存在监督')]:
  d=metrics[k]['valid'] if metrics[k].get('valid',{}).get('n',0) else selected[k];m=metrics[k]['check']
  table.append(f"| {label} | {d['fact_tp']}/20 | {m['schema_and_literal_evidence_valid']}/{m['n']} | {m['fact_tp']}/{m['fact_tp']+m['fact_fn']} | {pct(m['precision'])} | {pct(m['recall'])} | {m['fact_fp']} | {m['no_event_false_positive']}/{m['no_event_n']} |")
 training='\n'.join(f"- {k}: {e['micro_steps']} micro-steps / {e['optimizer_updates']} 次更新，{e['seconds']/60:.1f} 分钟，MLX峰值 {e['peak_memory_gb']:.2f} GB；开发集选择 epoch {e['selected_epoch']}。" for k,e in evidence.items())
 epoch_table='\n'.join(f"| {k} | {d['epoch']} | {d['fact_tp']} | {d['fact_fp']} | {d['fact_fn']} |" for k,e in evidence.items() for d in e['development'])
 text=f'''# 扩大 GPT 标注与 Qwen 微调实验

## 本轮完成了什么

- ChatGPT 网页实际完成 **537 条候选 × 两轮 = 1,074 份输出**；每轮单独聊天，不共享答案。模型设置为 GPT 6 High。
- 助手审阅183条分歧/抽查记录；字段一致517/537并不等于准确率。25条不确定、4条冲突及50条事件重复排除。
- 最终 **249训练 / 87开发 / 122检查**。训练60篇正例中AAPL47、AMZN13；开发19篇正例均为AAPL；检查22篇正例中AAPL17、AMZN5。
- 所有结果来自实际本地运行，不使用付费API或Sol。标签、原文和模型留本地，GitHub只发布代码和汇总。

## 确实进行了 training

{training}

两版均为同一Qwen3-1.7B 4bit，最后8层q/v上的rank8 LoRA，458,752个可训练参数，3轮、seed573。记录非零梯度、adapter变化、冻结基座哈希不变，并在重新加载后验证logits误差为0。训练细节见 TRAINING_EVIDENCE.json 和逐步曲线。

## 抽取结果（不是四小时涨跌成绩）

开发集按事实得分选epoch，检查集不参与选参。以下检查总表含原面板和补充挑战；两者的单独结果及分股票结果见 METRICS.csv。

| 方法 | 开发匹配事实 | 检查格式/字面证据通过 | 检查匹配事实 | 检查精确率 | 检查召回率 | 多抽/错抽事实 | 无事件文章误报 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

“匹配事实”要求事件类型、动作、旧值、新值、单位均与暂定标签一致，允许不同证据句ID。另做字面证据核验；这仍不能替代语义复核。无输出或不合法JSON会漏掉正事实。全部输出及输入均留在本地供核验。

始终输出空事件也能让100/122篇文章的字段集合“完全正确”，但会漏掉全部23个事实。因此本表以事实精确率/召回率为主，不把约82%的文章准确率当成任务成功。

## 为什么增加第二种损失

第一版epoch1和epoch3均在开发集输出空事件。文章层面给正例2/3权重仍不够：按答案长度平均后，空/非空分支token的负例总系数是正例的 **{diag['negative_to_positive_gate_weight_ratio']:.2f}倍**。长的正例JSON有很多标点/键名，平均loss下降可能掩盖关键分支失败。

第二版只额外监督这一分支，正负类在辅助项中总权重相等；其余训练条件相同。精确梯度检查证明辅助项仅作用在登记位置、原prompt不计loss。这是针对性机制检验，不证明它是全部失败原因；本轮不再追加网格。

开发集各轮变化如下；不能用“输出更多事件”或“loss更低”单独判断改善：

| 版本 | epoch | 匹配事实 | 多抽/错抽 | 漏抽 |
|---|---:|---:|---:|---:|
{epoch_table}

## 数据与评价限制

最初60篇三月开发候选全部为无事件，宽关键词误选了大量基金增减持。保留原面板，再登记更明确的标题筛选，补38篇三月候选和39篇四月挑战。补充面板不是自然分布准确率估计。537个原文条目共4546段精确位置已核验。

重复事件过滤仅使用有限broker字典/动作/数字/七日发布时间；仍可能漏掉转载或误合并相似连续动作。两篇Credit Suisse AMZN检查文章是同一事件，不能算两份独立证据。部分源新闻自身有异常数字、几个月前的报告与当前发布日期冲突；抽取忠于原文不等于原报道正确。

**两轮GPT及助手审阅都不是独立人工金标准。** 关键类型独立验收仍未完成；开发集没有AMZN正例、检查正例数量少。不能宣布完整事件抽取通过验收，也不能用本表替代四小时BA/Brier。

本轮冻结对照与适配模型使用同一简短、无few-shot的输入模板。这是固定提示下的适配对照，不是已经找到最佳冻结prompt。语法约束解码、更多示例或更大模型没有在本轮完成；不能据此断言整个小型LLM方向无效。

## 四小时流水线状态

本轮完成上游标注、真实训练和匹配抽取对照。普通版仍漏17个事实、错抽8个，辅助版仅匹配1个事实，语义质量不足；完整事件分支的独立质量门槛也仍未通过。因此尚未运行全量事实推理和新的四小时下游模型；原四小时价格＋文本baseline及历史结果保持原状。未来若进入探索性接入，必须另报覆盖与语义质量、严格使用5月1日以后的可用抽取器及过去OOF基线，保留所有原窗口和无事件回退。

助手逐项阅读两个adapter的全部匹配及误报事实，发现历史列表、共识与动作混淆、新设目标价误当上调，以及旧值/事件对象错配，见[案例笔记](CASE_NOTES.md)。537条输入最长939个prompt token，预留512输出仍在2048预算内；训练/推理前缀逐条一致，没有截断。此项排除了本轮的长度/模板错位，但没有证明输入语义或prompt已经最优。

复现入口：[README](README.md)。原始/补充协议及损失对照保留。历史价格时期均已暴露，不能把后续回测包装成新时期泛化证明。
'''
 (B/'REPORT.md').write_text(text)
 dump(B/'STATUS.json',{'annotation_completed':True,'actual_training_completed':True,'training_variants':2,'check_inference_completed':True,'independent_human_review_passed':False,'new_four_hour_forecast_run':False,'existing_four_hour_results_unchanged':True,'reason_downstream_not_promoted':'Selected development checkpoints still miss most facts; independent semantic review and type-count requirements unmet. Extraction metrics are not four-hour prediction metrics.'})
 print('Published aggregate report and training evidence; no raw article text copied.')
if __name__=='__main__':main()
