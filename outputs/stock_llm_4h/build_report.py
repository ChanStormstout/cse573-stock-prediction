"""Build public aggregates. Never includes source article text or model weights."""
from common import *
import csv

def main():
 versions={'rule_v1':'既有规则','frozen_v1':'冻结Qwen＋少量示例','tuned_v1':'QLoRA普通权重','tuned_balanced_v1':'QLoRA正事件加权','frozen_current_v1':'冻结Qwen＋当前段落截断'}
 records=[]
 for v,name in versions.items():
  metrics=json.loads((B/'runs'/v/'metrics.json').read_text())
  for split,m in metrics.items():records.append({'version':v,'method':name,'split':split,**m})
 keys=['version','method','split','n','schema_and_literal_evidence_valid','exact_fact_set','fact_tp','fact_fp','fact_fn','precision','recall','no_event_n','no_event_false_positive']
 with (B/'METRICS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(records)
 costs=[];curves=[]
 for name in ['smoke_v2','pilot_s573_v1','pilot_balanced_s573_v1']:
  costs.append({'run':name,**json.loads((B/'runs'/name/'summary.json').read_text())})
  curves.extend({'run':name,**r} for r in rows(B/'runs'/name/'training.jsonl'))
 with (B/'TRAINING_CURVES.csv').open('w') as f:
  keys=['run','micro_step','epoch','id','loss','loss_weight','seconds','optimizer_updates','peak_gb'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(curves)
 dump(B/'TRAINING_EVIDENCE.json',costs)
 pos=json.loads((B/'runs/position_v1/summary.json').read_text());trim=json.loads((B/'data/pilot_current_v1/manifest.json').read_text());integration=json.loads((B/'runs/integration_v2/status.json').read_text())
 table=['| 方法 | 开发：格式与字面证据通过 | 开发：匹配事实 / 9 | 开发：多抽事实 | 检查：匹配事实 / 3 |','|---|---:|---:|---:|---:|']
 for v,name in versions.items():
  d=next(r for r in records if r['version']==v and r['split']=='valid');c=next(r for r in records if r['version']==v and r['split']=='check');table.append(f"| {name} | {d['schema_and_literal_evidence_valid']}/8 | {d['fact_tp']}/9 | {d['fact_fp']} | {c['fact_tp']}/3 |")
 report='''# 小型LLM实施与实验报告 — 2026-09-16

## 结论

**已实际运行位置偏差诊断、冻结Qwen、规则对照、M5 QLoRA工程检查与两版任务微调。尚未得到可用于批量四小时预测的事件抽取器。** 新四小时事件模型没有训练，也没有新的BA提升结果。没有独立质量复核；不能将代码完成或格式改善当作方法有效。

## 做了什么

1. 12个既有案例×3种字母/类别顺序=36次生成：原顺序12/12选A（earnings）；移位3时11/12选A（product_business），移位6时8/12选A（market_recap）。12个案例的类别全部随顺序改变。支持选项敏感性，不能据此把所有失败唯一归因于位置。
2. 准备43个真实文章—公司输入，逐个阅读296个编号片段后写助手暂定标签；未提供目标四小时收益/标签，新闻本身仍可能描述股价。22个训练、8个开发、11个检查，另2个同Longbow事件报道只诊断、不当新增训练事件。训练只有7个正事件文章（8事实）和15个无事件输入；全部AMZN输入在当前限定范围内为负例。不是300–600例已验收语料。
3. 第一版抽取仅限当前主要披露的评级/目标价动作。排除指引、目标价撤销、个人估值和完整机构关系。标签只对供给的段落负责，不宣称完整文章无事件。
4. 同输入运行既有规则、冻结Qwen、普通QLoRA、加权QLoRA。最后根据训练样本的人工正确证据诊断，另登记并测试显式历史列表截断；不声称已经在新输入上再次微调。
5. 实现低维聚合/无截距修正接口、过去OOF账本检查、独立质量门槛及无事件精确回退。门槛未过，因此只做原底座回放，不做全量抽取/新预测训练。

## 匹配对照结果

'''+ '\n'.join(table)+'''

这里的“匹配事实”比较kind/action/old/new/unit，先经过schema及字面证据检查；**不等于独立确认整条证据关系正确**。加权模型找回的L29评级字段虽一致，却还引用了其他历史机构句。检查集只有3个正事实、8篇无事件，且已暴露；规则在这小面板上的高分不能推广为整体质量。不能用真空输出带来的格式通过率/负例准确率掩盖漏抽。

## 实际训练证据

- 机器：Apple M5、24GiB；MLX0.32.2 / mlx-lm0.31.3，本地Metal。
- 同源Qwen3-1.7B 4bit，最后8层Q/V，rank8、scale16、dropout.05，LR1e-4。458,752个可训练参数。MLX量化基座LoRA，不宣称完整复现NF4算法。
- 工程检查：8个样本，32 micro-steps、4次优化器更新；约40.08秒、峰值2.90GB。
- 普通pilot：22个非重复样本×3遍=66 micro-steps、9次更新；约168.35秒、峰值2.83GB；开发选epoch1。
- 加权pilot：相同数据/步数/参数，仅将正事件总期望损失权重设为2/3；约143.53秒、峰值2.83GB；开发选epoch2。没有挑检查集最优轮数。
- 三次成功训练合计164 micro-steps、22次更新。每次均验证非零梯度、adapter改变、基座哈希不变、重载logit最大误差0与确定性输出一致。`TRAINING_EVIDENCE.json`和`TRAINING_CURVES.csv`保存证据。
- 只有seed573；第二版仍未可用，按两次针对性失败停止扩种子和扩大参数搜索。22例小pilot不足以否定更大高质量监督语料的微调路线。

## 发现与解释边界

- 旧单字母任务主要反映受限输出/顺序敏感性，不能概括LLM能力。
- 普通QLoRA很快学会输出合法空事件；它在开发上的9个真实事实全部漏掉。降低训练loss或提高JSON合法率不代表完成任务。
- 加权恢复很少字段，仍缺少足够多样的正例，检查期也未恢复目标事实。不能因少量改变宣布适配成功。
- 七个训练正例改为人工指定证据后，冻结模型从0/8提高到2/8匹配事实。这是人工输入上界诊断，不是可部署方法，也非独立检查证据。
- 据此实现无标签的历史列表截断：16篇输入改变，移除58句，本面板未删掉暂定gold证据；实际冻结模型结果见表。这个数字不证明原文证据召回100%。
- 仅修复尾部引号/括号使8条输出能解析，但开发匹配事实仍为0/9；格式修补不足以解决对象/历史混淆。
- 原文还存在“早上抓取、正文叙述当日收盘”的现象（L31）。内部时间字段守住不能认证原站页面版本；此例不用于宣称时点可交易新闻。

## 为什么没有继续跑新的四小时BA

完整关系分支独立复核为0；本pilot每类型远不足30独立正事件检查，AMZN没有正例监督。继续对全语料生成特征会放大未经证明的错误，且违反事先质量停止条件。原1,607个窗口保留；对已有1,488条预测（含879条前向训练期记录和609条开发/后续记录）验证无事件返回原价格＋标题概率完全一致。**这只是回退/接入检查，不是新模型表现。**

本地`runs/integration_v2/independent_review.csv`绑定具体候选输出与输入指纹，供组员复核。它不是已经完成的独立审查；11个检查文章仍需补足类型覆盖。完整训练输入、原文和权重不进公开Git。

## 保留的失败与未完成项

- prepare v1在分公司之前按文章去重导致AMZN被移除；v2修复并保留v1。v3在推理之前加3篇盘外早期评级文章，使检查至少含3正例；这是人为分层补样，非随机总体。
- smoke_v1在bfloat16→NumPy哈希转换前失败，尚未训练；改为无损float32表示用于哈希后smoke_v2通过。
- integration_v1为初始接口回放；v2改为实际加权候选的检查期输出复核包，禁止把助手gold正确性当作模型质量。
- 未完成300–600标注、独立验收、8bit/thinking对照、多种子正式研究、全量新闻抽取与B1–B3四小时预测。未把有限评级schema当作完整金融事件系统。

## 下一步

先补足早期、非重复、真实正例（尤其AMZN），并完成独立标注复核，再决定是否恢复适配。可用当前输入修复作为一个单独对照；下一轮不能重新计作本轮已完成的全流程收益。扩大模型或上Sol暂不是已证实的解决方案，本轮小模型在Mac上可运行，主要卡在监督/抽取质量。

## 验证与复现

17项边界测试通过；296个输入字符片段与原文逐一对应，标注封存哈希与已登记分组/时间划分通过。语义近似分组是启发式＋助手检查，不保证所有重复都识别。命令见README。所有现有时期结果为探索性历史分析。
'''
 (B/'REPORT.md').write_text(report)
 dump(B/'STATUS.json',{'status':'PILOT_EXECUTED_QUALITY_NOT_ACCEPTED','new_four_hour_training':False,'actual_qlora_training':True,'training_runs':3,'micro_steps':sum(x['micro_steps'] for x in costs),'optimizer_updates':sum(x['optimizer_updates'] for x in costs),'independent_review_passed':False,'published_raw_data':False,'position_probe_articles':12,'pilot_articles':43,'training_unique':22,'tests':17,'integration':integration})
 print('Report and public aggregates generated.')
if __name__=='__main__':main()

# Append the separately registered follow-up without overwriting primary experiments.
if __name__ == '__main__' and (B/'runs/prompt_context_v1/metrics.json').exists():
 m=json.loads((B/'runs/prompt_context_v1/metrics.json').read_text())
 text='\n## Follow-up: fixed prompt/context 2x2\n\nSame eight exposed development articles, same frozen model/schema/decoding. Original cells reused;16 new generations. Short prompt also removes chat examples, so this compares prompt packages, not length alone.\n\n| Context | Prompt | Valid schema/literal evidence | Matched facts / 9 | Extra facts |\n|---|---|---:|---:|---:|\n'
 for key,r in m.items():
  context,prompt=key.split('|');text+=f"| {context} | {prompt} | {r['schema_and_literal_evidence_valid']}/8 | {r['fact_tp']}/9 | {r['fact_fp']} |\n"
 text+='\nInput filtering helped slightly; shortening the prompt did not help in this comparison. Neither proves that a better prompt cannot work. Current-vs-static-target definitions, primary-event scope, negative-example balance and evidence-span quality still require an independently checked task contract. No new forecast performance claim.\n'
 with (B/'REPORT.md').open('a') as f:f.write(text)
