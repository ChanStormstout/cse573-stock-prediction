"""Publish only aggregate metrics; source articles and raw predictions stay local."""
import argparse
import csv
import json
import shutil
from pathlib import Path
from contracts import ROOT
B=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--safety',type=Path);a=p.parse_args()
    summary=json.loads((a.run/'summary.json').read_text());manifest=json.loads((a.run/'manifest.json').read_text())
    assert summary['completed'] and not manifest['smoke_only']
    for name,src in [('METRICS.csv','metrics.csv'),('QUALITY.json','quality.json'),('VERIFICATION.json','verification.json'),('CASE_SELECTION.json','case_selection.json'),('EVENTWISE_DIAGNOSTIC.json','eventwise_diagnostic.json')]:
        shutil.copy2(a.analysis/src,B/name)
    if (a.analysis/'rule_repair.json').exists():shutil.copy2(a.analysis/'rule_repair.json',B/'RULE_REPAIR.json')
    public={k:manifest[k] for k in ['model','revision','variants','seed','temperature','thinking','max_output_per_call','max_total_per_call','training','input_sha256','labels_sha256','code','model_files','evaluation_status']}
    public.update(summary=summary)
    (B/'EXECUTION.json').write_text(json.dumps(public,indent=2)+'\n')
    safety_section=''
    if a.safety:
        safety={'manifest':json.loads((a.safety/'manifest.json').read_text()),'metrics':json.loads((a.safety/'metrics.json').read_text())}
        (B/'SAFETY_REPAIR.json').write_text(json.dumps(safety,indent=2)+'\n')
        m=safety['metrics']['check'];f1=2*m['fact_tp']/(2*m['fact_tp']+m['fact_fp']+m['fact_fn'])
        safety_section=f'''## 事后安全修复：单独报告

固定案例复核发现评级词表只认出Positive、不认出Mixed时，把旧评级误作新评级。
[安全修复](SAFETY_FIX.md)拒绝不能完整配对的字段，保留原始运行；没有新增模型调用。
检查结果为TP {m['fact_tp']}、FP {m['fact_fp']}、FN {m['fact_fn']}，F1 {f1:.1%}。
它删掉两条错误字段，未找回漏抽事实；这是检查后发现的正确性修复，不是独立确认的提升。
[修复证据](SAFETY_REPAIR.json)。
'''
    metrics=list(csv.DictReader((a.analysis/'metrics.csv').open()))
    names={'rules':'原规则','old_frozen':'旧1.7B冻结','old_tuned':'旧1.7B微调','original':'9B＋原提示方案','prompt':'9B＋新提示和示例','staged':'9B＋分步抽取和数值规则'}
    if any(r['model']=='staged_v2' for r in metrics):names['staged_v2']='分步版＋单项数值规则修正'
    def percent(value):return '—' if not value else f'{float(value)*100:.1f}%'
    lines=[]
    for model in names:
        m=next(r for r in metrics if r['model']==model and r['phase']=='check' and r['cohort']=='all' and r['symbol']=='ALL')
        lines.append(f"| {names[model]} | {m['fact_tp']} | {m['fact_fp']} | {m['fact_fn']} | {percent(m['precision'])} | {percent(m['recall'])} | {percent(m['f1'])} | {m['schema_and_literal_evidence_valid']}/{m['n']} |")
    dev=[]
    for model in [n for n in names if n not in {'rules','old_tuned'}]:
        m=next(r for r in metrics if r['model']==model and r['phase']=='valid' and r['cohort']=='all' and r['symbol']=='ALL')
        dev.append(f"| {names[model]} | {m['fact_tp']} | {m['fact_fp']} | {m['fact_fn']} | {percent(m['f1'])} |")
    times=[]
    for variant,s in summary['variants'].items():
        times.append(f"| {names[variant]} | {s['n']} | {s['calls']} | {s['seconds']/60:.1f} | {s['peak_mlx_gb']:.2f} | {s['length_stops']} |")
    report=f'''# 换模型、改prompt、拆分抽取：实际实验结果

## 范围

本轮已完成固定Qwen3.5-9B 4bit在同一批87篇开发、122篇检查新闻上的三组离线推理。
**没有微调或训练新权重，也没有训练新的四小时预测器。** 旧1.7B冻结/微调和规则作为保存的参考。
所有标签仍为模型辅助暂定；四月已用于先前错误分析，因此本表是暴露面板的回归诊断，不是新时期确认。

## 三个问题分开检验

1. 只换模型：保持原消息内容和字段要求，使用9B原生聊天模板。模型家族、规模和tokenizer一起变化，不能归因于参数量一个因素。
2. 改prompt：同一9B与输入，增加明确检查清单和5个合成示例。这检验整个提示方案，不能单独归因于某一句话。
3. 拆分任务：先选当前目标公司事件证据，再分类动作并复制证据子句，程序读取数值。最多两次模型调用，额外计算单独报告。该组也改变了提示内容、取消few-shot并增加规则拒绝，检验的是完整拆分方案，不能单独归因于“多一个步骤”。

开发阶段另发现“下调2美元至168美元”的规则覆盖缺口，登记[一次有限修正](RULE_AMENDMENT.md)：2是变化幅度，保留新值168、旧值仍未知。复用同一批已保存模型输出，零额外模型调用；原分步版和修正版分开报告。没有用分步版四月输出调这条规则。

## 检查结果：122篇，23条暂定事实

| 方法 | 正确事实 | 错抽/多抽 | 漏抽 | 精确率 | 召回率 | F1 | 格式及字面证据通过 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

严格复用原事件类型/动作/旧值/新值/单位匹配。某一事件不合法会使整篇响应被原校验器拒绝，
因此FP是通过响应中的错事实，不能代表全部坏输出。无事件、gate拒绝和规则失败的文章全部保留在分母。
分步组的格式列指最终程序输出，不能当作所有中间模型输出均合格；被规则删掉的事实仍会形成漏抽。
另存[逐事件诊断](EVENTWISE_DIAGNOSTIC.json)：只对已解析JSON逐事件校验，观察整篇拒绝是否遮住部分正确事实。这是事后诊断，放宽了原整篇策略，不替换上表，也不算新的系统成绩。
100篇参考为空，不能把始终空输出的约82%文章准确率当成成功。

## 开发结果：87篇，20条暂定事实

| 方法 | 正确事实 | 错抽/多抽 | 漏抽 | F1 |
|---|---:|---:|---:|---:|
{chr(10).join(dev)}

没有根据本轮开发输出继续修改prompt或筛掉检查失败；三组都完整报告。
完整分股票、原检查面板和富集挑战见[METRICS.csv](METRICS.csv)。开发正例全为AAPL，AMZN的语义结论尤其有限。

## 实际本机执行成本

| 方案 | 文章数 | 调用次数 | 分钟 | 累计MLX峰值GB | 输出触顶次数 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(times)}

峰值是同一进程累计MLX值，不是独立隔离的每组内存或整机内存。权重约6GB，本地M5执行；
输出最多512token/调用，总预算4096，拒绝输入截断。greedy、nonthinking，seed573。
原1.7B预算2048而9B安全上限4096；若原消息均未截断，容量上限不是新增输入信息。

## 工程核验与证据

保存固定模型revision、模型文件哈希、输入/标签哈希、代码快照、每次调用消息、原始输出和token/耗时。
三组覆盖相同209篇文章，原提示消息内容逐条核验，无静默截断。
合成smoke发现枚举占位符被照抄，正式新闻前修复并登记[协议修订](SMOKE_AMENDMENT.md)；旧输出保留。
边界单测检查字面证据、单位、数值配对、否定、gate及unknown动作。见[VERIFICATION.json](VERIFICATION.json)。

## 解释边界与后续

分步原版AAPL的F1为25.8%，低于旧微调的34.5%；AMZN为62.5%，高于旧微调的25.0%。
AMZN仅6条事实，且两篇文章重复同一Credit Suisse事件，因此不能声称跨股票稳定胜出。
固定21篇案例及额外定向检查见[CASE_NOTES.md](CASE_NOTES.md)：静态评级、历史列表、证据遗漏和程序语法覆盖仍造成错误。

格式、字面证据和语义准确性分开。规则可拒绝不明确的数值，但不能独自证明事件属于目标公司或当前时点。
更高抽取F1也不能证明股价方向收益。没有新四小时BA/Brier；原四小时预测结果保持不变。
完整独立人工质量验收未通过，新模型结果不自动提升到全量预测流水线。
本轮沿用旧greedy策略以保持对照；[Qwen官方](https://huggingface.co/Qwen/Qwen3.5-9B)另有随机采样及presence penalty建议，未在本轮尝试。thinking、约束JSON和更长输出也未测试，不能把单一配置结果等同模型能力上限。
案例选择预先固定，见[CASE_PROTOCOL.md](CASE_PROTOCOL.md)和[CASE_SELECTION.json](CASE_SELECTION.json)。

{safety_section}

[复现入口](README.md)；[协议](PROTOCOL.md)；[执行证据](EXECUTION.json)；[组件失败统计](QUALITY.json)。
原新闻、暂定标签和权重仅留本地，不在公开仓库分发。
'''
    (B/'REPORT.md').write_text(report)
    print('Published aggregate evidence; raw data stay local.')

if __name__=='__main__':main()
