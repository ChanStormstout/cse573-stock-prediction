"""Build report and a standalone offline replay, using only saved public results."""
from pathlib import Path
import json
import pandas as pd,numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];P=HERE/'v1'

def main():
 d=pd.read_csv(P/'predictions.csv');m=pd.read_csv(P/'metrics.csv');monthly=pd.read_csv(P/'monthly_metrics.csv');e=json.loads((P/'execution.json').read_text());im=pd.read_csv(P/'intervention_metrics.csv');days=pd.read_csv(P/'amzn_day_contributions.csv');cases=json.loads((P/'cases.json').read_text());ci=pd.read_csv(P/'paired_intervals.csv')
 def markdown(methods):
  lines=['| 方法 | AAPL开发 BA / Brier | AAPL后续 | AMZN开发 | AMZN后续 |','|---|---|---|---|---|']
  for method,label in methods:
   cells=[]
   for s,p in [('AAPL','development'),('AAPL','later'),('AMZN','development'),('AMZN','later')]:
    r=m[(m.symbol==s)&(m.phase==p)&(m.method==method)].iloc[0];cells.append(f'{r.BA:.2%} / {r.Brier:.4f}')
   lines.append('| '+label+' | '+' | '.join(cells)+' |')
  return '\n'.join(lines)
 four=markdown([('B0','B0 原F2'),('B1','B1 F2＋温度'),('B2','B2 R1＋FinBERT'),('B3','B3 R1＋FinBERT＋自身温度')])
 g=e['gate'];day=days[days.phase=='later'].sort_values('BA_contribution',ascending=False);total=day.BA_contribution.sum();top=day.head(3).BA_contribution.sum()
 am=d[(d.symbol=='AMZN')&(d.phase=='later')];before=(am.N0M>=.5)==am.label;after=(am.N1M>=.5)==am.label
 changed=am[before!=after];changed_days=changed.day.nunique();removed=am[~am.day.isin(day.head(3).day)]
 def ba(y,p):
  q=p>=.5;return .5*(q[y==1].mean()+(~q[y==0]).mean())
 sensitivity=ba(removed.label,removed.N1M)-ba(removed.label,removed.N0M)
 lines=['# 有限组合验证与AMZN聚合归因','',
 '**结论：组合降低了概率误差，但仍未通过训练期逐股Brier护栏；停止扩模型，进入课程交付。AMZN的聚合局部增益主要出现在训练/选参改变后，而非预测时直接替换当前聚合向量。**','',
 '## 实际执行','',
 f'保存预注册后，实际建立{e["calibration_records"]}个逐月/最终校准记录，其中{e["optimized_calibrators"]}个满足最少30条过去新闻OOF而进行温度优化，其余使用预定identity。重新拟合了F2温度以核对旧结果，并拟合J3自己的温度。复用已有266次LR模型和分数，本轮没有重复训练分类器或FinBERT。另完成8个保存模型×替换输入推理，检查两个端点精确重现N0M/N1M。计算约%.2f秒。'%e['seconds'],'',
 '固定1,607窗口；233初始训练窗口不伪造OOF，765训练前向、252开发、357后续有预测。校准只用每股过去有新闻OOF；March为identity，April–August用更早月份，September以后冻结August末校准器。无新闻继续逐值等于R1，正温度不改变任何方向。所有时期已经暴露。','',
 '## 四组匹配结果','',four,'',
 '温度使J3后续Brier从AAPL 0.2589 / AMZN 0.2669降到0.2494 / 0.2462。BA保持56.71% / 53.79%。相对原F2的后续AMZN方向仍略低；不能把更合理概率称为方向提升。','',
 '## 训练期决策','',
 f'主比较B3对B1：6个前向月中{g["positive_months"]}个月宏平均BA为正，平均BA增加{g["BA_delta"]*100:.2f}个百分点；宏平均Brier恶化{g["Brier_delta"]:.4f}。AAPL BA +2.85个百分点、Brier改善0.0010；AMZN BA +0.33个百分点、Brier恶化0.0220，超过0.002。因此组合未通过，不能自动启动更大编码器或注意力。','',
 '这是预注册的正温度限定实验；上轮Platt被训练规则选中而后续失效的结果继续保留。没有把当前温度结果追溯改写成上轮选择。','',
 '## AMZN 57.32%究竟从哪里来','',
 '公平参照是带相同元信息的N0M 54.74%，不是只与不含元信息的F2比较。N1M 57.32%相对它增加2.58个百分点。179个后续窗口中，84有新闻、95无新闻；全部增量来自有新闻部分，无新闻完全相同。','',
 '| 保存的训练模型 | 当前窗口输入 | AMZN开发BA | AMZN后续BA | 后续Brier |','|---|---|---:|---:|---:|']
 for method,label,inp in [('N0M_article','文章训练','文章均值'),('N0M_event','文章训练','事件均值'),('N1M_article','事件训练','文章均值'),('N1M_event','事件训练','事件均值')]:
  r=im[(im.symbol=='AMZN')&(im.phase=='later')&(im.method==method)].iloc[0];v=im[(im.symbol=='AMZN')&(im.phase=='development')&(im.method==method)].iloc[0]
  lines.append(f'| {label} | {inp} | {v.BA:.2%} | {r.BA:.2%} | {r.Brier:.4f} |')
 lines+=['',
 '固定训练模型，只替换当前聚合向量，两个时期均未改变AMZN方向。后续仅2个窗口的向量确实发生变化，且没有贡献方向增量。全部方向增量出现在当前向量未变的177个窗口。',
 '', '两个训练分支选中了不同正则强度：AMZN N0M C=1，N1M C=0.01，后者正则强100倍。因此这个试验分离了“预测时输入变化”和“过去聚合改变后重新训练＋重新选参”，尚未把训练表示与C变化彼此分离。不能声称已证明去重的经济因果效应。','',
 f'后续共改对{int((after&~before).sum())}、改错{int((before&~after).sum())}，涉及{changed_days}个日期；贡献最大的3天合计{100*top:.2f}个百分点，占净BA增量约{100*top/total:.1f}%。事后去掉这3天的敏感性差为{100*sensitivity:.2f}个百分点；这不替换保留全部窗口的正式成绩。','',
 '| 月份 | N0M BA | N1M BA | 窗口数 |','|---|---:|---:|---:|']
 for month,x in monthly[(monthly.symbol=='AMZN')&(monthly.phase=='later')].groupby('month'):
  r=x[x.method=='N0M'].iloc[0];q=x[x.method=='N1M'].iloc[0];lines.append(f'| {month} | {r.BA:.2%} | {q.BA:.2%} | {int(r.n)} |')
 lines+=['','February只有2窗口，不当作完整月份稳定性证据。训练期晋级失败的结果不因这几个后续月份而改变。','',
 '| 配对块 | 后续AMZN BA差95%区间 | Brier差95%区间 |','|---|---|---|']
 for r in ci[(ci.symbol=='AMZN')&(ci.phase=='later')&(ci.method=='N1M')].itertuples():lines.append(f'| {r.block_days}日 | [{100*r.BA_low:+.2f}, {100*r.BA_high:+.2f}] pp | [{r.Brier_low:+.4f}, {r.Brier_high:+.4f}] |')
 lines+=['','BA区间跨0。所有区间是暴露回测的敏感性记录，不是排除探索偏差后的显著性证明。','',
 '## 案例：输入事实与模型行为','',
 '按预定类别内样本键哈希取例，共8个槽位、6个不同窗口。以下阅读对应标题和元信息，不冒充完整正文/独立盲审；这些案例不参与选参。','',
 '- 2019-01-08：长期股价观点与平台竞争新闻。两篇文章、两个簇，无当前去重；概率0.606→0.499，真实下跌，改对。不能说是删了重复新闻修复了这一例。',
 '- 2019-01-03：会员信息披露和拆股评论。两篇不同新闻、两个簇，0.591→0.488，真实下跌，改对。',
 '- 2019-01-31：财报预告与车载内容合作。两篇不同新闻，0.621→0.493，真实上涨，改错。',
 '- 2019-01-16：云业务利润率与招聘新闻，0.549→0.478，真实上涨，改错。这体现训练后的整体倾向变化也会引入错误。',
 '- 两个无新闻案例中N0M/N1M概率逐值相同，验证收益不来自更换无新闻回退。','',
 '## 课程报告与演示','',
 '[课程报告主稿](../../../docs/COURSE_REPORT_4H.md)以F0为baseline、F1为主要方法、F2为现代语义对照，纳入A1与概率校准消融。题目、方法、实验、案例、局限及演示讲稿已串联；成员信息和最终教师版式要求尚未代填。','',
 '[离线demo](demo.html)是609个development/later窗口的保存预测回放，非实时推理。可选择股票/时期/窗口，查看截止与目标区间、各模型概率、新闻与事件质量状态，手动揭示实际标签。原文与权重不在公开HTML中；无事件状态不表示原始新闻不存在。','',
 '观察：组合概率质量改善但训练期AMZN仍未过关；聚合模型收益来自训练/选参变更。解释：收缩可能减少了历史词/向量偶然关系，但尚未固定C来单独识别该效应。下一步以课程交付为主，不自动扩大模型搜索。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 # Durable course main draft, grounded in current frozen tables.
 course='''# CSE 573: Four-hour stock direction from prices and news

## Abstract

We compare price and news representations for AAPL and AMZN using 1,607 fixed four-hour windows. A chronological training protocol controls vocabulary, PCA, classifier selection and probability calibration. Full-text sparse features provide a useful reference improvement over a title baseline, while frozen FinBERT, recent-price features, news aggregation and event adaptation expose different tradeoffs between direction accuracy and probability quality. Event-task adaptation improves provisional extraction F1 to 79.17%, but does not establish incremental four-hour prediction value. Positive temperature scaling reduces overconfidence while preserving decisions. All evaluation periods have been exposed during exploration, so results are historical replay rather than independent deployment evidence.

## 1. Task and data

For each stock, predict whether the close at the end of a four-hour target exceeds its opening price. The information cutoff is five minutes BEFORE target start; the future actual open is never an input. News enters only after its recorded availability. Completed price bars and the original target labels are preserved.

The fixed universe contains 1,607 windows: 998 original training windows, 252 September–October development windows and 357 November-onward later windows. Of training, 233 January–February windows initialize the model; 765 March–August windows receive chronological forward predictions. Overlapping windows and repeated news reduce effective independence. AAPL dominates the supplied news corpus, and AMZN has much weaker news/event coverage.

## 2. Methods

F0 combines historical prices with title TF-IDF and regularized logistic regression. It is the project's operational classical baseline, not a claim of full replication of a published system. F1 replaces titles with full-body binary word features, with training-only chi-square selection. F2 uses frozen FinBERT article representations, training-only PCA16, article averaging and the same classifier family. With no original news, all these systems use the identical recent-price R1 fallback.

R1 adds cutoff-safe 5/15/30/60-minute returns, ranges, volatility and missingness. Joint experiments insert these into the text classifier rather than averaging two predictions. News aggregation compares article means against conservative near-republication groups, with matched count/source/age metadata. Positive temperature scales logits without changing the 0.5 decision; its scalar is fitted only to earlier OOF probabilities of that exact branch.

The event adapter predicts target-specific evidence, rating/target-price type and action. Numbers are copied and checked by code. Frozen heads, top-two-layer adaptation and rank-4 q/v LoRA were compared with three seeds. A1 was selected on extraction-training forward folds; labels remain dual-model provisional labels with assistant adjudication, not independently reviewed human gold.

## 3. Evaluation

We report balanced accuracy (mean up/down recall), MCC, Brier probability error, AUC, class recalls, monthly and coverage results. Transformations and hyperparameters use past data. Development and later results remain separate. Final models freeze at August; no later-label tuning or per-stock hindsight winner assignment is permitted.

The resource gate requires at least three forward months, positive macro BA differences in at least two, mean BA gain of one percentage point, no stock losing more than one point, and Brier worsening no more than 0.002 per stock and overall. This controls exploration, not statistical significance. Day/block paired intervals reflect serial dependence imperfectly and do not eliminate repeated-exploration bias.

## 4. Main results

'''+markdown([('F0','F0 title baseline'),('R1','R1 recent price'),('F1','F1 full text'),('B0','F2 frozen FinBERT'),('B1','F2 + temperature'),('B2','R1 + FinBERT'),('B3','R1 + FinBERT + temperature'),('N0M','Article mean + metadata'),('N1M','Event mean + metadata')])+'''

F1 is the most defensible simple full-text main method across the original four stock/period cells, but it does not improve every cell versus F0. F2 has better AAPL later BA but poor raw probability quality. The calibrated joint model preserves direction while reducing Brier; it still fails the training-period AMZN guardrail. Later-period improvements do not override that decision.

## 5. Event understanding is distinct from prediction

On 122 provisional April check articles (23 positive facts), exact fact F1 was 16.29% for current rules, 32.43% for Qwen1.7B QLoRA, 38.30% for staged Qwen9B, and 79.17% for A1. AAPL/AMZN A1 fact F1 was 75.68%/90.91%; AMZN has only six check facts. The coverage-limited oracle mapped to only 15 stock-training OOF event windows. Shared, separate and partially shared downstream corrections did not beat zero correction; the final probabilities therefore remained exactly F1. Extraction improvement is demonstrated against provisional labels, not formal independent acceptance or a market forecast improvement.

## 6. Case study and ablation insight

AMZN event aggregation reaches 57.32% later BA versus the matched article+metadata 54.74%. However, swapping only current aggregation vectors under fixed models does not change a single AMZN direction. Event-trained versus article-trained models also selected different regularization strengths, C=0.01 versus C=1. Thus training and selection changes, rather than direct deletion of duplicate current news, explain the computational path to the changed decisions. Their separate contributions remain unresolved.

Long-term opinion and competition news can become a corrected bearish call even when two distinct articles remain two groups. Conversely earnings-preview and partnership articles can be changed from a correct bullish call to an incorrect bearish one. Evidence content, model contribution and realized market direction must be discussed separately.

## 7. Limitations and conclusion

All periods were exposed during exploration; modern pretrained models applied to 2018 data may carry later pretraining knowledge. Availability timestamps are source metadata, not externally authenticated first disclosure. Near-republication grouping uses titles and cannot guarantee full-body event identity. Small, correlated samples and scarce AMZN events constrain statistical power. No trading-profit claim follows from these direction metrics.

Our contribution is a reproducible, controlled pipeline and component analysis: richer text sometimes helps; better event understanding does not necessarily help four-hour prediction; calibration improves confidence without inventing direction information; aggregation gains require retraining-aware attribution. We retain F0, F1 and F2 as the principal comparison, with the adapter and finite combinations as transparent ablations. Further models were not automatically launched when the training gate failed.

## Reproduction and presentation

- Latest experiment: [report](../outputs/stock_combination_4h/v1/REPORT.md), [protocol](../outputs/stock_combination_4h/PRE_REGISTRATION.md).
- Full core comparisons: [Phase 1](../outputs/stock_paper_methods_4h/v1/REPORT.md), [event adapter](../outputs/stock_finbert_event_adapter_4h/v1/REPORT.md).
- Offline replay: [demo](../outputs/stock_combination_4h/v1/demo.html). It displays saved historical predictions, not new live forecasts.
- Suggested demo: choose AAPL later; compare F2 and temperature (same direction, different confidence); choose AMZN later and a no-news window (strict fallback); reveal label; show aggregation success and failure in the report.
- Raw data and weights remain local; public source and model manifests alone do not permit full retraining without the course data.

This is the report main draft. Group names/contributions and the instructor's final page/template/submission requirements must be filled from the authoritative course instructions; no submission has been made.
'''
 (ROOT/'docs/COURSE_REPORT_4H.md').write_text(course)
 # All-inline, no network or raw news. Label disclosure is presentation-only.
 demo=d[d.phase.isin(['development','later'])].copy();demo['start']=demo.key.str.split('|').str[1]
 demo['cutoff']=(pd.to_datetime(demo.start,utc=True)-pd.Timedelta('5min')).astype(str);demo['end']=(pd.to_datetime(demo.start,utc=True)+pd.Timedelta('4h')).astype(str)
 payload=json.loads(demo.to_json(orient='records'));data=json.dumps(payload,separators=(',',':')).replace('</','<\\/')
 html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CSE 573 · 四小时历史回放</title>
<style>body{font:16px/1.6 system-ui;margin:0;background:#f3f5f8;color:#15263d}main{max-width:1060px;margin:auto;padding:32px}h1{font-size:30px}p{max-width:850px}.notice{background:#fff3d8;padding:12px;border-radius:8px}.controls{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}select,button{font:inherit;padding:8px;border:1px solid #aab7c9;border-radius:6px}label{display:grid;gap:5px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}.card{background:white;border:1px solid #dce3ec;border-radius:10px;padding:18px}.prob{font-size:28px;font-weight:700}.bar{height:8px;background:#edf0f6}.bar span{display:block;height:100%;background:#3568bd}#timing,#coverage,#actual{margin:18px 0}small{color:#56677d}a{color:#245eb8}</style>
<main><small>CSE 573 · AAPL / AMZN · 4 HOURS</small><h1>价格、新闻与预测信心</h1><p class="notice">保存预测的离线历史回放 · 开发期和后续期均已暴露 · 此页面不执行实时推理。方向按上涨概率 ≥ 50% 判为上涨。</p>
<div class="controls"><label>股票<select id="stock"><option>AAPL</option><option>AMZN</option></select></label><label>时期<select id="phase"><option value="later">后续期（已暴露）</option><option value="development">开发期（已暴露）</option></select></label><label>信息条件<select id="news"><option value="all">全部窗口</option><option value="yes">有新闻</option><option value="no">无新闻</option></select></label><label>目标起点 · UTC<select id="window"></select></label></div>
<div id="timing"></div><div id="coverage"></div><div class="cards" id="cards"></div><p><button id="reveal">显示实际方向</button></p><div id="actual" hidden></div>
<p><small>温度使用对应模型过去OOF拟合；无新闻保持R1。合格事件是暂定A1检查状态，并非独立人工验收；事件数量不直接证明四小时预测有效。</small></p><p><a href="REPORT.md">实验报告</a> · <a href="../../../docs/COURSE_REPORT_4H.md">课程报告主稿</a> · <a href="metrics.csv">指标数据</a></p></main>
<script>const data=__DATA__;const $=id=>document.getElementById(id);let current;const methods=[['F0','标题 baseline'],['F1','价格＋全文'],['B0','价格＋FinBERT'],['B1','FinBERT＋温度'],['B2','近期价格＋FinBERT'],['B3','近期价格＋FinBERT＋温度'],['N0M','文章等权＋元信息'],['N1M','事件组等权＋元信息']];
function choices(){const rows=data.filter(r=>r.symbol===$('stock').value&&r.phase===$('phase').value&&($('news').value==='all'||r.has_original_news===($('news').value==='yes'?1:0)));$('window').replaceChildren(...rows.map(r=>{let o=document.createElement('option');o.value=r.key;o.textContent=r.start;return o}));render()}
function render(){current=data.find(r=>r.key===$('window').value);$('actual').hidden=true;$('reveal').textContent='显示实际方向';if(!current){$('cards').textContent='此筛选条件没有窗口';$('timing').textContent='';$('coverage').textContent='';$('reveal').disabled=true;return} $('reveal').disabled=false;const r=current;$('timing').textContent=`信息截止 ${r.cutoff} ｜ 目标 ${r.start} → ${r.end}`;$('coverage').textContent=`原始新闻：${r.articles}篇 ｜ 近似转载组：${r.clusters} ｜ 可用文本：${r.has_usable_text?'有':'无'} ｜ 暂定合格事件：${r.has_qualified_event===null?'未知':r.has_qualified_event?'有':'无'}${r.has_original_news?'':' ｜ 无新闻：严格回退R1'}`;$('cards').replaceChildren(...methods.map(([k,name])=>{let c=document.createElement('div');c.className='card';let n=document.createElement('div');n.textContent=name;let p=document.createElement('div');p.className='prob';p.textContent=(100*r[k]).toFixed(2)+'%';let t=document.createElement('small');t.textContent='上涨概率 · '+(r[k]>=.5?'预测上涨':'预测下跌');let bar=document.createElement('div');bar.className='bar';let b=document.createElement('span');b.style.width=(100*r[k])+'%';bar.append(b);c.append(n,p,t,bar);return c}));$('actual').textContent='历史实际方向：'+(r.label?'上涨':'下跌')+'。单例正确不证明因果或泛化。'}
['stock','phase','news'].forEach(id=>$(id).addEventListener('change',choices));$('window').addEventListener('change',render);$('reveal').addEventListener('click',()=>{$('actual').hidden=!$('actual').hidden;$('reveal').textContent=$('actual').hidden?'显示实际方向':'隐藏实际方向'});choices();</script></html>'''.replace('__DATA__',data)
 (P/'demo.html').write_text(html)
 print('Wrote experiment report, course main draft and 609-window standalone demo')
if __name__=='__main__':main()
