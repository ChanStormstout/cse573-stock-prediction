"""Independent metric reconciliation and concise human-facing interpretation."""
from core import *
def main():
 p=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');metrics=pd.read_csv(OUT/'metrics.csv');maxerr=0
 for r in metrics.itertuples():
  g=p[(p.symbol==r.symbol)&(p.phase==r.phase)];y=g.label.to_numpy();q=g[r.method].to_numpy()>=.5;prob=g[r.method].to_numpy();tp=np.sum(q&(y==1));tn=np.sum(~q&(y==0));fp=np.sum(q&(y==0));fn=np.sum(~q&(y==1));ba=(tp/(tp+fn)+tn/(tn+fp))/2;den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)));mcc=(tp*tn-fp*fn)/den if den else 0;err=max(abs(ba-r.BA),abs(mcc-r.MCC),abs(np.mean((prob-y)**2)-r.Brier));maxerr=max(maxerr,err);assert err<1e-12
 evidence=json.loads((OUT/'training_evidence.json').read_text());counts={'LR_fits':0,'boosting_fits':0,'TabPFN_conditioning':0,'bilinear_final_fits':0,'bilinear_early_stop_fits':0};times={};hashchecks=0
 for method,record in evidence.items():
  times[method]=sum(f['seconds'] for f in record['fits'])
  for f in record['fits']:
   if 'seed_training' in f:
    for se in f['seed_training']:
     tr,ev=split(data(),f['symbol'],f['month']);path=PRIVATE/method/f"model_{tr[-1]}_{f['C']}_{se['seed']}.joblib";assert sha(path)==se['sha256'];hashchecks+=1
    if 'interaction' in method:counts['bilinear_final_fits']+=len(f['seed_training']);counts['bilinear_early_stop_fits']+=len(f['seed_training'])
    else:counts['LR_fits']+=len(f['seed_training'])
   else:
    name=f.get('model') or f.get('conditioning_bundle');assert sha(PRIVATE/method/name)==f['sha256'];hashchecks+=1
    counts['TabPFN_conditioning' if 'tabpfn' in method else ('boosting_fits' if 'hgb' in method else 'LR_fits')]+=1
 inference={path.stem:json.loads(path.read_text()) for path in PRIVATE.glob('dense_*.json') if path.name not in ('dense_inputs.json','dense_tokens.json','dense_encoding.json')};inference['history']=json.loads((PRIVATE/'history_embeddings.json').read_text());dump(OUT/'inference_evidence.json',inference)
 summary=dict(counts=counts,fit_seconds=times,independent_metric_rows=len(metrics),max_metric_error=maxerr,model_bundle_hashes=hashchecks,all_advanced=any(g['passed'] for g in json.loads((OUT/'advancement.json').read_text())),all_periods_exposed=True);dump(OUT/'execution_summary.json',summary)
 # Human-facing summary is regenerated idempotently.
 path=OUT/'REPORT.md';body=path.read_text();marker='## 中文结论';body=body.split(marker)[0].rstrip();section='''

## 中文结论

**这轮已经实际完成，但没有找到两股稳定超过60%的方法；没有机制通过预先规定的晋级线，因此最终没有强行组合。**

1. **跨股票信息与非线性：**跨股LR在六月至八月两股月均BA为56.88%/57.06%，却没有保持到后续AMZN（49.20%）。浅层树在AAPL训练期可达60.30%，AMZN仅44.05%。TabPFN同样未显示两股一致优势。
2. **A1稠密迁移：**同样正文、同样价格和线性头，A1相对原始FinBERT的外层月均BA为AAPL+0.10pp、AMZN+2.75pp；较弱股票的改善不足1pp。A1后续为54.34%/46.65%。不能把抽取能力的进步直接等同于市场预测能力。
3. **交互：**增加价格×正文的rank2交互没有通过对照；三个种子的分数和波动全部报告，没有挑最好种子。
4. **历史新闻：**AMZN开发/后续原无新闻的53/95个窗口全部找到合格过去新闻。但H1后续AMZN只有48.72%，H2为48.67%；覆盖改善没有转化为稳定方向增量。
5. **8+8组合：**修正为与F2_new完全相同的价格输入后，外层较弱股BA仅增加0.21pp，未达到1pp晋级线。早期混入R1价格的无效归因版本及其blend已保存为私有调试记录，不进入最终报告。
6. **F2_new：**实际发出概率的全局选参得到后续55.14%/56.79%，但AMZN开发为49.17%。这是局部结果，不能称为跨时期稳定提升。旧F0/F1/F2保留，历史分数没有被改写。

### 观察、解释与下一步

已观察到：训练期较高成绩未跨期保持、加入历史新闻能填补覆盖、A1的少量抽取适配不足以保证稠密股价表示有效。合理解释包括信息冗余、当前窗口和旧信息的关系不足、样本/市场条件变化；本轮不能确定它们各自的因果贡献，更不能证明不存在可预测信号。

下一项尚未完成的独立信息分支是合格SPY/QQQ市场数据。Alpaca固定2018日期试取返回401，需要账户后核验真实历史权限和完整性；本轮没有使用这部分数据，没有付费，也不以注册账户为已完成实验。现有方案不再扩网格追分。

所有结果仍是已经暴露的历史回测。课程主baseline继续F0；旧F1是统一全文参照，旧F2是现代语义参照。这里没有用后续期成绩给两只股票各自选择不同架构。
''';path.write_text(body+section)
 cases=OUT/'CASE_NOTES.md';base=cases.read_text().split('## Input inspection')[0].rstrip();cases.write_text(base+'''

## Input inspection of fixed AMZN no-current-news cases

- **2019-01-14 14:30 UTC:** nine older accepted articles. The latest titles include repeated stock-ranking/market-cap commentary, a multi-company recommendations roundup, and an institutional-holdings report. This supports a limited observation: historical coverage contains recurring commentary as well as potentially relevant company information. It does not establish a newly disclosed event or explain the four-hour outcome causally.
- **2018-11-09 16:30 UTC:** six older accepted articles. Titles include multi-company analyst commentary, a competitor-focused retail article, and two similar reports of Amazon retaking a market-cap ranking. These entered under the unchanged company-title filter. A filled window is therefore not equivalent to six independent new Amazon events.

This inspection checked titles and availability metadata of five most recent older articles per fixed case, not independent human gold or full-text extraction quality. Exact new-model probabilities and correctness are in cases.json. Historical-state aggregation deliberately did not introduce a new semantic event filter in this round, preserving the intended H0/H1/H2 comparison.
''');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
