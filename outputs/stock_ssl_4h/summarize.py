"""Regenerate interpretation and training-cost summaries without altering models."""
from run import *
def main():
 record=json.loads((PRIVATE/'done.json').read_text());rows=[]
 for r in record['encoders']:
  rows.append(dict(month=r['month'],seed=r['seed'],unlabeled_n=r['unlabeled_n'],endpoint_days=r['unique_endpoint_days'],epochs=r['selected_epochs'],best_validation_loss=min(a['validation_loss'] for a in r['early']['curve']),zero_validation_loss=r['early']['validation_zero_loss'],seconds=r['seconds'],parameters=r['final']['parameters']))
 pd.DataFrame(rows).to_csv(OUT/'pretraining_summary.csv',index=False);core.dump(OUT/'execution_summary.json',dict(encoder_early_stop_fits=21,encoder_final_fits=21,lr_fits=len(record['heads']),encoder_parameters=rows[0]['parameters'],encoder_training_and_encoding_seconds=sum(r['seconds'] for r in rows),device='cpu',labels_used_in_pretraining=False,independent_human_review=False,LLM_news_increment_executed=False))
 p=OUT/'REPORT.md';s=p.read_text().split('## 人话结论')[0].rstrip();s+='''

## 人话结论

**这次确实训练了，但没有得到可以替换现有价格模型的稳定改进。** 一个编码器仅1,988参数，最终使用2,172个重叠片段、169个端点日期；它们不等于2,172份独立市场样本。M5 CPU上21次内部早停＋21次最终预训练及编码约103秒，另实际拟合266个LR。

- 最终三种子的内部重建损失约0.2036，简单标准化零值对照0.2370。学习到了过去行情结构，但该目标与未来四小时方向仍有差距。
- 相对随机冻结表示，SSL外层AAPL−0.08pp、AMZN+4.77pp；这个有限匹配比较通过探索线。它没有胜过既有B，两股外层反而−0.32pp/−3.58pp，因此不晋级为新完整系统。
- SSL后续AAPL/AMZN为49.24%/50.66%，对照B为53.19%/50.26%。不能把重建效果或者AMZN一个局部子集的改善当成全任务成功。
- AMZN原无新闻95窗口：SSL修复旧F2的10个错误，新增6个错误，净+4；简单展平PCA修复8个、新增7个，净+1。子集是固定诊断，不能据此创建后续期专属路由。
- 没有扩大预训练轮数、改变任务或继续用后续期调参。训练期和后续期的差异仍需新时期验证。此轮不等于完整TS2Vec失败，也没有检验所有自监督目标。

LLM新闻增量组件目前仅完成设计（docs/LLM_NEWS_INCREMENT_PLAN.md），未提交ChatGPT网页标注、未新训练LLM，也未运行新闻差分下游实验。它应区分重复、背景、当前对象和有证据的事实变化；不得把“市场是否消化”当成可直接标注的事实。
''';p.write_text(s)
 cases=json.loads((OUT/'cases.json').read_text());lines=['# 固定案例诊断','', '复用上一轮按SHA256预先选定的案例键，不按本轮收益挑选。下表是预测变化，不是市场变化的因果解释。','', '| 窗口 | 标签 | B正确 | RAW正确 | SSL正确 |','|---|---:|---|---|---|']
 for r in cases:lines.append(f"| {r['key']} | {r['label']} | {r['correct']['B']} | {r['correct']['RAW']} | {r['correct']['SSL']} |")
 lines+=['','AMZN后续原无新闻窗口固定95个；SSL相对旧F2净多对4个，但不能把这些已经观察到的窗口用于选择开关。完整共同错误/新增错误计数见error_transitions.csv。','', '输入异常与市场预测错误分开：缺少完整48根输入时严格回退B；具备完整输入且预测错，只能证明该预测错，不能自动归因于某种价格形态。'];(OUT/'CASE_NOTES.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
