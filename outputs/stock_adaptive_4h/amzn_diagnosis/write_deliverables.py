from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;A=B/'analysis_v1';ROOT=B.parents[2]
notes=json.loads((A/'content_notes_before_reveal.json').read_text());seal=json.loads((A/'content_notes_seal.json').read_text())
assert hashlib.sha256((A/'content_notes_before_reveal.json').read_bytes()).hexdigest()==seal['sha256']
for p,h in json.loads((A/'sources.json').read_text()).items():
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
panel=pd.read_csv(A/'case_panel_with_outcomes.csv');m=pd.read_csv(A/'metrics.csv');assert len(panel)==24 and panel.day.nunique()==24
interpretations={
'C01':'节日业绩推演、Alexa合作及长期展望混杂，还存在可疑金额单位。多个模型预测正确，不能因此认定文章数字可靠或模型读懂了事件。',
'C02':'没有新闻。标题/全文预测上涨而价格预测下跌；全文错误不是文本理解出错的实例，来自各模型不同的价格系数和截距。',
'C03':'Google新增销量与Amazon存量优势、三年翻倍展望混在一起。全文方向正确，但最大词贡献含baba/twitter/siri等；只能证明词影响了计算，不能证明理解了公司关系。',
'C04':'没有新闻。标题/价格正确、全文略高于0.5而错误，说明保留全文自己的价格部分也并非总是更好。',
'C05':'文章重述FAANG此前下跌及反弹，很多内容涉及Facebook。两模型都正确，无法归因于Amazon新事件。',
'C06':'没有新闻；多模型略偏上涨，共同漏掉较大跌幅。不能由新闻模块解决没有输入的信息。',
'C07':'没有新闻但全文正确、标题错误。正文特征严格为零，收益来自联合训练出的价格系数/截距。',
'C08':'两条8月29日报道9月4日才收到。正文把历史目标价上调等转成负向总贡献，实际区间上涨；旧闻与词语关联并存，尚未证明删旧闻可修复。',
'C09':'购物建议与探索医疗收购新闻，正文说明谈判已失败且涉及过去。全文/组合都正确；需要状态变化与否定关系，但这不证明正确预测来自该语义。',
'C10':'没有新闻，价格概率约0.5004、全文约0.536，两者均错。概率接近阈值和收益大小是两回事。',
'C11':'没有新闻，全文上涨概率52.87%正确；旧组合强制采用价格45.04%而改错。已精确核实全文文本贡献为零，不能把这次修复算作新闻价值。',
'C12':'财报预告明确为收盘后发布，目标区间却结束于收盘前。预期EPS不能当实际结果。全文预测下跌错误，组合上涨正确。',
'C13':'标题说避开Amazon，正文资金流为中性，Facebook乐观论述混入，两标题近似重复。多个模型上涨预测正确，不代表高置信度获得了事件证据支持。',
'C14':'政府支持公司否认芯片指控；正文重复原指控与此前下跌。模型偏涨但未来四小时下跌2.30%。正面更新并不保证短期上涨，不能保证抽取修复就修复方向。',
'C15':'EPS超预期，但营收、下一季指引低于预期；实际继续下跌。全文48.64%正确、组合58.50%错误。不过最大贡献是direct/momentum/year等一般词，不能声称模型真的学会了预期差。',
'C16':'Echo/AI产品更新与长期增长预测，首次披露时间不明确。全文略偏跌而错、组合略偏涨而对，属于长期事实和短期目标错配的例子。',
'C17':'没有新闻，多模型均正确预测上涨；作为正常价格预测案例保留。',
'C18':'没有新闻，多模型共同预测上涨却下跌；不是通过更好的新闻抽取就能解释的全部错误。',
'C19':'Amazon Pay比较和Graviton新品含当前劣势/未来机会的相反观点。标题52.12%正确，gate49.42%错误；gate只是价格，因为新闻分支被禁用。不能说选择器读错了本文语义，它根本未使用该修正。',
'C20':'Apple Music与Echo当前公布、未来上线，未来实施不等于旧闻。标题略低于0.5错误，gate正确；gate仍为价格，不应归功于语义理解。',
'C21':'9月5/6日市场下跌报道9月19日收到，Apple发布会预告也已过期。两模型仍正确，说明输入缺陷与预测错误不能逐条等同。',
'C22':'没有新闻，多模型共同错误。必须保留这类缺少文本解释的案例，避免只讲新闻故事。',
'C23':'没有新闻，标题略偏涨正确、价格/gate略偏跌错误；基线的不同价格拟合可以解释预测差，不涉及事件抽取。',
'C24':'没有新闻，实际下跌0.034%，标题错误、gate正确。小收益窗口仍保留，不能事后删去以抬高成绩。'}
attr={x['case_id']:x for x in json.loads((B/'addendum_v2/case_attributions.json').read_text())}
lines=['# AMZN 24个案例：内容检查后揭示结果','',
'选择方案先写入PROTOCOL.md，再按三组模型比较×两个时期×四种对错结果确定性抽取。24个窗口来自24个不同日期，13个有新闻、11个无新闻；阅读全部所选窗口标题及20篇正文节选（每篇至多5500字符）。内容笔记已先保存并封存，随后读取标签；不是独立人工盲审，也不能用此分层面板估计全体错误原因的比例。', '',
'以下概率都是上涨概率；≥50%判断上涨。时间为UTC；涨跌为原四小时目标。全文指旧联合模型，gate指最新零截距版新闻系统候选；最终选定系统仍是标题baseline。','']
for r in panel.itertuples():
    at=attr[r.case_id]
    lines += [f'## {r.case_id} · {r.start_utc} · {int(r.news_count)}条新闻', '',
      f'比较层：{r.comparison} / {r.case_outcome}；实际收益 **{r.target_return*100:+.3f}%**。', '',
      f'上涨概率：标题 {r.title*100:.2f}%｜价格 {r.price*100:.2f}%｜全文 {r.old_body*100:.2f}%｜旧组合 {r.old_integrated*100:.2f}%｜新gate {r.gate*100:.2f}%。', '',interpretations[r.case_id], '',
      f'全文模型精确logit分解：截距 {at["intercept"]:+.4f} + 价格 {at["price_logit"]:+.4f} + 文本 {at["text_logit"]:+.4f}。该分解仅解释线性模型计算，不解释市场因果。','']
lines += ['## 可追溯材料','',
'`analysis_v1/content_before_outcome.json`保留标题、可用/发布时间、原记录键及阅读范围；`content_notes_before_reveal.json`和`content_notes_seal.json`保留揭示结果前的笔记。`case_panel_with_outcomes.csv`保留全部模型概率和标签。`addendum_v2/case_attributions.json`保留24个模型分解和最大的5个词贡献；重载概率误差见verification.json。']
(B/'CASE_INTERPRETATIONS.md').write_text('\n'.join(lines)+'\n')
# A concise, auditable source receipt for the final answer.
main=m[(m.period=='test')&(m.slice=='all')&m.model.isin(['constant','title','price','old_body','old_semantic','old_integrated','gate'])]
paired=pd.read_csv(A/'paired_decomposition.csv');paired=paired[paired.period=='test'].groupby(['comparison','has_news'],as_index=False)[['first_wins','second_wins','BA_contribution']].sum()
cv=pd.read_csv(A/'training_selection_audit.csv');cv=cv[(cv.version=='zero_bias_v1')&(cv.kind=='body')&(cv.C==.01)]
receipt={'schemaVersion':1,'items':[]}
for id,title,files,cols,frame,caveat in [
 ('amzn-results','AMZN四小时：同179窗口对比',['metrics.csv'],['model','n','BA','Brier','up_recall','down_recall'],main,'所有历史已暴露；最新gate为价格回退，最终选定系统仍为标题baseline。'),
 ('paired-changes','组合在哪些窗口改变对错',['paired_decomposition.csv'],['comparison','has_news','first_wins','second_wins','BA_contribution'],paired,'BA贡献使用全体涨/跌类别分母，可相加；不能平均各子集BA。first_wins指第一模型正确而第二模型错误。'),
 ('training-gate','训练期为什么关闭AMZN新闻',['training_selection_audit.csv'],['kind','C','BA','Brier','base_Brier','Brier_difference','eligible'],cv,'允许Brier恶化最多0.002；此候选恶化0.003847而被拒绝。这里只说明工程选择，不证明关闭能提高准确率。')]:
    receipt['items'].append({'id':id,'title':title,'queries':[{'id':id+'-data','source':{'label':'保存预测和训练记录的复核','files':[{'label':f} for f in files],'caveats':[caveat]},'columns':cols,'rows':json.loads(frame[cols].to_json(orient='records'))}]})
(B/'sources_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
status={'status':'COMPLETE_EXPLORATORY_DIAGNOSIS','horizon':'4h','new_training':False,'population_windows':305,'later_windows':179,'later_dates':60,'case_windows':24,'case_dates':24,'news_cases':13,'no_news_cases':11,'article_excerpts_read':20,'excerpt_char_limit':5500,'content_notes_seal_verified':True,'independent_gold':False,'raw_experiment_sources_unchanged':True,'single_branch_omission_completed':'addendum_v2','model_attributions_reproduced':24,'original_selected_system_unchanged':'title','failed_attempt':'addendum_v1 saved selection; pickle __main__.Transform resolution error. Fixed class alias in v2; no fitted weights or selection rule changed.'}
(A/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(status,ensure_ascii=False,indent=2))
