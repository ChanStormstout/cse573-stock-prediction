from run import *
out=B/'runs/v1'
assert json.loads((out/'status.json').read_text())['state']=='complete'
d=pd.read_pickle(out/'data.pkl');p=pd.read_csv(out/'predictions.csv');m=pd.read_csv(out/'metrics.csv');ci=pd.read_csv(out/'paired_intervals.csv');cv=pd.read_csv(out/'cv.csv')
checks=json.loads((out/'checks.json').read_text());assert len(p)==3*len(d[~d.split.eq('train')]);assert not d.duplicated(['symbol','horizon','start_utc']).any()
assert np.isfinite(d[PRICE].to_numpy()).all();assert (d.history_end<=d.cutoff_utc).all();assert (d.cutoff_utc==d.start_utc-pd.Timedelta(minutes=5)).all()
# Shared-cutoff inputs must be identical, even though future labels differ.
for _,g in d.groupby(['symbol','start_utc']):
 for c in PRICE+['text','article_keys','news_count']:
  assert g[c].nunique()==1,(c,g)
for path,h in json.loads((out/'sources.json').read_text()).items():assert sha(Path(path))==h,path
# Recheck every saved model's probabilities and label-cutoff separation.
for rec in json.loads((out/'models.json').read_text()):
 stock,h,kind=rec['model'].removesuffix('.joblib').split('_',2);s=d[(d.symbol==stock)&(d.horizon==h)];b=s[~s.split.eq('train')]
 assert pd.Timestamp(rec['train_end'])<pd.Timestamp(rec['first_cutoff'])
 model=joblib.load(out/rec['model']);saved=p[(p.symbol==stock)&(p.horizon==h)&(p.method==kind)]
 np.testing.assert_allclose(model.predict_proba(b)[:,1],saved.p,atol=1e-12,rtol=0)
checks.append(dict(saved_models_reloaded=18,cv_fits=len(cv),shared_cutoff_inputs_equal=True,all_model_label_boundaries_valid=True,source_hashes_unchanged=True,finite_features=True))
dump(out/'verification.json',checks)
# Reference probabilities require no new fitting or model selection.
refs=[]
for (stock,h),s in d.groupby(['symbol','horizon']):
 prior=s[s.split.eq('train')].label.mean()
 for split,b in s[~s.split.eq('train')].groupby('split'):
  for kind,q in [('constant_half',.5),('training_prior',prior)]:refs.append(dict(symbol=stock,horizon=h,split=split,method=kind,**metric(b.label,np.full(len(b),q))))
pd.DataFrame(refs).to_csv(out/'reference_metrics.csv',index=False)
# Semantic increment is price_text relative to price_count, not just relative to price.
incs=[]
for (stock,h,period),g in m.groupby(['symbol','horizon','period']):
 g=g.set_index('method');incs.append(dict(symbol=stock,horizon=h,period=period,text_vs_price_ba=g.loc['price_text','ba']-g.loc['price','ba'],text_vs_count_ba=g.loc['price_text','ba']-g.loc['price_count','ba'],text_vs_price_brier=g.loc['price_text','brier']-g.loc['price','brier']))
pd.DataFrame(incs).to_csv(out/'increments.csv',index=False)
def table(frame):return frame.to_markdown(index=False,floatfmt='.4f')
counts=d.groupby(['symbol','horizon','split']).agg(n=('label','size'),days=('day','nunique'),news_ratio=('has_news','mean')).reset_index()
text=m[(m.method=='price_text')&m.period.isin(['validation','test'])][['symbol','horizon','period','n','ba','brier']]
lines=['# 预测周期比较：实际训练结果','', '**结论：本轮没有发现两只股票共同、跨时期稳定受益的更长周期；不据此更换项目主目标。** 四小时值得保留为辅助任务，日频暂不扩大搜索。所有分数都是已暴露历史回测，不是新独立测试。','', '## 实际执行','',f'- 18 个最终 L2 LR 模型，{len(cv)} 个训练期候选折拟合；三个周期、两股、三种输入。所有最终模型重载概率一致。','- 原 1 小时 3,233 行的样本、标签、价格输入、新闻标题逐项一致；旧实验文件未修改。','- 六月—八月选 C，训练变换仅拟合过去；九十月与十一月以后分开报告。','- 4h 每小时滚动，目标重叠；区间按共同交易日配对，提供1日/5日块敏感性。','', '## 样本与新闻覆盖', '',table(counts),'','日频包含短交易日；4h没有足够盘中时长的短交易日会排除。三者平盘分别排除，所以共同开盘面板只保留两股三种标签均合格的日期。AMZN日频原测试仅39.34%窗口有新闻，AAPL则100%；不能说所有日频都缺新闻。','', '## 全部匹配模型：开发与原测试', '',table(m[m.period.isin(['validation','test'])][['symbol','horizon','method','period','n','ba','mcc','brier','pred_up','constant']]),'', '## 价格＋数量＋标题文本：周期比较', '',table(text),'', '### 观察与解释','', '- AAPL 文本：原测试1h 46.65%、4h 53.02%、day 45.35%；4h价格自身已52.92%，文本只增加约0.10个百分点。不能归因为新闻理解改善。','- AMZN 文本：原测试1h 52.86%、4h 51.21%、day 48.98%；4h相对自己的价格基线46.20%有增量，但九十月49.72%低于价格50.09%。','- 日频每股训练仅167个日期，原测试61个；AAPL日频文本九十月55.00%而后来45.35%，再次显示只挑单段最高分的风险。','- 全样本原测试中，六个“价格＋文本”模型Brier全部高于固定0.5概率的0.25。方向分数与概率质量必须分开判断。','', '## 共同开盘日期面板', '',table(pd.read_csv(out/'common_open_metrics.csv')),'','AMZN 4h文本在共同开盘原测试58个日期上BA58.29%、Brier0.2396，但同面板九十月BA42.45%、Brier0.3036；新闻数量对照原测试已57.93%。不能将58.29%单独宣传为稳定文本优势。这个面板未重训开盘专用模型，训练期样本数量仍有差异，因此不是纯预测距离因果实验。','', '## 新闻相对价格：配对不确定性', '',table(ci),'','区间未校正探索和模型选择；区间跨0不能证明没有作用。部分月样本很小，完整逐月值在 metrics.csv，二月尾部不完整月份不得用于单月稳定性宣传。','', '## 建议与下一步','', '1. 保留1h主线与4h辅助对照；本轮不宣布最佳周期。改变项目主目标需要更稳定的月份证据。','2. 优先执行Pro提出的严格时间交叉拟合/数量校准匹配对照；不要因为AAPL 4h较高就同时叠加复杂模型。','3. 如继续日频，应另立24/72小时新闻回看实验，先检查覆盖和事件时效。当前4小时输入对照没有排除其他日频输入设计，但不应再无限搜索。','4. 获取真正新时期后，才有条件确认周期选择的泛化表现。','', '## 文件与复现', '', '- run.py：数据构造、训练、全候选保存、bootstrap。','- finalize.py：模型与输入再核验、动态表格与报告。报告观察段针对本次v1，换run需重新解读。','- runs/v1/protocol.json：事前协议；coverage.csv：全部候选窗口及排除原因。','- runs/v1/models.json：每个模型训练键、预测键、训练指标、哈希。','- runs/v1/predictions.csv / metrics.csv / common_open_metrics.csv / increments.csv：逐样本、逐月与对照增量。','- runs/v1/reference_metrics.csv：固定0.5和训练先验参考。','- 默认拒绝覆盖，完整命令见 README.md。']
(B/'REPORT.md').write_text('\n'.join(lines))
dump(out/'artifact_sha256.json',{str(x.relative_to(B)):sha(x) for x in B.rglob('*') if x.is_file() and '__pycache__' not in str(x) and x.name!='artifact_sha256.json'})
print('verified and report saved')
