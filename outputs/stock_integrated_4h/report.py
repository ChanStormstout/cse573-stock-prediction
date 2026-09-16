from run import *
OUT=B/'runs/v1'
def intervals(p):
 rows=[];rng=np.random.default_rng(573)
 # Both stocks share the same sampled date sequence. All intraday origins are retained.
 for period,g in [('outer',p[p.phase=='outer']),('validation',p[(p.phase=='frozen')&(p.split=='validation')]),('test',p[(p.phase=='frozen')&(p.split=='test')])]:
  dates=sorted(g.day.unique());n=len(dates)
  for block in [1,5,10]:
   starts=rng.integers(n,size=(10000,int(np.ceil(n/block))));ids=((starts[:,:,None]+np.arange(block))%n).reshape(10000,-1)[:,:n];w=np.zeros((10000,n));np.add.at(w,(np.arange(10000)[:,None],ids),1)
   for sym,sg in g.groupby('symbol'):
    scores={}
    for method in ['title','integrated','multi_only']:
     x=pd.DataFrame(dict(day=sg.day,tp=((sg[method]>=.5)&(sg.label==1)).astype(int),tn=((sg[method]<.5)&(sg.label==0)).astype(int),pos=sg.label,neg=1-sg.label,err=(sg[method]-sg.label)**2,n=1));a=w@x.groupby('day').sum().reindex(dates,fill_value=0).to_numpy();scores[method]={'BA':.5*(a[:,0]/a[:,2]+a[:,1]/a[:,3]),'Brier':a[:,4]/a[:,5]}
    for method in ['integrated','multi_only']:
     for measure in ['BA','Brier']:
      delta=scores[method][measure]-scores['title'][measure];lo,hi=np.nanquantile(delta,[.025,.975]);rows.append(dict(period=period,symbol=sym,method=method,metric=measure,block=block,lower=lo,upper=hi))
 pd.DataFrame(rows).to_csv(OUT/'paired_intervals.csv',index=False)
 return pd.DataFrame(rows)

def main():
 p=pd.read_csv(OUT/'predictions.csv');m=pd.read_csv(OUT/'metrics.csv');choices=json.loads((OUT/'selection.json').read_text());status=json.loads((OUT/'status.json').read_text());ci=intervals(p)
 names={'price':'价格 LR','title':'价格＋标题TF-IDF：baseline','body':'价格＋全文词袋','semantic':'价格＋FinBERT语义PCA16','integrated':'完整选择流水线','multi_only':'至少两分支组合','no_semantic':'去掉FinBERT后重新选权重','no_body':'去掉全文后重新选权重','no_title':'去掉标题后重新选权重','uniform':'四分支等权'}
 def table(period,phase='frozen'):
  x=m[(m.phase==phase)&(m.period==period)].copy();x['方法']=x.method.map(names);x['BA']=x.BA*100
  x=x.pivot(index='方法',columns='symbol',values=['BA','Brier']);return x.to_markdown(floatfmt='.4f')
 selected=[x for x in choices if x['month']=='final' and x['branch'] in ['integrated','multi_only','no_semantic','no_body','no_title']]
 # Freeze deployable bundles; price/title/body/semantic learners already trained.
 verification=[]
 inputs=pd.read_pickle(OUT/'inputs.pkl');z=np.load(B/'prepared/articles.npz');keys=z['keys'];emb=z['embeddings'].astype(np.float64)
 fingerprint=json.loads((B/'prepared/manifest.json').read_text())
 for name,h in fingerprint['prepared_hashes'].items():assert sha(B/'prepared'/name)==h
 for sym,d in inputs[inputs.month>='2018-09'].groupby('symbol'):
  c=next(x for x in selected if x['symbol']==sym and x['branch']=='integrated');models={k:joblib.load(OUT/'models'/f'{sym}_final_{k}.joblib') for k in BRANCHES};prob=d[['key','has_news']].copy()
  for kind,obj in models.items():prob[kind]=obj['classifier'].predict_proba(obj['transform'].transform(d,keys,emb))[:,1]
  computed=prob.title.to_numpy() if c['weights'] is None else combine(prob,c['weights']);saved=p[(p.symbol==sym)&(p.phase=='frozen')].set_index('key').loc[d.key].integrated.to_numpy();error=float(np.max(np.abs(saved-computed)));assert error<1e-12
  joblib.dump(dict(symbol=sym,models=models,choice=c,embedding_sha256=sha(B/'prepared/articles.npz')),OUT/f'{sym}_system.joblib');verification.append(dict(symbol=sym,end_to_end_reload_error=error,n=len(d)))
 dump(OUT/'system_verification.json',verification)
 q=p.copy();base=(q.title>=.5)==q.label;new=(q.integrated>=.5)==q.label;q['case_type']=np.select([~base&new,base&~new,~base&~new],['fixed','introduced_error','both_wrong'],default='both_correct');q.to_csv(OUT/'cases_all.csv',index=False);q['hash']=q.key.map(lambda s:hashlib.sha256(s.encode()).hexdigest());q.sort_values('hash').groupby(['symbol','phase','case_type']).head(8).to_csv(OUT/'case_panel.csv',index=False);q.groupby(['symbol','phase','case_type']).size().rename('n').to_csv(OUT/'case_counts.csv')
 weightrows=[]
 for x in selected:weightrows.append(dict(symbol=x['symbol'],method=x['branch'],weights=x['weights'],OOF_BA=x.get('BA'),OOF_Brier=x.get('Brier'),fallback=x.get('fallback','none')))
 weighttable=pd.DataFrame(weightrows).to_markdown(index=False,floatfmt='.4f')
 report=f'''# 四小时完整组合流水线与传统新闻 baseline

## 这次实际完成了什么
四分支训练 → 过去样本外预测 → 过去数据选融合权重 → 冻结完整系统 → 开发/后续历史回放 → 消融/案例/区间。完成 {status['LR_fits']} 次成功LR训练，运行 {status['seconds']:.1f} 秒。未使用一小时实验的权重或标签。

任务保持原四小时、1607窗口不变；每股训练499/499，开发126/126，后续178/179。完整系统并非把历次最高分拼接：它在每个截止点重新选择分支参数和组合权重。全部时段已参与过研究设计，本轮为探索性历史回放。

## 系统结构
- 价格：原16价格特征＋L2 LR。
- 标题 baseline：相同价格＋标题TF-IDF500＋L2 LR。
- 全文：相同价格＋既有清洗词干全文二值词袋、训练内卡方500＋L2 LR。
- 现代方法：相同价格＋冻结FinBERT文章向量、唯一训练文章PCA16、新闻数量/可用性＋L2 LR。
- 融合：四个概率非负加权；0/.25/.5/.75/1权重，共35个单纯形候选。无新闻严格回退价格。
- 选择：各分支C=.01/.1/1，按过去月份BA/Brier选择；融合使用严格前向分支OOF，Brier不比标题baseline恶化>.002的候选中选月均BA最高。没有合格项则保留原标题模型，不强迫新模型。

这里“机制协调”的含义是互补预测的保守融合、过去数据选择、无新闻回退、模型容量控制与端到端验证。未叠加没有收益的价格新特征、GRU、LoRA；独立复核未通过的目标正文抽取也未接入。全文是既有一般清洗正文，不冒充独立验收的事件抽取。

## 1. 开发时期：2018年9–10月
BA列单位为百分数，Brier越低越好。

{table('validation')}

## 2. 后续时期：2018年11月–2019年2月初
该时期已暴露，不能称新的独立测试。

{table('test')}

## 3. 外层向前评价：六月–八月合并窗口
本表是合并窗口BA；选参内部用月均BA，两者不必相等。

{table('all','outer')}

## 4. 最终冻结权重
顺序固定为[价格, 标题, 全文, FinBERT]。权重及C仅来自三月至八月前向预测；表内OOF分数用于选择，不当独立泛化分数。`multi_only`是预先声明的至少两分支对照，不能根据后续结果替换`integrated`。

{weighttable}

## 5. 组合相对标题baseline的不确定性
后续时期5交易日配对块区间，10000次，两股共享采样日期，包含全部日内窗口。BA差以0–1计，例如.01表示1个百分点。

{ci[(ci.period=='test')&(ci.block==5)].to_markdown(index=False,floatfmt='.4f')}

描述性区间未校正长期重复探索；另保存1日和10日块敏感性，不挑最有利区间。

## 6. 核验与产物
- 每次模型、PCA、词表、缩放只拟合过去训练前缀；训练标签结束必须早于评价首截止点。
- 原1607窗口、新闻记录身份和截止可用性检查；缓存标题逐条核对、模型revision/编码长度核对、缓存原哈希核对。
- 每次保存/重载概率误差≤1e-12且方向一致；独立装配的最终系统完整回放也验证一致。
- 冻结FinBERT向量转float64后重新在四小时训练文章上拟合PCA。新增缺失文章只做冻结推断。
- 保存全部候选、选择月份、逐窗口概率、逐月指标、权重、模型、输入和代码哈希。
- cases_all.csv与case_panel.csv覆盖改对/改错/共同错误/共同正确，尚未人工审阅全部案例。

## 重现
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 work/stock-data/finbert-env/bin/python outputs/stock_integrated_4h/run.py --output outputs/stock_integrated_4h/runs/reproduction
work/stock-data/finbert-env/bin/python outputs/stock_integrated_4h/test_pipeline.py
```
已有目录拒绝覆盖。此系统是可复现历史新闻预测流水线，非实时交易部署。已封装每股`*_system.joblib`；独立predict.py可从已准备特征表和原文章缓存重放，无需标签。
'''
 (B/'REPORT.md').write_text(report)
 print(pd.DataFrame(weightrows).to_string(index=False));print('system verification',verification)
if __name__=='__main__':main()
