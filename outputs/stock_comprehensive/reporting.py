from core import *
import html

def paired_intervals(a,b,draws=2000):
    z=a.merge(b,on=['symbol','start_utc','label'],suffixes=('_a','_b'),validate='one_to_one');z['day']=z.start_utc.str[:10];stocks=sorted(z.symbol.unique());days=sorted(set.intersection(*[set(z.loc[z.symbol.eq(s),'day']) for s in stocks]));z=z[z.day.isin(days)];mats=[]
    for stock in stocks:
        s=z[z.symbol.eq(stock)];records=[]
        for day in days:
            g=s[s.day.eq(day)];y=g.label.to_numpy();pa=g.p_a.to_numpy();pb=g.p_b.to_numpy();records.append([sum(y==1),sum(y==0),sum((y==1)&(pa>=.5)),sum((y==0)&(pa<.5)),sum((y==1)&(pb>=.5)),sum((y==0)&(pb<.5)),sum((pa-y)**2),sum((pb-y)**2),len(g)])
        mats.append(np.asarray(records))
    output=[]
    for block in [1,5]:
        rng=np.random.default_rng(573);starts=rng.integers(0,len(days)-block+1,size=(draws,int(np.ceil(len(days)/block))));idx=(starts[:,:,None]+np.arange(block)).reshape(draws,-1)[:,:len(days)];diffs=[];briers=[]
        for mat in mats:
            q=mat[idx].sum(axis=1);diffs.append(.5*((q[:,2]-q[:,4])/q[:,0]+(q[:,3]-q[:,5])/q[:,1]));briers.append((q[:,6]-q[:,7])/q[:,8])
        diff=np.mean(diffs,axis=0);br=np.mean(briers,axis=0);output.append(dict(block_days=block,common_days=len(days),draws=draws,BA_delta_95=np.quantile(diff,[.025,.975]).tolist(),Brier_delta_95=np.quantile(br,[.025,.975]).tolist(),interpretation='paired common-day descriptive interval conditional on selected models; no exploration adjustment'))
    return output

def evaluate(run):
    out=run/'evaluation'
    if out.exists():raise FileExistsError(out)
    out.mkdir();summary=[];monthly=[];interval=[];panels=[];counts=[];allp=[]
    for mechanism in ['M02','M03','M08']:
        verify(json.loads((run/mechanism/'hashes.json').read_text()));p=pd.read_csv(run/mechanism/'predictions.csv');p['mechanism']=mechanism;allp.append(p)
        for (stock,method,split),g in p.groupby(['symbol','method','split']):summary.append(dict(mechanism=mechanism,symbol=stock,method=method,split=split,coverage=g.has_news.mean(),**metrics(g.label,g.p)))
        for (stock,method,month),g in p.groupby(['symbol','method',p.start_utc.str[:7]]):monthly.append(dict(mechanism=mechanism,symbol=stock,method=method,month=month,**metrics(g.label,g.p)))
        contrasts=([(f'k{k}_{pen}',f'k500_{pen}') for k in [25,100] for pen in ['l1','l2']] if mechanism=='M02' else [(x,'frozen') for x in ['coefficients','transforms','retuned']] if mechanism=='M03' else [('fusion','price'),('fusion','text8')])
        for alt,base in contrasts:
            for split in ['validation','test']:
                a=p[p.method.eq(alt)&p.split.eq(split)];b=p[p.method.eq(base)&p.split.eq(split)]
                for subset in ['macro','AAPL','AMZN']:
                    aa=a if subset=='macro' else a[a.symbol.eq(subset)];bb=b if subset=='macro' else b[b.symbol.eq(subset)]
                    interval.append(dict(mechanism=mechanism,alternative=alt,baseline=base,split=split,stock=subset,intervals=paired_intervals(aa,bb)))
        alt,base={'M02':('k100_l2','k500_l2'),'M03':('coefficients','frozen'),'M08':('fusion','price')}[mechanism]
        a=p[p.method.eq(alt)];b=p[p.method.eq(base)];z=a.merge(b,on=['symbol','start_utc','label','split'],suffixes=('_new','_base'));goodnew=(z.p_new>=.5)==z.label;goodbase=(z.p_base>=.5)==z.label;z['outcome']=np.select([goodnew&~goodbase,~goodnew&goodbase,~goodnew&~goodbase],['fixed','broken','both_wrong'],default='both_correct');z['mechanism']=mechanism;z['new_method']=alt;z['base_method']=base
        counts.append(z.groupby(['mechanism','symbol','split','outcome']).size().rename('n').reset_index())
        for (stock,split,outcome),g in z.groupby(['symbol','split','outcome']):
            # Fixed chronological, month-diverse selection; no manual selection on story appeal.
            q=g.sort_values('start_utc');first=q.groupby(q.start_utc.str[:7]).head(1);panels.append(pd.concat([first,q.drop(first.index)]).head(4))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(monthly).to_csv(out/'monthly.csv',index=False);dump(out/'paired_intervals.json',interval);pd.concat(allp).to_csv(out/'all_predictions.csv',index=False);pd.concat(counts).to_csv(out/'case_counts.csv',index=False);cases=pd.concat(panels);cases.to_csv(out/'cases.csv',index=False)
    d=load(run);news=pd.read_pickle(W/'audit/news_index.pkl');news['key']=news.archive+'::'+news.member;news=news.set_index('key');lookup=d.set_index(['symbol',d.start_utc.astype(str)]);cards=[]
    for r in cases.itertuples():
        row=lookup.loc[(r.symbol,r.start_utc)];ids=[k for k in str(row.news_record_keys).split('|') if k];evidence=[dict(key=k,title=news.loc[k,'title'],published=str(news.loc[k,'published_utc']),available=str(news.loc[k,'available_utc']),crawl_delay_hours=float(news.loc[k,'lag_hours'])) for k in ids[:5]]
        cards.append(dict(symbol=r.symbol,start=r.start_utc,cutoff=str(row.cutoff_utc),mechanism=r.mechanism,outcome=r.outcome,p_base=r.p_base,p_new=r.p_new,label=int(r.label),news_count=len(ids),evidence=evidence,input_error='not adjudicated',extraction_error='not applicable to these price/fulltext/semantic comparisons',prediction_error='observed label disagreement' if r.outcome!='both_correct' else 'none in this window',causal_explanation='unknown; case is not proof of cause'))
    dump(out/'case_cards.json',cards);dump(out/'status.json',{'state':'complete','cases':len(cards)})

def table(df):return df.to_markdown(index=False,floatfmt='.4f')
def report(run):
    out=run/'report'
    if out.exists():raise FileExistsError(out)
    out.mkdir();s=pd.read_csv(run/'evaluation/summary.csv');lc=pd.read_csv(run/'M02/learning_curve.csv');up=pd.read_csv(run/'M03/predictions.csv');rl=pd.read_csv(run/'M09/summary.csv');g=json.loads((run/'advanced_gate.json').read_text());q=json.loads((run/'M06/quality_gate.json').read_text());co=pd.read_csv(run/'M05/candidates.csv');tim=pd.read_csv(run/'M04_v2/summary.csv' if (run/'M04_v2').exists() else run/'M04/summary.csv')
    simple=s[s.split.eq('test')][['mechanism','symbol','method','n','balanced_accuracy','brier','constant_prediction']]
    lines=['# CSE 573 全面机制实验报告','', '**范围：探索性历史回测。没有新的独立测试期；所有分数均不得包装成独立泛化证明。**','', '## 完成与门槛','', 'M01 统一入口与不可覆盖目录；M02 容量/学习曲线；M03 更新因素；M04 事件时点；M05 原始覆盖与免费数据调查；M06 抽取与证据关系存储；M08 语义压缩/概率融合；M09 六个 PPO 均已执行。M07 实现了 offset 修正器及回退/门槛测试，但未通过独立事件质量门槛，未训练。高级预测门槛未通过，因此未启动注意力或新 LoRA。Graph 只保存证据关系，没有训练图模型。','', '## 分类结果：原已暴露测试时期（11月—次年2月1日）','',table(simple),'', '## 为什么尚不能宣布稳定改善','', '压缩词表降低了多数概率误差，但 BA 收益并不稳定。训练期门槛显示，k25 的均值改善主要集中在八月；k100 L2 虽有两个月正增量，平均仅 +0.66 个百分点，未达预设 +1 点。不能只用平均涨分进入更大模型。','', '固定变换后仅更新系数，也未获得一致提升；重新拟合变换/重新选参没有稳定解决问题。这使“只是参数陈旧”这一单一解释缺乏支持。','', '语义融合训练期选择：AAPL 文本权重为 0，输出严格等于价格模型；AMZN 权重为 0.75、PCA8。AMZN 的已暴露时期 BA 为 52.27%，Brier约0.2501；这是局部结果，训练期跨月门槛未通过。','', '## 学习曲线：同历史跨度、完整共同交易日抽样','',table(lc.groupby(['symbol','fraction'])[['train_balanced_accuracy','balanced_accuracy','train_brier','brier']].mean().reset_index()),'', '三种子是日期抽样重复；100% 数据三次相同，不是三份独立数据。六月/七月/八月折本身有依赖，曲线只能支持诊断，不能证明增加数据必然改善。','', '## 新闻到达时点','',table(tim[(tim.context=='intraday')&(tim.horizon==60)][['level','symbol','late','total','valid','mean_return','median_abs_return','mean_vol']]),'', '全文响应表包含前60分钟、后15/30/60/120分钟。到达后向上取整至下一个5分钟开盘；不跨缺失线或正常交易时段。盘外新闻保留在覆盖分母，但没有硬接到次日开盘。近似事件组为在线标题代理，尚未人工认证，统计不代表因果效应。','', '## 覆盖检查','',table(co.groupby(['symbol','valid_base','title_hit','body_hit']).size().rename('n').reset_index()),'', '这里是去重前原始记录计数，不是可训练事件数。candidate_review.csv 的直接/比较/顺带类别是启发式候选标签；新增正文提及没有直接纳入模型。免费来源可行性见 EXTERNAL_DATA.md；本轮未建立新保留期。','', '## 抽取、事件修正与 Graph','', '修复方向：机构名称有限规范化，无法可靠识别则空值；评级必须有正文旧/新评级配对；标题纠错冲突时放弃动作；目标价与评级分开；数值方向、货币单位、会计期间显式检查；保留未知首次披露时间及规范化正文证据偏移。','', '完整事件没有独立复核，质量门槛保持未通过。开发/检查按在线事件组分开，过去已审阅组全部归开发。检查材料包含拒绝/未知字段，不删除失败字段来宣称验收通过。M07 无事件严格回退；正式训练被门槛阻止。','', '## 独立 RL 交易实验','',table(rl[rl.period.eq('test')&rl.cost_bps.eq(10)][['symbol','method','steps','days','net_return','max_drawdown','turnover','cash_fraction']]),'', '六个 PPO 各训练20,000步，三种子全部报告。单层16单元 actor/critic、学习率3e-4；本地CPU。日内交易、日终清仓，收益连接相邻成交开盘，末段到最后完整小时收盘；5/10/20bps含双向换仓和清仓成本。原平盘小时保留；缺线日不参与交易模拟。','', '**所有 PPO 在10bps已暴露回测均亏损；空仓基线为0。** 这轮 RL 没有证明持仓决策优势，更不能代替原方向预测指标。净收益未包含真实流动性、冲击、借券约束，重复行情训练也不增加独立样本。','', '## 误差案例和区间','', 'evaluation/case_cards.json 按机制、股票、时期、改对/改错/共同错/共同对固定选取，月份优先且按时间排序。保留原始时间、新闻标题/键和概率，不把故事当成因果解释。paired_intervals.json 提供共同交易日配对重采样、单日和5日块敏感性，条件于已选模型，不校正多轮探索选择。','', '## 决定','', '完成可靠工程与课堂交付，保留所有结果。当前不扩大网格，不启动条件未满足的高级模型。可以将“减少容量改善概率质量但没有稳定方向收益；模型更新不能单独解决；RL 成本后失败”作为清楚的实验结论。正式事件/Graph 等待组员独立复核；新的泛化确认等待合格的新时期。','', '## 重现与展示','', '使用根目录 README.md 的 prepare/train/evaluate/report/verify 命令。离线 demo.html 展示冻结历史预测、输入截止点和新闻证据；它不是实时投资预测服务。']
    (out/'REPORT.md').write_text('\n'.join(lines));build_demo(run,out)

def build_demo(run,out):
    d=load(run);p=pd.read_csv(run/'M08/predictions.csv');piv=p.pivot(index=['symbol','start_utc'],columns='method',values='p');n=pd.read_pickle(W/'audit/news_index.pkl');n['key']=n.archive+'::'+n.member;n=n.set_index('key');rows=[]
    for x in d[~d.split.eq('train')].itertuples():
        values=piv.loc[(x.symbol,str(x.start_utc))];ids=[k for k in x.news_record_keys.split('|') if k];rows.append(dict(stock=x.symbol,time=str(x.start_utc),cutoff=str(x.cutoff_utc),split=x.split,label=int(x.label),base=float(values['price']),fusion=float(values['fusion']),text=float(values['text8']),news=[dict(title=n.loc[k,'title'],available=str(n.loc[k,'available_utc']),published=str(n.loc[k,'published_utc']),key=k) for k in ids]))
    payload=json.dumps(rows,ensure_ascii=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>CSE573 历史预测演示</title><style>body{font:17px system-ui;background:#f2f5f9;color:#17263c;max-width:1000px;margin:36px auto;padding:20px}section{background:white;border-radius:14px;padding:24px;margin:16px 0}select{font:inherit;max-width:100%;padding:9px}.cards{display:flex;gap:35px;flex-wrap:wrap}.big{font-size:30px;font-weight:700}small{color:#536780}li{margin:14px 0;overflow-wrap:anywhere}.tag{color:#8b4400}h1{font-size:28px}</style><h1>CSE 573 · 股票新闻与小时方向</h1><p class="tag">已暴露历史回测 · 非实时预测 · 完整事件抽取尚未独立验收</p><section><label>股票 <select id="stock"><option>AAPL</option><option>AMZN</option></select></label> <label>预测窗口 <select id="time"></select></label><p id="meta"></p></section><section class="cards"><div>价格基础上涨概率<div class="big" id="base"></div></div><div>冻结融合上涨概率<div class="big" id="fusion"></div></div><div>文本 PCA8 上涨概率<div class="big" id="text"></div></div></section><section><h2>输入与结果</h2><p id="truth"></p><p id="explain"></p><h3>截止前可用的新闻</h3><ul id="news"></ul></section><section><h2>适用范围</h2><p>AAPL / AMZN；原一小时方向；训练至2018年8月。参数和融合权重在训练期选择。没有新的独立保留期。RL 在独立交易报告中评价，不以交易收益替代方向准确率。</p></section><script>const data=PAYLOAD;const $=x=>document.getElementById(x);function options(){let rows=data.filter(x=>x.stock==$('stock').value);$('time').replaceChildren(...rows.map((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.time;return o}));show()}function show(){let r=data.filter(x=>x.stock==$('stock').value)[+$('time').value];$('meta').textContent='信息截止：'+r.cutoff+'；时期：'+r.split;for(let k of ['base','fusion','text'])$(k).textContent=(100*r[k]).toFixed(2)+'%';$('truth').textContent='历史实际方向：'+(r.label?'上涨':'下跌')+'；新闻数量：'+r.news.length;$('explain').textContent=r.stock==='AAPL'?'训练期选择文本权重0，所以融合严格等于价格模型。':'训练期选择文本权重0.75，价格权重0.25。局部收益不等于稳定改善。';$('news').replaceChildren(...r.news.map(n=>{let li=document.createElement('li');li.textContent=n.title+' ｜发布：'+n.published+' ｜可用：'+n.available;return li}));if(!r.news.length){let li=document.createElement('li');li.textContent='此窗口没有合格新闻。';$('news').append(li)}}$('stock').onchange=options;$('time').onchange=show;options();</script></html>'''.replace('PAYLOAD',payload)
    (out/'demo.html').write_text(page)
