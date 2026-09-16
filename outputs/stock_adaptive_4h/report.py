"""Recompute metrics, date-block intervals, diagnostics and fixed case panel."""
from core import *
import argparse,zipfile

NAMES={'title':'标题 baseline','price':'价格 baseline','old_integrated':'旧四小时组合','old_body':'旧价格＋全文','old_semantic':'旧价格＋FinBERT','body':'正文修正','semantic':'语义修正','joint_body':'允许价格再修正＋正文','joint_semantic':'允许价格再修正＋语义','static':'静态修正组合','gate':'条件修正组合','price_cal':'价格＋校准','static_cal':'静态组合＋校准','gate_cal':'条件组合＋校准','selected_system':'预设规则选择的最终系统','constant':'固定0.5','prior':'过去类别比例','online_weights':'每日误差更新权重','online_price':'价格＋每日截距更新','online_title':'标题＋每日截距更新'}

def metrics_frame(p):
    rows=[]
    for (symbol,phase),g in p.groupby(['symbol','phase']):
        periods=[('all',g)]+[(m,x) for m,x in g.groupby('month')]
        if phase=='frozen':periods += [(s,x) for s,x in g.groupby('split')]
        else:periods += [('outer',g[g.month>='2018-06'])]
        for period,x in periods:
            for method in NAMES:
                if method not in x or x[method].isna().any() or x.empty:continue
                rows.append({'symbol':symbol,'phase':phase,'period':period,'method':method,**score(x.label,x[method]),'news_coverage':float(x.has_news.mean()),'dates':x.day.nunique()})
    return pd.DataFrame(rows)

def block_intervals(p):
    rng=np.random.default_rng(573);out=[]
    for period,d in [('validation',p[p.split=='validation']),('test',p[p.split=='test'])]:
        days=sorted(d.day.unique());lookup={v:i for i,v in enumerate(days)};n=len(days)
        for block in [1,5,10]:
            starts=rng.integers(0,n,size=(10000,int(np.ceil(n/block))));indices=((starts[:,:,None]+np.arange(block))%n).reshape(10000,-1)[:,:n]
            def aggregate(g,method):
                a=np.zeros((n,5));y=g.label.to_numpy();q=g[method].to_numpy();ids=g.day.map(lookup).to_numpy()
                data=np.c_[(y==1)&(q>=.5),(y==0)&(q<.5),y==1,y==0,(q-y)**2]
                for j in range(5):np.add.at(a[:,j],ids,data[:,j])
                count=np.zeros(n);np.add.at(count,ids,1)
                r=a[indices].sum(1);den=count[indices].sum(1)
                with np.errstate(divide='ignore',invalid='ignore'):ba=.5*(r[:,0]/r[:,2]+r[:,1]/r[:,3]);br=r[:,4]/den
                return ba,br
            for sym,g in d.groupby('symbol'):
                baseba,basebr=aggregate(g,'title')
                for method in ['static','gate','static_cal','gate_cal','body','semantic','old_integrated','online_weights','online_title']:
                    ba,br=aggregate(g,method)
                    for metric,v in [('BA',ba-baseba),('Brier',br-basebr)]:
                        valid=v[np.isfinite(v)];lo,hi=np.quantile(valid,[.025,.975])
                        out.append({'symbol':sym,'period':period,'method':method,'metric':metric,'block_days':block,'draws':10000,'valid_draws':len(valid),'low':float(lo),'high':float(hi),'difference':score(g.label,g[method])[metric]-score(g.label,g.title)[metric]})
    return pd.DataFrame(out)

def main(run,online,out):
    run=run.resolve();online=online.resolve()
    out.mkdir(parents=True,exist_ok=False);p=pd.read_pickle(run/'predictions.pkl');op=pd.read_pickle(online/'predictions.pkl')
    cols=['key','online_weights','online_price','online_title'];p=p.merge(op[cols],on='key',how='left',validate='one_to_one')
    metrics=metrics_frame(p);metrics.to_csv(out/'metrics.csv',index=False)
    intervals=block_intervals(p[p.phase=='frozen']);intervals.to_csv(out/'paired_intervals.csv',index=False)
    diagnostics=[]
    for (sym,split),g in p[p.phase=='frozen'].groupby(['symbol','split']):
        for group,xx in [('all',g),('with_news',g[g.has_news==1]),('no_news',g[g.has_news==0])]:
            if xx.empty:continue
            for m in ['price','title','body','semantic','static','gate']:
                diagnostics.append({'symbol':sym,'split':split,'slice':group,'method':m,**score(xx.label,xx[m])})
    pd.DataFrame(diagnostics).to_csv(out/'news_slices.csv',index=False)
    mechanism=[];cases=[]
    for (sym,split),g in p[p.phase=='frozen'].groupby(['symbol','split']):
        for method in ['body','semantic','static','gate','static_cal','gate_cal','online_weights']:
            for ref in ['title','price','static']:
                good=(g[method]>=.5)==g.label;bg=(g[ref]>=.5)==g.label
                mechanism.append({'symbol':sym,'split':split,'method':method,'reference':ref,'repaired':int((good&~bg).sum()),'introduced':int((~good&bg).sum()),'common_wrong':int((~good&~bg).sum()),'common_correct':int((good&bg).sum()),'n':len(g)})
        q=g.copy();a=(q.gate>=.5)==q.label;b=(q.title>=.5)==q.label
        q['case_type']=np.select([a&~b,~a&b,~a&~b],['repaired','introduced','common_wrong'],default='common_correct')
        q['order']=q.key.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
        for kind,z in q.groupby('case_type'):
            for row in z.sort_values('order').drop_duplicates('day').head(4).to_dict('records'):cases.append(row)
    pd.DataFrame(mechanism).to_csv(out/'case_changes.csv',index=False)
    raw=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key');needed=set()
    for row in cases:
        ids=[k for k in row['news_record_keys'].split('|') if k];row['case_articles']=sorted(ids,key=lambda k:raw.loc[k,'available_utc'],reverse=True)[:2];needed.update(row['case_articles'])
    bodies={}
    for archive,g in raw.loc[sorted(needed)].groupby('archive'):
        with zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/archive) as z:
            for r in g.itertuples():bodies[r.Index]=json.loads(z.read(r.member)).get('text','')
    cards=[]
    for r in cases:
        card={k:r[k] for k in ['key','symbol','split','case_type','label','target_return','price','title','body','semantic','static','gate','db','ds','weight_zero','weight_body','weight_semantic','gate_reason','news_count']}
        card['articles']=[{'record_key':k,'title':raw.loc[k,'title'],'available_utc':str(raw.loc[k,'available_utc']),'published_utc':str(raw.loc[k,'published_utc']),'excerpt':bodies[k][:8000]} for k in r['case_articles']];cards.append(card)
    dump(out/'case_cards.json',cards)
    pd.DataFrame([{k:v for k,v in c.items() if k!='articles'} for c in cards]).to_csv(out/'case_panel.csv',index=False)
    txt=['# 固定案例面板：条件组合对比标题 baseline','按两股、两时期、四象限最多各4个不同日期确定性抽样；正文每窗最多两篇最新文章。不是人工全量审阅。']
    for i,c in enumerate(cards):
        txt += [f"\n## {i+1}. {c['key']} / {c['case_type']}",f"真实收益 {c['target_return']:.4%}；标题 {c['title']:.4f}；价格 {c['price']:.4f}；正文 {c['body']:.4f}；语义 {c['semantic']:.4f}；静态 {c['static']:.4f}；条件 {c['gate']:.4f}。",f"零/正文/语义权重 {c['weight_zero']:.3f}/{c['weight_body']:.3f}/{c['weight_semantic']:.3f}；状态 {c['gate_reason']}；新闻 {c['news_count']}。"]
        for a in c['articles']:txt += [f"### {a['title']}",f"可用 {a['available_utc']} / 发布 {a['published_utc']}",a['excerpt']]
    (out/'CASE_CARDS.md').write_text('\n\n'.join(txt))
    selection=json.loads((run/'selection.json').read_text());final=[x for x in selection if x['month']=='2018-09'];dump(out/'final_components.json',final)
    gatecounts=p[p.phase=='frozen'].groupby(['symbol','split','gate_reason']).size().reset_index(name='windows');gatecounts.to_csv(out/'gate_usage.csv',index=False)
    # Plain markdown tables avoid an optional formatting dependency.
    def table(data):
        head='| '+' | '.join(data.columns)+' |\n|'+ '|'.join(['---']*len(data.columns))+'|\n'
        return head+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in data.itertuples(index=False,name=None))
    report=['# 四小时新闻增量与条件组合：执行结果','所有结果来自已暴露历史，非新的独立测试。训练、抽取质量、预测收益分别记录。','## 冻结后续时期：2018-11 至 2019-02 初']
    main=[]
    for method in ['title','price','old_integrated','old_body','old_semantic','body','semantic','joint_body','joint_semantic','static','gate','static_cal','gate_cal','selected_system']:
        row={'方法':NAMES[method]}
        for s in ['AAPL','AMZN']:
            r=metrics[(metrics.symbol==s)&(metrics.phase=='frozen')&(metrics.period=='test')&(metrics.method==method)].iloc[0]
            row[s+' BA']=f'{r.BA:.2%}';row[s+' Brier']=f'{r.Brier:.4f}'
        main.append(row)
    report += [table(pd.DataFrame(main)),'## 九至十月开发回放',table(metrics[(metrics.phase=='frozen')&(metrics.period=='validation')&metrics.method.isin(['title','static','gate','static_cal','gate_cal'])][['symbol','method','BA','Brier']].round(4)), '## 在线更新：独立信息协议',table(metrics[(metrics.phase=='frozen')&(metrics.period=='test')&metrics.method.isin(['title','price','static','gate','online_weights','online_price','online_title'])][['symbol','method','BA','Brier']].round(4))]
    status=json.loads((run/'status.json').read_text());choice=json.loads((run/'system_selection.json').read_text())
    report += ['## 训练期选择与停止',f"预设规则选择 {choice['selected_method']}。完整候选的跨月BA、Brier限制和继续条件在 {run.relative_to(B)}/system_selection.json；未因后续成绩放宽规则。",'## 后续逐月',table(metrics[(metrics.phase=='frozen')&metrics.period.str.match(r'201[89]-')&metrics.method.isin(['title','static','gate','online_weights'])][['symbol','period','method','n','BA','Brier']].round(4)),'## 五日日期块区间：相对标题 baseline',table(intervals[(intervals.period=='test')&(intervals.block_days==5)&(intervals.metric=='BA')&intervals.method.isin(['static','gate','old_integrated','online_weights'])][['symbol','method','difference','low','high']].round(4)),'区间为10,000次共同日期配对重采样的描述性区间，不校正反复探索。BA均为比例，差值0.01等于1个百分点。单日/十日敏感性见CSV。','## 回退与案例',table(gatecounts),'完整改对/改错、新闻分层与最多64例确定性面板见同目录CSV和CASE_CARDS.md。生成面板不等于全部人工审阅。','## 工程状态',f"核心模型使用同样1607分类窗口，{status['actual_fits']}次主运行拟合。FinBERT冻结，未微调；baseline最终概率与旧运行重现一致。在线另有408次小截距拟合，另408次仅用于未来标签扰动验证，不作为新的候选实验。",'初版v1因新代码概率共享内存错误中断；v2完成预测但部分NumPy元数据写为字符串；v3修正类型且预测与v2一致。zero_bias_v1是随后预登记的唯一机制扩展：新闻截距固定零。保留全部记录。','关系与新文章接入受独立复核条件限制。coverage目录是审计和复核材料，任何规则候选数量不能宣传为已增加的有效预测覆盖。','## 文件',f"metrics.csv / paired_intervals.csv / final_components.json / gate_usage.csv / case_changes.csv / news_slices.csv / CASE_CARDS.md；运行权重与OOF在 {run.relative_to(B)}；在线在 {online.relative_to(B)}。"]
    (out/'REPORT.md').write_text('\n\n'.join(report)+'\n')
    sources={str(x):sha(x) for x in [run/'predictions.pkl',online/'predictions.pkl',run/'selection.json',B/'report.py',B/'core.py']};dump(out/'sources.json',sources)
    print(pd.DataFrame(main).to_string(index=False),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--online',type=Path,required=True);a.add_argument('--output',type=Path,required=True);p=a.parse_args();main(p.run,p.online,p.output)
