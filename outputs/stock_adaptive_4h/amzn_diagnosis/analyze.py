from pathlib import Path
import hashlib,json,re,zipfile
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,brier_score_loss
B=Path(__file__).resolve().parent;A=B.parent;ROOT=A.parents[1]
OUT=B/'analysis_v1';OUT.mkdir(exist_ok=False)
def dump(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
d=pd.read_pickle(A/'runs/zero_bias_v1/predictions.pkl');d=d[(d.symbol=='AMZN')&(d.phase=='frozen')].copy()
f=pd.read_pickle(A/'runs/v3/predictions.pkl').set_index('key');d['primary_gate']=d.key.map(f.gate)
assert len(d)==305 and d.key.is_unique
methods=['constant','prior','title','price','old_body','old_semantic','old_integrated','body','semantic','static','gate','gate_cal','primary_gate']
assert all(k in d for k in methods)
def metrics(g,m):
    y=g.label.to_numpy();p=g[m].to_numpy();a=p>=.5
    return dict(n=len(g),dates=g.day.nunique(),BA=balanced_accuracy_score(y,a) if len(set(y))==2 else None,MCC=matthews_corrcoef(y,a),Brier=brier_score_loss(y,p),up_recall=a[y==1].mean() if (y==1).any() else None,down_recall=(~a[y==0]).mean() if (y==0).any() else None,pred_up=a.mean(),actual_up=y.mean(),mean_probability=p.mean())
rows=[]
for period,g in d.groupby('split'):
    for slice_name,h in [('all',g),('news',g[g.has_news==1]),('no_news',g[g.has_news==0])]+[(m,h) for m,h in g.groupby('month')]:
        for model in methods:rows.append(dict(period=period,slice=slice_name,model=model,**metrics(h,model)))
pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
comparisons=[('body_vs_title','old_body','title'),('body_vs_old_mix','old_body','old_integrated'),('title_vs_new_gate','title','gate')]
comparisons_full=comparisons+[('old_mix_vs_title','old_integrated','title')]
changes=[];panel=[];used=set();days=set();strata=[]
for name,first,second in comparisons_full:
    for period,g in d.groupby('split'):
        c1=g[first].ge(.5).eq(g.label);c2=g[second].ge(.5).eq(g.label)
        outcome=np.select([c1&c2,~c1&~c2,c1&~c2],['both_correct','both_wrong','first_wins'],default='second_wins')
        g=g.assign(case_outcome=outcome)
        for news,h in g.groupby('has_news'):
            for label,z in h.groupby('label'):
                repairs=int((z.case_outcome=='first_wins').sum());hurts=int((z.case_outcome=='second_wins').sum());den=int((g.label==label).sum())
                changes.append(dict(comparison=name,period=period,has_news=news,label=label,n=len(z),first_wins=repairs,second_wins=hurts,BA_contribution=.5*(repairs-hurts)/den))
        if name not in {x[0] for x in comparisons}:continue
        for outcome in ['both_correct','both_wrong','first_wins','second_wins']:
            h=g[g.case_outcome==outcome].copy();h['rank']=h.key.map(lambda key:hashlib.sha256(('573|'+key+'|'+name).encode()).hexdigest())
            h=h.sort_values('rank');available=h[~h.key.isin(used)];unique_day=available[~available.day.isin(days)];selected=unique_day if len(unique_day) else available
            strata.append(dict(comparison=name,period=period,outcome=outcome,available=len(h),selected=bool(len(selected))))
            if not len(selected):continue
            r=selected.iloc[0].to_dict();r.update(comparison=name,first=first,second=second,case_id=f'C{len(panel)+1:02d}',repeated_day=r['day'] in days);used.add(r['key']);days.add(r['day']);panel.append(r)
pd.DataFrame(changes).to_csv(OUT/'paired_decomposition.csv',index=False)
# Exact reconciliation of the additive BA decomposition.
for name,first,second in comparisons_full:
    for period,g in d.groupby('split'):
        total=sum(x['BA_contribution'] for x in changes if x['comparison']==name and x['period']==period)
        assert abs(total-(metrics(g,first)['BA']-metrics(g,second)['BA']))<1e-12
pd.DataFrame(panel).to_csv(OUT/'case_panel_with_outcomes.csv',index=False);dump('selection_strata.json',strata)
# News-only reading copy, intentionally no price outcome or model probabilities.
raw=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key');blind=[]
for r in panel:
    ids=[k for k in r['news_record_keys'].split('|') if k];ids=sorted(ids,key=lambda k:(str(raw.loc[k,'available_utc']),k),reverse=True);seen=set();articles=[]
    for key in ids:
        m=raw.loc[key];norm=re.sub(r'\W+',' ',str(m.title).lower().split(' - ')[0]).strip()
        if norm in seen:continue
        seen.add(norm)
        with zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/m.archive) as z:text=json.loads(z.read(m.member)).get('text','')
        articles.append(dict(record_key=key,title=m.title,published=str(m.published_utc),available=str(m.available_utc),chars=len(text),excerpt=text[:5500],truncated=len(text)>5500))
        if len(articles)==2:break
    blind.append(dict(case_id=r['case_id'],key=r['key'],day=r['day'],start=str(r['start_utc']),news_count=r['news_count'],all_titles=[str(raw.loc[k,'title']) for k in ids],articles=articles))
dump('content_before_outcome.json',blind)
for i in range(0,len(blind),6):dump(f'content_batch_{i//6+1}.json',blind[i:i+6])
cv=[]
for run in ['v3','zero_bias_v1']:
    z=pd.read_csv(A/'runs'/run/'residual_cv.csv');z=z[z.symbol.eq('AMZN')];z=z.groupby(['kind','C'])[['BA','Brier','base_Brier']].mean().reset_index();z['version']=run;z['Brier_difference']=z.Brier-z.base_Brier;z['eligible']=z.Brier_difference<=.002;cv.append(z)
pd.concat(cv).to_csv(OUT/'training_selection_audit.csv',index=False)
# Exact news membership and mature labels preserved. Record immutable source fingerprints.
paths=[B/'PROTOCOL.md',Path(__file__),A/'runs/zero_bias_v1/predictions.pkl',A/'runs/v3/predictions.pkl',A/'runs/zero_bias_v1/residual_cv.csv',A/'runs/v3/residual_cv.csv',ROOT/'work/stock-data/audit/news_index.pkl']
dump('sources.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
dump('STATUS.json',dict(status='POPULATION_AND_PANEL_READY_CONTENT_REVIEW_PENDING',windows=len(d),cases=len(panel),dates=len(days),article_excerpts=sum(len(x['articles']) for x in blind),new_training=False))
print(pd.DataFrame(rows).query("period=='test' and slice=='all'")[['model','n','BA','Brier','up_recall','down_recall','pred_up']].to_string(index=False))
print('Panel:',len(panel),'cases',len(days),'dates',sum(len(x['articles']) for x in blind),'article excerpts')
