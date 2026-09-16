"""Descriptive full-window comparison and balanced case cards, no retraining."""
import json,re,html
from functools import lru_cache
import numpy as np,pandas as pd,joblib
from nltk.stem.snowball import EnglishStemmer
from common import B,O,prepared
D,_=prepared();R=B/'results';articles=pd.read_csv(O/'stock_finbert/results/article_probabilities.csv').set_index('record_key');bodies=json.loads((O/'stock_content/results/source_bodies.json').read_text());stemmer=EnglishStemmer();panels=[];cards=[];counts={}
@lru_cache(maxsize=150000)
def stem(w):return stemmer.stem(w)
@lru_cache(maxsize=10000)
def cleaned_input(key):
    # Match prepare.py exactly: locate evidence in actual model text, not a
    # removed HTML/navigation occurrence that happens to share the same word.
    body=html.unescape(re.sub(r'<[^>]+>',' ',bodies[key]))
    bad=re.compile(r'newsletter|privacy policy|terms of use|sign up now|please enter|advertisement|copyright|subscribe|click here',re.I)
    body=' '.join(line for line in body.splitlines() if not bad.search(line))
    return articles.loc[key,'title']+' '+re.sub(r'https?://\S+',' ',body)

def snippet(key,word):
    body=cleaned_input(key)
    for m in re.finditer(r'[a-zA-Z]+',body):
        if stem(m.group().lower())==word:return re.sub(r'\s+',' ',body[max(0,m.start()-100):m.end()+170])
    return None


for s,dd in D.groupby('symbol'):
    d=dd[dd.split.eq('validation')].reset_index(drop=True);p=pd.read_csv(R/f'linear/{s}/validation_predictions.csv');assert d.start_utc.tolist()==p.start_utc.tolist();a=p.paper_stem_title.ge(.5).eq(p.label);b=p.paper_stem_body.ge(.5).eq(p.label)
    p['case_group']=np.select([~a&b,a&~b,~a&~b],['正文改对','正文改错','共同错误'],default='共同正确');p['month']=p.start_utc.str[:7];p['target_return']=d.target_return;p['news_count']=d.news_count;panels.append(p);counts[s]=p.case_group.value_counts().to_dict()
    fit=joblib.load(R/f'linear/{s}/paper_stem_body.joblib');names=fit['features'].get_feature_names_out();co=fit['model'].coef_[0]
    for (group,month),g in p.groupby(['case_group','month']):
        for idx in g.sample(min(2,len(g)),random_state=580).index:
            r=p.loc[idx];source=d.iloc[idx];x=fit['features'].transform(d.iloc[[idx]]);v=np.asarray(x.toarray() if hasattr(x,'toarray') else x).ravel()*co
            used=np.flatnonzero(v);top=sorted(used,key=lambda i:abs(v[i]),reverse=True)[:5];keys=source.news_record_keys.split('|') if source.news_record_keys else []
            parts=[f'## {s} / {r.start_utc} / {group}',f'目标收益 {r.target_return:.3%}；标题概率 {r.paper_stem_title:.3f} → 正文概率 {r.paper_stem_body:.3f}；{len(keys)} 条新闻。', '', '主要线性分数贡献（不是价格变化原因）：']
            for i in top:
                name=names[i];parts.append(f'- `{name}`：{v[i]:+.3f}')
                if name.startswith('words__'):
                    word=name.removeprefix('words__')
                    for key in keys:
                        quote=snippet(key,word)
                        if quote:
                            parts.append(f'  - 记录：`{key}`；标题：{articles.loc[key,"title"]}\n  - 清理后输入定位片段：{quote}');break
            cards.append('\n'.join(parts))
allp=pd.concat(panels,ignore_index=True);allp.to_csv(R/'all_504_body_comparisons.csv',index=False);(R/'case_group_counts.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2))
intro='# 全文词特征：32 张分层诊断卡\n\n按股票、月份、改对/改错/共同错误/共同正确各取最多两例，固定随机种子580。均为程序定位与模型贡献分析，不是独立人工语义或因果标注；所有504个窗口另存。输入摘录可能包含源文本噪声，保留以便审阅。\n\n'
(B/'CASES_32.md').write_text(intro+'\n\n'.join(cards)+'\n');print(json.dumps(counts,ensure_ascii=False));print('Generated',len(cards),'cards')
# Five-trading-day moving blocks as sensitivity to longer dependence; descriptive.
days=sorted(allp.start_utc.str[:10].unique());rng=np.random.default_rng(580);delta={s:[] for s in ['AAPL','AMZN']};groups={s:{day:np.flatnonzero(p.start_utc.str[:10].eq(day)) for day in days} for s,p in zip(['AAPL','AMZN'],panels)}
def ba(y,p):return .5*((p[y==1]>=.5).mean()+(p[y==0]<.5).mean())
for _ in range(2000):
    starts=rng.integers(0,len(days)-4,size=int(np.ceil(len(days)/5)));draw=np.concatenate([np.arange(i,i+5) for i in starts])[:len(days)]
    for s,p in zip(['AAPL','AMZN'],panels):
        ix=np.concatenate([groups[s][days[j]] for j in draw]);y=p.label.to_numpy()[ix];delta[s].append(ba(y,p.paper_stem_body.to_numpy()[ix])-ba(y,p.paper_stem_title.to_numpy()[ix]))
ci={s:np.quantile(v,[.025,.975]).tolist() for s,v in delta.items()};ci['equal_stock_mean']=np.quantile(np.mean(list(delta.values()),axis=0),[.025,.975]).tolist();(R/'body_five_day_sensitivity.json').write_text(json.dumps(dict(method='Noncircular moving blocks of five adjacent trading days, paired across stocks; 2000 draws; posthoc descriptive sensitivity only',intervals=ci),indent=2));print('Five-day intervals',ci)
