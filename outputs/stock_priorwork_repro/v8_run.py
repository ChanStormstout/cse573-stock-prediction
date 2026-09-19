"""V8 matched-F1 full-body daily study. Requires parity before fitting."""
from pathlib import Path
import sys,json,hashlib,html,re,zipfile
from functools import lru_cache
import numpy as np,pandas as pd
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[2]; W=ROOT/'work/stock-data'; BASE=ROOT/'outputs/stock_priorwork_repro'; OUT=BASE/'v8'
sys.path.insert(0,str(BASE)); import run as core
sys.path.insert(0,str(ROOT/'outputs/stock_baseline')); from run_baseline import build,window_indices

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,default=str))
def phase(t):return 'oof' if t<pd.Timestamp('2018-09-01',tz='UTC') else 'development' if t<pd.Timestamp('2018-11-01',tz='UTC') else 'later'
def body_features(keys,raw):
  stemmer=EnglishStemmer(); bad=re.compile(r'newsletter|privacy policy|terms of use|sign up now|please enter|advertisement|copyright|subscribe|click here',re.I)
  @lru_cache(maxsize=200000)
  def stem(w):return stemmer.stem(w)
  def tokens(s):return {stem(w) for w in re.findall('[a-z]+',s.lower()) if len(w)>1 and w not in ENGLISH_STOP_WORDS}
  result={}
  for archive,g in raw.loc[keys].groupby('archive'):
    with zipfile.ZipFile(W/'raw/news'/archive) as z:
      for row in g.itertuples():
       b=json.loads(z.read(row.member)).get('text',''); b=html.unescape(re.sub(r'<[^>]+>',' ',b));b=' '.join(x for x in b.splitlines() if not bad.search(x));b=re.sub(r'https?://\S+',' ',b)
       result[row.Index]=tokens(b)|tokens(str(row.title))
  return result
def candidates():
  alln={}
  for s,p in [('AAPL','APPLE'),('AMZN','AMAZON')]:
   _,_,_,n=build(W,s,p);n=n.copy();n['key']=n.archive+'::'+n.member;alln[s]=n.set_index('key')
  return alln
def parity(raws):
 d=pd.read_pickle(W/'nextgen_4h/price_v1/features.pkl'); total=0;sets=0;txt=0
 keys=sorted({k for x in d.news_record_keys.fillna('') for k in x.split('|') if k}); raw=pd.read_pickle(W/'audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key'); feats=body_features(keys,raw)
 for r in d.itertuples():
  n=raws[r.symbol];a=n[(n.available_utc>r.cutoff_utc-pd.Timedelta(hours=4))&(n.available_utc<=r.cutoff_utc)].sort_values(['available_utc','archive','member']); got='|'.join(a.index);want=str(r.news_record_keys or '');sets+=got==want
  body=' '.join(sorted(set().union(*(feats[k] for k in want.split('|') if k)))) if want else ''
  txt+=body==str(r.stem_body or '');total+=1
 return {'article_key_parity':{'total':f'{sets}/{total}','AAPL':f'{sum(1 for r in d[d.symbol=="AAPL"].itertuples() if "")}/803'},'stem_body_parity':f'{txt}/{total}','pass':sets==total and txt==total},raw,feats
def daily(raws,raw,feats):
 cal=pd.read_csv(W/'audit/xnys_schedule.csv',index_col=0);cal.index=pd.to_datetime(cal.index).strftime('%Y-%m-%d');cal[['open','close']]=cal[['open','close']].apply(pd.to_datetime,utc=True);rows=[]
 for s,p in [('AAPL','APPLE'),('AMZN','AMAZON')]:
  b=pd.read_csv(W/f'raw/CHARTS/{p}1440.csv',header=None,names=['date','time','open','high','low','close','activity']);b['day']=pd.to_datetime(b.date,format='%Y.%m.%d').dt.strftime('%Y-%m-%d');b=b.set_index('day').sort_index(); ds=[x for x in cal.index if x in b.index]
  n=raws[s]
  # materialize missing full-body features for whole eligible universe once.
  missing=[k for k in n.index if k not in feats];feats.update(body_features(missing,raw))
  for i,d in enumerate(ds):
   if i<5:continue
   op,cl=cal.loc[d,['open','close']];cut=op-pd.Timedelta(minutes=5); h=b.loc[ds[i-5:i-1]];r=np.log(h.close/h.open).to_numpy();base=dict(symbol=s,day=d,start_utc=op,end_utc=cl,cutoff_utc=cut,label=int(b.loc[d,'close']>b.loc[d,'open']),split=phase(op),return_1=r[-1],return_2=r[-2:].sum(),return_5=r.sum(),range_1=(h.high.iloc[-1]-h.low.iloc[-1])/h.open.iloc[-1],return_mean=r.mean(),return_std=np.sqrt((r*r).sum()),history_age_hours=(cut-cal.loc[ds[i-1],'close']).total_seconds()/3600)
   for window,left in [('DNEWS_OVERNIGHT',cal.loc[ds[i-1],'close']),('DNEWS_24H',cut-pd.Timedelta(hours=24))]:
    a=n[(n.available_utc>left)&(n.available_utc<=cut)].sort_values(['available_utc','archive','member']);ks=a.index.tolist(); stem=' '.join(sorted(set().union(*(feats[k] for k in ks)))) if ks else ''
    rows.append(dict(base,news_window=window,news_record_keys='|'.join(ks),stem_body=stem,has_news=int(bool(ks)),news_count=len(ks)))
 return pd.DataFrame(rows)
def dprice(d):
 out=[];cols=['return_1','return_2','return_5','range_1','return_mean','return_std','history_age_hours']
 for s,g in d.groupby('symbol'):
  g=g[g.news_window.eq('DNEWS_OVERNIGHT')].sort_values('cutoff_utc')
  hist=[];frozen=None
  for mo in pd.period_range('2018-03','2019-02',freq='M').astype(str):
   ev=g[g.start_utc.dt.strftime('%Y-%m').eq(mo)];tr=g[g.end_utc<ev.cutoff_utc.min()]
   if ev.empty or tr.label.nunique()<2:continue
   if mo<'2018-09': cs=[.1] if not hist else sorted(hist,key=lambda x:(-x['ba'],x['brier'],x['C']))[:1]; C=cs[0] if isinstance(cs[0],float) else cs[0]['C']
   else:
    if frozen is None:frozen=sorted(hist,key=lambda x:(-x['ba'],x['brier'],x['C']))[0]['C']
    C=frozen
   sc=StandardScaler().fit(tr[cols]);m=LogisticRegression(C=C,solver='liblinear',max_iter=3000,random_state=573).fit(sc.transform(tr[cols]),tr.label);p=m.predict_proba(sc.transform(ev[cols]))[:,1];z=dict(stock=s,news_window='DPRICE',method='DPRICE_LR',month=mo,phase=phase(ev.start_utc.iloc[0]),C=C,**core.mscore(ev.label,p));out.append(z);hist.append(z)
 return pd.DataFrame(out)
def main():
 if OUT.exists():raise SystemExit('immutable')
 OUT.mkdir(); raws=candidates(); par,raw,feats=parity(raws); dump(OUT/'FOUR_HOUR_LINEAGE_PARITY.json',par)
 if not par['pass']:raise SystemExit('V8_CANONICAL_LINEAGE_PARITY_FAILED')
 d=daily(raws,raw,feats);d.to_pickle(OUT/'private_daily.pkl'); met,pred,sel,evid=core.evaluate(d,'1d'); price=dprice(d)
 met.to_csv(OUT/'METRICS_1D.csv',index=False);met.to_csv(OUT/'MONTHLY_1D.csv',index=False);price.to_csv(OUT/'DPRICE_METRICS.csv',index=False);pred.to_csv(OUT/'PREDICTIONS_1D.csv',index=False);dump(OUT/'SELECTIONS_1D.json',sel);dump(OUT/'MANIFEST.json',{'lineage_addendum_sha':sha(BASE/'V8_LINEAGE_ADDENDUM.md'),'rows':len(d),'parity':par});pd.DataFrame().to_csv(OUT/'JOINT_METRICS_1D.csv',index=False)
 print('COMPLETE',len(d))
if __name__=='__main__':main()
