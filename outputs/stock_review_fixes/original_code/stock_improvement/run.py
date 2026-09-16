import sys,json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.linear_model import LogisticRegression
import joblib
B=Path(__file__).resolve().parent;ROOT=B.parents[1]/'work/stock-data';OUT=B/'results';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(B.parent/'stock_baseline'));from run_baseline import scores
P=json.loads((B/'protocol.json').read_text());old=json.loads((B.parent/'stock_baseline/results/results.json').read_text())
news=pd.read_pickle(ROOT/'audit/news_index.pkl');news['record_key']=news.archive+'::'+news.member;news=news.set_index('record_key')
probs=pd.read_csv(B.parent/'stock_finbert/results/article_probabilities.csv').set_index('record_key')
embpath=ROOT/'finbert_embeddings.npz'
if not embpath.exists():
 torch.manual_seed(573);torch.set_num_threads(4)
 kw=dict(revision='4556d13015211d73dccd3fdd39d39232506f3e43',cache_dir=str(ROOT/'finbert-cache'),trust_remote_code=False,local_files_only=True)
 tok=AutoTokenizer.from_pretrained('ProsusAI/finbert',**kw);m=AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert',**kw)
 device='mps' if torch.backends.mps.is_available() else 'cpu';m.to(device).eval();arr=[]
 for i in range(0,len(probs),32):
  batch=tok(probs.title.iloc[i:i+32].tolist(),padding=True,truncation=True,max_length=256,return_special_tokens_mask=True,return_tensors='pt').to(device)
  special=batch.pop('special_tokens_mask');mask=(batch['attention_mask']*(1-special)).unsqueeze(-1)
  with torch.inference_mode():h=m.bert(**batch).last_hidden_state;v=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
  arr.append(v.cpu().numpy())
  if i%1280==0:print('Embedding',i,len(probs),flush=True)
 np.savez_compressed(embpath,keys=probs.index.to_numpy(dtype=str),embeddings=np.concatenate(arr),revision=kw['revision'])
e=np.load(embpath);assert str(e['revision'])=='4556d13015211d73dccd3fdd39d39232506f3e43';embedding=pd.DataFrame(e['embeddings'],index=e['keys']);assert set(embedding.index)==set(probs.index)
cal=pd.read_csv(ROOT/'audit/xnys_schedule.csv',index_col=0);cal.index=pd.to_datetime(cal.index).strftime('%Y-%m-%d')
for c in ['open','close']:cal[c]=pd.to_datetime(cal[c],utc=True)
result={'protocol':P,'stocks':{}}
for symbol,prefix in [('AAPL','APPLE'),('AMZN','AMAZON')]:
 d=pd.read_csv(B.parent/f'stock_finbert/results/{symbol}/development_features.csv').fillna({'news_record_keys':''})
 original=pd.read_csv(B.parent/f'stock_baseline/results/{symbol}/samples.csv').fillna({'text':''});original=original[original.split.ne('test')].reset_index(drop=True)
 assert d.start_utc.tolist()==original.start_utc.tolist();d['text']=original.text
 m=pd.read_csv(ROOT/f'raw/CHARTS/{prefix}5.csv',header=None,names=['date','time','open','high','low','close','activity']);m['ts']=pd.to_datetime(m.date+' '+m.time,format='%Y.%m.%d %H:%M',utc=True);m['end']=m.ts+pd.Timedelta(minutes=5)
 day=m.ts.dt.strftime('%Y-%m-%d');opens=day.map(cal.open);closes=day.map(cal.close);m=m[(m.ts>=opens)&(m.end<=closes)].sort_values('ts')
 rows=[];embs=[]
 for r in d.itertuples():
  cutoff=pd.Timestamp(r.cutoff_utc);ids=[k for k in r.news_record_keys.split('|') if k];n=news.loc[ids];pr=probs.loc[ids];f={}
  for mode in ['fresh','weighted']:
   keep=n.lag_hours.le(4) if mode=='fresh' else pd.Series(True,index=n.index)
   q=pr.loc[keep];nn=n.loc[keep]
   if len(q):
    age=(cutoff-nn.available_utc).dt.total_seconds().to_numpy()/3600;w=np.exp2(-age) if mode=='weighted' else np.ones(len(q));w=w/w.sum()
    for c in ['positive','negative','neutral']:f[f'{mode}_{c}']=float(w@q[c].to_numpy())
    sig=q.positive-q.negative;mu=w@sig.to_numpy();f[mode+'_std']=float(np.sqrt(w@((sig-mu)**2).to_numpy()))
   else:
    for c in ['positive','negative','neutral','std']:f[f'{mode}_{c}']=0.
   f[mode+'_count']=float(np.log1p(len(q)));f[mode+'_has']=int(bool(len(q)))
  embs.append(embedding.loc[ids].mean(axis=0).to_numpy() if ids else np.zeros(768))
  hist=m[m.end<=cutoff].tail(11);assert len(hist)>0;f['recent_end']=str(hist.end.iloc[-1]);f['recent_age']=(cutoff-hist.end.iloc[-1]).total_seconds()/3600
  for nbar in [1,3,11]:
   h=hist.tail(nbar);valid=len(h)==nbar and (len(h)==1 or h.ts.diff().iloc[1:].eq(pd.Timedelta(minutes=5)).all())
   minutes=nbar*5;f[f'continuous_{minutes}']=int(valid);f[f'ret_{minutes}']=float(np.log(h.close.iloc[-1]/h.open.iloc[0])) if valid else 0.
   if nbar==11:f['range_55']=float((h.high.max()-h.low.min())/h.open.iloc[0]) if valid else 0.
  assert hist.end.max()<=cutoff;rows.append(f)
 extra=pd.DataFrame(rows);d=pd.concat([d,extra],axis=1);enames=[f'emb_{i}' for i in range(768)];d=pd.concat([d,pd.DataFrame(embs,columns=enames)],axis=1)
 price=old['stocks'][symbol]['features'];count=['log_news_count','has_news'];sent=['finbert_positive','finbert_negative','finbert_neutral','sentiment_std'];recent=[c for c in extra if c.startswith(('ret_','continuous_','range_'))]+['recent_age']
 def model(kind,C):
  transforms=[];num=price
  if kind=='recent_price':num=price+recent
  if kind=='price_count':num=price+count
  if kind=='price_sentiment':num=price+count+sent
  if kind=='fresh_sentiment':num=price+[c for c in extra if c.startswith('fresh_')]
  if kind=='timeweighted_sentiment':num=price+[c for c in extra if c.startswith('weighted_')]
  if kind=='price_embedding':num=price+count
  if kind!='paper_bow':transforms.append(('numeric',StandardScaler(),num))
  if kind=='price_embedding':transforms.append(('embedding',make_pipeline(PCA(n_components=16,svd_solver='randomized',random_state=573),StandardScaler()),enames))
  if kind in ['paper_bow','paper_price_bow']:
   transforms.append(('words',make_pipeline(CountVectorizer(binary=True,ngram_range=(1,1),min_df=3,stop_words='english',max_features=10000),SelectKBest(chi2,k=500)),'text'))
  return Pipeline([('features',ColumnTransformer(transforms)),('model',LogisticRegression(C=C,penalty='l1' if kind.startswith('paper') else 'l2',solver='liblinear',max_iter=3000,random_state=573))])
 tr=d[d.split.eq('train')];val=d[d.split.eq('validation')];dates=pd.to_datetime(tr.start_utc,utc=True)
 grid=[];chosen={};pred=val[['start_utc','label']].copy()
 for kind in P['variants']:
  candidates=[]
  for C in [.01,.1,1.]:
   foldscores=[]
   for month in [6,7,8]:
    start=pd.Timestamp(f'2018-{month:02d}-01',tz='UTC');end=start+pd.offsets.MonthBegin()
    train=tr[dates<start];valid=tr[(dates>=start)&(dates<end)]
    fit=model(kind,C).fit(train,train.label);met=scores(valid.label,fit.predict_proba(valid)[:,1]);foldscores.append(met)
    grid.append({'model':kind,'C':C,'fold_month':month,**met})
   ba=np.mean([z['balanced_accuracy'] for z in foldscores]);br=np.mean([z['brier'] for z in foldscores]);candidates.append((ba,-br,C))
  ba,_,C=max(candidates);fit=model(kind,C).fit(tr,tr.label);prob=fit.predict_proba(val)[:,1];met=scores(val.label,prob);pred[kind]=prob
  chosen[kind]={'C':C,'training_cv_ba':ba,'validation':met,'monthly':{month:scores(val.loc[mask,'label'],prob[mask.to_numpy()]) for month in ['2018-09','2018-10'] if (mask:=val.start_utc.str.startswith(month)).any()}}
  folder=OUT/symbol;folder.mkdir(exist_ok=True);joblib.dump(fit,folder/(kind+'.joblib'));print(symbol,kind,C,round(ba,4),round(met['balanced_accuracy'],4),flush=True)
 pd.DataFrame(grid).to_csv(folder/'training_cv_grid.csv',index=False);pred.to_csv(folder/'validation_predictions.csv',index=False)
 d.drop(columns=enames+['text']).to_csv(folder/'feature_manifest.csv',index=False)
 np.save(folder/'development_embeddings.npy',np.asarray(embs))
 result['stocks'][symbol]={'n_train':len(tr),'n_validation':len(val),'models':chosen,'data_changes':{'original_news_rows':int(tr.news_count.sum()),'fresh_news_rows':int(np.rint(np.expm1(tr.fresh_count)).sum()),'validation_original_coverage':float(val.has_news.mean()),'validation_fresh_coverage':float(val.fresh_has.mean()),'median_old_history_age_hours':float(val.history_age_hours.median()),'median_recent_age_hours':float(val.recent_age.median())}}
 (OUT/'results.json').write_text(json.dumps(result,indent=2))
print('E04 complete; final test not evaluated.',flush=True)
