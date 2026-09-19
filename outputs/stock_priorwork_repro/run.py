"""Frozen, leakage-safe prior-work classical reproduction."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[2]; W=ROOT/'work/stock-data'; BASE=ROOT/'outputs/stock_priorwork_repro'; OUT=BASE/'v6'
sys.path.insert(0,str(ROOT/'outputs/stock_baseline'))
from run_baseline import build
SEED=573; CS=[.01,.1,1.]; R1=[f'{v}_{k}' for k in range(1,7) for v in ['return','range']]+['history_age_hours','return_mean','return_std','ny_hour']
METHODS=['PAPER_1G_L1LR','PAPER_1G_LINSVM','PAPER_2G_L1LR','TFIDF_LR','TFIDF_LINSVM','TFIDF_RF','TFIDF_ADABOOST','TFIDF_KNN']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,default=str))
def cleantext(x): return x.fillna('').astype(str)
def mscore(y, p, decision=False):
    y=np.asarray(y); pred=(np.asarray(p)>=0 if decision else np.asarray(p)>=.5).astype(int)
    both=len(np.unique(y))==2
    d={'n':int(len(y)),'accuracy':float(accuracy_score(y,pred)),'ba':float(balanced_accuracy_score(y,pred)) if both else None,'mcc':float(matthews_corrcoef(y,pred)) if both else None,'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'up_recall':float(recall_score(y,pred,pos_label=1,zero_division=0)),'down_recall':float(recall_score(y,pred,pos_label=0,zero_division=0)),'pred_up':float(pred.mean()),'true_up':float(y.mean()),'constant':bool(pred.min()==pred.max())}
    d['auc']=float(roc_auc_score(y,p)) if both else None; d['brier']=None if decision else float(brier_score_loss(y,p)); return d
def spec(name):
    if name.startswith('PAPER_1G'): return ('count',(1,1),'lr' if name.endswith('L1LR') else 'svm')
    if name=='PAPER_2G_L1LR': return ('count',(2,2),'lr')
    if name.startswith('TFIDF_'):
        tail=name.split('_',1)[1].lower()
        return ('tfidf',(1,1),{'linsvm':'svm','lr':'lr','rf':'rf','adaboost':'adaboost','knn':'knn'}[tail])
    raise ValueError(name)
def grid(clf):
    if clf in ('lr','svm'): return [{'C':c} for c in CS]
    if clf=='rf': return [{'max_depth':d,'min_samples_leaf':l} for d in (8,None) for l in (1,5)]
    if clf=='adaboost': return [{'n_estimators':n,'learning_rate':r} for n in (50,100) for r in (.05,.1)]
    if clf=='knn': return [{'n_neighbors':n,'weights':w} for n in (5,15,31) for w in ('uniform','distance')]
def key(p): return json.dumps(p,sort_keys=True)
def fit_pred(name, prm, tr, ev, price=False):
    typ,ng,clf=spec(name); txt=cleantext(tr.stem_body if 'stem_body' in tr else tr.text); etxt=cleantext(ev.stem_body if 'stem_body' in ev else ev.text)
    vec=(CountVectorizer(binary=True,ngram_range=ng,min_df=3) if typ=='count' else TfidfVectorizer(ngram_range=ng,min_df=3,max_features=10000,sublinear_tf=True))
    X=vec.fit_transform(txt); Xe=vec.transform(etxt); k=min(500,X.shape[1])
    if k: sel=SelectKBest(chi2,k=k).fit(X,tr.label); X=sel.transform(X); Xe=sel.transform(Xe)
    if price:
        cols=[c for c in R1 if c in tr.columns]
        sc=StandardScaler().fit(tr[cols].fillna(0)); X=sparse.hstack([X,sparse.csr_matrix(sc.transform(tr[cols].fillna(0)))]); Xe=sparse.hstack([Xe,sparse.csr_matrix(sc.transform(ev[cols].fillna(0)))])
    if clf=='lr': model=LogisticRegression(penalty='l1' if name.startswith('PAPER') else 'l2',solver='liblinear',max_iter=3000,random_state=SEED,**prm)
    elif clf=='svm': model=LinearSVC(random_state=SEED,**prm)
    elif clf=='rf': model=RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=1,**prm)
    elif clf=='adaboost': model=AdaBoostClassifier(random_state=SEED,**prm)
    elif clf=='knn': model=KNeighborsClassifier(metric='cosine',**prm)
    model.fit(X,tr.label)
    decision=clf=='svm'; p=model.decision_function(Xe) if decision else model.predict_proba(Xe)[:,1]
    # Explicit no-news fallback makes the registered policy independent of sparse transform details.
    prior=float(tr.label.mean()); p=np.asarray(p,dtype=float); p[ev.has_news.to_numpy()==0]=(prior-.5) if decision else prior
    return p,decision,{'raw_terms':int(len(vec.vocabulary_)),'selected_terms':int(k),'fit_rows':int(len(tr))}
def choose(history, name):
    g=pd.DataFrame(history)
    if g.empty: return {'C':.1} if spec(name)[2] in ('lr','svm') else grid(spec(name)[2])[0]
    g=g.groupby('params',as_index=False).agg(ba=('ba','mean'),brier=('brier','mean')).sort_values(['ba','brier','params'],ascending=[False,True,True]); return json.loads(g.iloc[0].params)
def phase(ts):
    if ts<pd.Timestamp('2018-09-01',tz='UTC'): return 'oof'
    if ts<pd.Timestamp('2018-11-01',tz='UTC'): return 'development'
    return 'later'
def daily():
    cal=pd.read_csv(W/'audit/xnys_schedule.csv',index_col=0); cal.index=pd.to_datetime(cal.index).strftime('%Y-%m-%d')
    for c in ['open','close']: cal[c]=pd.to_datetime(cal[c],utc=True)
    allrows=[]
    for sym,prefix in [('AAPL','APPLE'),('AMZN','AMAZON')]:
        _,_,_,news=build(W,sym,prefix); news=news.sort_values('available_utc')
        b=pd.read_csv(W/f'raw/CHARTS/{prefix}1440.csv',header=None,names=['date','time','open','high','low','close','activity'])
        b['day']=pd.to_datetime(b.date,format='%Y.%m.%d').dt.strftime('%Y-%m-%d'); b=b.set_index('day').sort_index()
        valid=[d for d in cal.index if d in b.index]
        for i,d in enumerate(valid):
            if i<5: continue
            target=b.loc[d]; op,cl=cal.loc[d,['open','close']]; cutoff=op-pd.Timedelta(minutes=5); prev_close=cal.loc[valid[i-1],'close']
            hist=b.loc[valid[i-5:i]]
            base=dict(symbol=sym,day=d,start_utc=op,end_utc=cl,cutoff_utc=cutoff,label=int(target.close>target.open),target_return=float(target.close/target.open-1),split=phase(op),
                      return_1=float(hist.close.iloc[-1]/hist.open.iloc[-1]-1),return_2=float(hist.close.iloc[-2:]/hist.open.iloc[-2] .prod()-1) if False else float(hist.close.iloc[-1]/hist.close.iloc[-2]-1),
                      return_5=float(hist.close.iloc[-1]/hist.close.iloc[0]-1),range_1=float((hist.high.iloc[-1]-hist.low.iloc[-1])/hist.open.iloc[-1]),return_mean=float((hist.close.pct_change().dropna()).mean()),return_std=float((hist.close.pct_change().dropna()).std()),history_age_hours=float((cutoff-prev_close).total_seconds()/3600),ny_hour=9.5)
            for window,left in [('DNEWS_OVERNIGHT',prev_close),('DNEWS_24H',cutoff-pd.Timedelta(hours=24))]:
                a=news[(news.available_utc>left)&(news.available_utc<=cutoff)]
                r=dict(base,news_window=window,text=' . '.join(a.title.fillna('')),stem_body=' . '.join(a.title_norm.fillna('')),news_count=len(a),has_news=int(len(a)>0),article_keys='|'.join((a.archive+'::'+a.member).astype(str))); allrows.append(r)
    return pd.DataFrame(allrows)
def canonical():
    d=pd.read_pickle(W/'nextgen_4h/price_v1/features.pkl').copy(); d['split']=d.start_utc.map(phase); d['news_window']='4H'; return d
def evaluate(d,horizon):
    out=[]; evidence=[]; select={}; allpred=[]
    for name in METHODS:
      for (sym,news_window),s in d.groupby(['symbol','news_window']):
        hcode=horizon if horizon=='4h' else horizon+':'+news_window
        s=s.sort_values('cutoff_utc').reset_index(drop=True); hist=[]; frozen=None
        for month in pd.period_range('2018-03','2019-02',freq='M').astype(str):
          ev=s[s.start_utc.dt.strftime('%Y-%m').eq(month)];
          if ev.empty: continue
          tr=s[s.end_utc<ev.cutoff_utc.min()];
          if tr.empty or tr.label.nunique()<2: continue
          if month<'2018-09': prm=choose(hist,name)
          else:
            if frozen is None: frozen=choose(hist,name)
            prm=frozen
          p,dec,meta=fit_pred(name,prm,tr,ev,False); rec=dict(stock=sym,horizon=hcode,method=name,params=key(prm),month=month,phase=phase(ev.start_utc.iloc[0]),decision_only=dec,**mscore(ev.label,p,dec)); out.append(rec); hist.append(rec)
          z=ev[['symbol','day','start_utc','cutoff_utc','label','has_news','news_count']].copy();z['horizon']=hcode;z['method']=name;z['month']=month;z['phase']=rec['phase'];z['p']=p;z['decision_only']=dec;allpred.append(z); evidence.append(dict(**rec,train_n=len(tr),train_end=str(tr.end_utc.max()),**meta))
        select[f'{hcode}:{sym}:{name}']=frozen or choose(hist,name)
    return pd.DataFrame(out),pd.concat(allpred,ignore_index=True),select,evidence
def global_selected(metrics,horizon):
    x=metrics[(metrics.horizon.eq(horizon))&metrics.phase.eq('oof')].copy(); q=x.groupby(['method','stock']).ba.mean().unstack(); avg=x.groupby('method')[['ba','brier']].mean(); rows=[]
    for n in q.index: rows.append(dict(method=n,weaker=float(q.loc[n].min()),macro=float(avg.loc[n,'ba']),brier=float(avg.loc[n,'brier']) if pd.notna(avg.loc[n,'brier']) else 9))
    return [r['method'] for r in sorted(rows,key=lambda r:(-r['weaker'],-r['macro'],r['brier'],r['method']))[:3]]
def random_diag(d):
    from sklearn.model_selection import train_test_split
    rows=[]
    for sym,s in d.groupby('symbol'):
      tr,ev=train_test_split(s,stratify=s.label,test_size=.2,random_state=SEED)
      for n in ['TFIDF_LR','TFIDF_RF']:
       prm=choose([],n); p,dec,_=fit_pred(n,prm,tr,ev); rows.append(dict(label='RANDOM_SPLIT_DIAGNOSTIC_ONLY_NOT_DEPLOYMENT_ESTIMATE',stock=sym,method=n,**mscore(ev.label,p,dec)))
    return pd.DataFrame(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--approve-run',action='store_true');a=ap.parse_args()
    if not a.approve_run: raise SystemExit('requires --approve-run')
    if OUT.exists(): raise SystemExit('immutable output already exists')
    prereg=BASE/'PREREGISTRATION.md'; prehash=sha(prereg); OUT.mkdir(parents=True)
    manifest={'preregistration_sha256':prehash,'starting_commit':'63a8e811f132e9cbada6460bb82ae7af63adadfe','sources':{str(W/'nextgen_4h/price_v1/features.pkl'):sha(W/'nextgen_4h/price_v1/features.pkl'),str(W/'audit/news_index.pkl'):sha(W/'audit/news_index.pkl')}};dump(OUT/'manifest.json',manifest); (OUT/'PREREGISTRATION_SNAPSHOT.md').write_text(prereg.read_text())
    four=canonical(); one=daily(); assert len(four)==1607 and len(four[four.start_utc>=pd.Timestamp('2018-03-01',tz='UTC')])==1374
    four.to_pickle(OUT/'private_four.pkl'); one.to_pickle(OUT/'private_one.pkl')
    m4,p4,s4,e4=evaluate(four,'4h'); m1,p1,s1,e1=evaluate(one,'1d')
    # Explicit dependency outcomes: the frozen environment has neither package.
    unavailable=[]
    for h in ['4h','1d']:
      for n in ['TFIDF_XGBOOST','W2V_LR','W2V_LINSVM','W2V_RF']:
       reason='NOT_RUN_DEPENDENCY_UNAVAILABLE'; unavailable += [dict(stock=s,horizon=h,method=n,phase='all',status=reason) for s in ['AAPL','AMZN']]
    # joint candidates run using exact same representation plus R1/DPRICE fields.
    joint=[]
    for d,h,metrics in [(four,'4h',m4),(one[one.news_window.eq('DNEWS_OVERNIGHT')].copy(),'1d:DNEWS_OVERNIGHT',m1)]:
      chosen=global_selected(metrics,h)
      for n in chosen:
       for sym,s in d.groupby('symbol'):
        s=s.sort_values('cutoff_utc'); tr=s[s.start_utc<pd.Timestamp('2018-09-01',tz='UTC')]; ev=s[s.start_utc>=pd.Timestamp('2018-09-01',tz='UTC')]
        prior=metrics[(metrics.stock.eq(sym)) & (metrics.method.eq(n)) & (metrics.phase.eq('oof'))]
        prm=choose(prior.to_dict('records'),n)
        p,dec,_=fit_pred(n,prm,tr,ev,True); z=ev[['symbol','day','start_utc','cutoff_utc','label','has_news','news_count']].copy();z['horizon']=h;z['method']='JOINT_'+n;z['phase']=ev.split;z['month']=ev.start_utc.dt.strftime('%Y-%m');z['p']=p;z['decision_only']=dec; p4=pd.concat([p4,z]) if h=='4h' else p4; joint.append(dict(stock=sym,horizon=h,method='JOINT_'+n,phase='development_later',**mscore(ev.label,p,dec)))
    m4.to_csv(OUT/'METRICS_4H.csv',index=False); m1.to_csv(OUT/'METRICS_1D.csv',index=False); m4[m4.phase.eq('oof')].to_csv(OUT/'MONTHLY_4H.csv',index=False);m1[m1.phase.eq('oof')].to_csv(OUT/'MONTHLY_1D.csv',index=False)
    p4.to_csv(OUT/'predictions_4h.csv',index=False); p1.to_csv(OUT/'predictions_1d.csv',index=False); pd.DataFrame(joint).to_csv(OUT/'JOINT_METRICS.csv',index=False); random_diag(four).to_csv(OUT/'RANDOM_SPLIT_DIAGNOSTIC.csv',index=False); dump(OUT/'SELECTIONS.json',dict(four=s4,one=s1,global_four=global_selected(m4,'4h'),global_one_overnight=global_selected(m1,'1d:DNEWS_OVERNIGHT'),global_one_24h=global_selected(m1,'1d:DNEWS_24H'),unavailable=unavailable)); dump(OUT/'training_evidence.json',e4+e1)
    cov=[]
    for h,x in [('4h',four),('1d',one)]:
      for (s,sp),g in x.groupby(['symbol','split']):cov.append(dict(horizon=h,stock=s,phase=sp,n=len(g),news_coverage=float(g.has_news.mean()),no_news=int((g.has_news==0).sum()),mean_articles=float(g.news_count[g.has_news.eq(1)].mean()) if g.has_news.any() else 0,dates=int(g.day.nunique())))
    pd.DataFrame(cov).to_csv(OUT/'COVERAGE.csv',index=False); print('COMPLETE',flush=True)
if __name__=='__main__': main()
