"""Reproducible first baseline. No test predictions or test metrics are computed."""
import argparse, json, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score, brier_score_loss
import joblib

def window_indices(times, cutoff):
    return (np.searchsorted(times, (cutoff-pd.Timedelta(hours=4)).value, side='right'),
            np.searchsorted(times, cutoff.value, side='right'))

def build(root, symbol, prefix):
    news=pd.read_pickle(root/'audit/news_index.pkl')
    ledger={'raw_news':len(news)}
    news=news[news.language.eq('english') & news.chars.ge(100) & news.lag_hours.ge(0)].copy()
    ledger['valid_language_body_time']=len(news)
    pattern=r'\b(?:AAPL|Apple)\b' if symbol=='AAPL' else r'\b(?:AMZN|Amazon)\b'
    news=news[news.title.fillna('').str.contains(pattern,case=False,regex=True)&~news.physician].copy()
    ledger['company_title_candidates']=len(news)
    news=news.sort_values(['available_utc','archive','member']).drop_duplicates('normalized_hash').drop_duplicates('title_norm').reset_index(drop=True)
    ledger['deduplicated_titles']=len(news)
    times=news.available_utc.astype('int64').to_numpy()
    cal=pd.read_csv(root/'audit/xnys_schedule.csv',index_col=0)
    cal.index=pd.to_datetime(cal.index).strftime('%Y-%m-%d')
    for c in ['open','close']:cal[c]=pd.to_datetime(cal[c],utc=True)
    m=pd.read_csv(root/'raw/CHARTS'/f'{prefix}5.csv',header=None,names=['date','time','open','high','low','close','activity'])
    m['ts']=pd.to_datetime(m.date+' '+m.time,format='%Y.%m.%d %H:%M',utc=True)
    m=m.sort_values('ts')
    m['bucket']=(m.ts-pd.Timedelta(minutes=30)).dt.floor('h')+pd.Timedelta(minutes=30)
    bars=[]
    for start,w in m.groupby('bucket',sort=True):
        end=start+pd.Timedelta(hours=1);day=start.strftime('%Y-%m-%d')
        if day not in cal.index or start<cal.loc[day,'open'] or end>cal.loc[day,'close']:continue
        expected=pd.date_range(start,periods=12,freq='5min')
        if len(w)!=12 or not np.array_equal(w.ts.astype('int64'),expected.astype('int64')):continue
        bars.append(dict(start=start,end=end,open=w.open.iloc[0],high=w.high.max(),low=w.low.min(),close=w.close.iloc[-1]))
    bars=pd.DataFrame(bars);ledger['complete_regular_hours']=len(bars)
    rows=[];numeric=[]
    for b in bars.itertuples():
        cutoff=b.start-pd.Timedelta(minutes=5)
        history=bars[bars.end<=cutoff].tail(6)
        if len(history)<6 or b.close==b.open:continue
        left,right=window_indices(times,cutoff);articles=news.iloc[left:right]
        assert history.end.max()<=cutoff
        assert articles.empty or articles.available_utc.max()<=cutoff
        row=dict(symbol=symbol,start_utc=b.start,end_utc=b.end,cutoff_utc=cutoff,
                 history_ends='|'.join(history.end.astype(str)),news_ids='|'.join(articles.uuid.astype(str)),
                 text=' . '.join(articles.title.fillna('')),label=int(b.close>b.open),
                 target_return=b.close/b.open-1,news_count=len(articles),has_news=int(len(articles)>0))
        for k,h in enumerate(history.iloc[::-1].itertuples(),1):
            row[f'return_{k}']=float(np.log(h.close/h.open))
            row[f'range_{k}']=float((h.high-h.low)/h.open)
        row['history_age_hours']=(cutoff-history.end.iloc[-1]).total_seconds()/3600
        row['return_mean']=np.mean([row[f'return_{k}'] for k in range(1,7)])
        row['return_std']=np.std([row[f'return_{k}'] for k in range(1,7)])
        row['ny_hour']=b.start.tz_convert('America/New_York').hour+.5
        row['split']='train' if b.start<pd.Timestamp('2018-09-01',tz='UTC') else 'validation' if b.start<pd.Timestamp('2018-11-01',tz='UTC') else 'test'
        rows.append(row)
    numeric=[f'{v}_{k}' for k in range(1,7) for v in ['return','range']]+['history_age_hours','return_mean','return_std','ny_hour']
    return pd.DataFrame(rows),numeric,ledger,news

def scores(y,p):
    pred=(p>=.5).astype(int)
    both=len(np.unique(y))==2
    return dict(accuracy=accuracy_score(y,pred),balanced_accuracy=balanced_accuracy_score(y,pred) if both else None,
                mcc=matthews_corrcoef(y,pred) if both else None,f1_up=f1_score(y,pred,zero_division=0),brier=brier_score_loss(y,p))

def estimator(kind,numeric,C):
    transforms=[]
    if kind in ['price','combined']:transforms.append(('price',StandardScaler(),numeric))
    if kind in ['text','combined']:
        transforms.append(('text',TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=3,max_df=.98,sublinear_tf=True),'text'))
    return Pipeline([('features',ColumnTransformer(transforms)),('model',LogisticRegression(C=C,max_iter=2000,solver='liblinear',random_state=573))])

def run(root,out):
    out.mkdir(parents=True,exist_ok=True)
    summary={'protocol':{'target':'full regular-session hour close > open; flat hours excluded','cutoff':'5 minutes before target start',
        'news_window':'(cutoff - 4 hours, cutoff] using max(published,crawled); negative lag excluded',
        'split':'train Jan-Aug 2018; validation Sep-Oct 2018; test Nov 2018-Feb 1 2019',
        'C_grid':[.01,.1,1.0],'selection':'validation balanced accuracy, then lower Brier','test_evaluated':False,
        'seed':573,'python':platform.python_version(),'sklearn':sklearn.__version__},'stocks':{}}
    for symbol,prefix in [('AAPL','APPLE'),('AMZN','AMAZON')]:
        data,numeric,ledger,news=build(root,symbol,prefix)
        folder=out/symbol;folder.mkdir(exist_ok=True)
        data.to_csv(folder/'samples.csv',index=False)
        news[['uuid','title','available_utc','archive','member']].head(20).to_csv(folder/'first_20_news_for_review.csv',index=False)
        tr=data[data.split.eq('train')];va=data[data.split.eq('validation')]
        assert tr.end_utc.max()<va.cutoff_utc.min()
        prediction=va[['start_utc','label','news_count']].copy();experiments=[];selected={}
        dummy=DummyClassifier(strategy='prior').fit(np.zeros((len(tr),1)),tr.label)
        p=dummy.predict_proba(np.zeros((len(va),1)))[:,1]
        selected['majority']=scores(va.label,p);prediction['majority']=p
        for kind in ['price','text','combined']:
            candidates=[]
            for C in [.01,.1,1.0]:
                model=estimator(kind,numeric,C).fit(tr,tr.label)
                p=model.predict_proba(va)[:,1];metrics=scores(va.label,p)
                experiments.append(dict(model=kind,C=C,**metrics));candidates.append((metrics,C,model,p))
            metrics,C,model,p=max(candidates,key=lambda x:(x[0]['balanced_accuracy'],-x[0]['brier']))
            selected[kind]=dict(C=C,**metrics);prediction[kind]=p
            joblib.dump(model,folder/f'{kind}_model.joblib')
            if kind in ['text','combined']:
                names=model.named_steps['features'].get_feature_names_out();coef=model.named_steps['model'].coef_[0]
                pd.DataFrame({'feature':names,'coefficient':coef}).sort_values('coefficient').to_csv(folder/f'{kind}_coefficients.csv',index=False)
        prediction.to_csv(folder/'validation_predictions.csv',index=False)
        pd.DataFrame(experiments).to_csv(folder/'validation_grid.csv',index=False)
        slices={}
        for label,mask in [('September',va.start_utc.dt.month.eq(9)),('October',va.start_utc.dt.month.eq(10)),('with_news',va.news_count.gt(0)),('no_news',va.news_count.eq(0))]:
            if mask.sum():slices[label]={'n':int(mask.sum()),**{k:scores(va.loc[mask,'label'],prediction.loc[mask,k].to_numpy()) for k in selected}}
        summary['stocks'][symbol]={'cleaning':ledger,'features':numeric,'split_counts':data.split.value_counts().to_dict(),
             'news_coverage':data.groupby('split').has_news.mean().to_dict(),'train_up_fraction':float(tr.label.mean()),
             'validation_up_fraction':float(va.label.mean()),'selected_validation':selected,'validation_slices':slices,
             'samples_sha256':hashlib.sha256((folder/'samples.csv').read_bytes()).hexdigest()}
        print(symbol,json.dumps(summary['stocks'][symbol]['selected_validation']),flush=True)
    (out/'results.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run(args.data_root,args.out)
