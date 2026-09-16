from core import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def capacity(d,out):
    rows=[];preds=[];selects={};allcv=[]
    for stock,s in d.groupby('symbol'):
        for k in [25,100,500]:
            for pen in ['l1','l2']:
                tag=f'k{k}_{pen}'; grid=[]
                for C in [.01,.1,1.]:
                    for month,a,b in folds(s):
                        m=model(k,pen,C).fit(a,a.label);p=m.predict_proba(b)[:,1];tr=metrics(a.label,m.predict_proba(a)[:,1]);va=metrics(b.label,p)
                        r=dict(symbol=stock,method=tag,C=C,month=month,nonzero=int(np.count_nonzero(m[-1].coef_)),n_train=len(a),**{'train_'+k:v for k,v in tr.items()},**va);rows.append(r);grid.append(r)
                        z=pred_rows(b,p,tag,C=C,fold=month);allcv.append(z)
                g=pd.DataFrame(grid).groupby('C')[['balanced_accuracy','brier']].mean().sort_values(['balanced_accuracy','brier'],ascending=[False,True]);c=float(g.index[0]);selects[stock+'_'+tag]=c
                a=s[s.split.eq('train')];b=s[~s.split.eq('train')];temporal(a,b);m=model(k,pen,c).fit(a,a.label);p=m.predict_proba(b)[:,1]
                save_model(m,a,b,out/stock/tag,p);preds.append(pred_rows(b,p,tag))
        print('capacity complete',stock,flush=True)
    pd.DataFrame(rows).to_csv(out/'cv.csv',index=False);pd.concat(allcv).to_csv(out/'cv_predictions.csv',index=False);pd.concat(preds).to_csv(out/'predictions.csv',index=False);dump(out/'selected.json',selects)
    curve=[];days=d.assign(day=d.start_utc.dt.strftime('%Y-%m-%d')).groupby('symbol').day.apply(set);common=sorted(set.intersection(*days.tolist()))
    selections={}
    for month,_,_ in folds(d):
        eligible=[x for x in common if x<month+'-01']
        for frac in [.25,.5,1.]:
            for seed in [573,574,575]:
                # Stratify across equal chronological bins; preserve first/last dates and historical span.
                rng=np.random.default_rng(seed);n=max(2,round(len(eligible)*frac));bins=np.array_split(np.arange(1,len(eligible)-1),max(1,n-2));chosen=sorted(set([eligible[0],eligible[-1]]+[eligible[int(rng.choice(v))] for v in bins if len(v)])) if frac<1 else eligible
                selections[f'{month}_{frac}_{seed}']=chosen
                for stock,s in d.groupby('symbol'):
                    _,a,b=next(x for x in folds(s) if x[0]==month);a=a[a.start_utc.dt.strftime('%Y-%m-%d').isin(chosen)]
                    m=model(100,'l2',.1).fit(a,a.label)
                    curve.append(dict(symbol=stock,month=month,fraction=frac,seed=seed,days=len(chosen),train_start=str(a.start_utc.min()),train_end=str(a.end_utc.max()),nonzero=int(np.count_nonzero(m[-1].coef_)),**{'train_'+k:v for k,v in metrics(a.label,m.predict_proba(a)[:,1]).items()},**metrics(b.label,m.predict_proba(b)[:,1])))
    pd.DataFrame(curve).to_csv(out/'learning_curve.csv',index=False);dump(out/'sampled_dates.json',selections)

def updates(d,out):
    protocol=json.loads((O/'stock_final_test/protocol.json').read_text());preds=[];grid=[]
    for stock,s in d.groupby('symbol'):
        spec=protocol['models'][stock]['body_l2'];path=ROOT/spec['path'];assert sha(path)==spec['sha256'];original=joblib.load(path);C=spec['C']
        for month in sorted(s.loc[~s.split.eq('train'),'start_utc'].dt.strftime('%Y-%m').unique()):
            a=old.train_rows(s,month,'expanding');b=s[s.start_utc.dt.strftime('%Y-%m').eq(month)];temporal(a,b)
            for arm in ['frozen','coefficients','transforms','retuned']:
                if arm=='frozen':m=original
                elif arm=='coefficients':
                    transform=original.named_steps['features'];lr=clone(original[-1]).fit(transform.transform(a),a.label);m=Pipeline([('features',transform),('model',lr)])
                elif arm=='transforms':m=model(500,'l2',C).fit(a,a.label)
                else:
                    c,g=old.select(a,month,'expanding',lambda c:model(500,'l2',c),[.01,.1,1.]);g['symbol']=stock;g['month']=month;grid.append(g);m=model(500,'l2',c).fit(a,a.label)
                p=m.predict_proba(b)[:,1];train=s[s.split.eq('train')] if arm=='frozen' else a
                save_model(m,train,b,out/stock/month/arm,p);preds.append(pred_rows(b,p,arm))
        print('updates complete',stock,flush=True)
    pd.concat(preds).to_csv(out/'predictions.csv',index=False);pd.concat(grid).to_csv(out/'cv.csv',index=False)
