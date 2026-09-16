"""E13: monthly expanding/rolling versus original static body LR, both penalties."""
import json,time,datetime,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from common import *
B=Path(__file__).resolve().parent;R=B/'results'
if __name__=='__main__':
    require_empty(R);R.mkdir(exist_ok=True)
    paths=[B/'common.py',B/'run.py',O/'stock_robust/common.py',O/'stock_baseline/run_baseline.py',O/'stock_robust/results/data.pkl',O/'stock_final_test/results/test_inputs.pkl',O/'stock_final_test/protocol.json']
    old=json.loads((O/'stock_final_test/protocol.json').read_text())
    for stock in ['AAPL','AMZN']:
        for penalty in ['l1','l2']:paths.append(O.parent/old['models'][stock]['body_'+penalty]['path'])
    protocol=dict(id='E13',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='post-E12 exploratory historical replay; no independent holdout',policies=['frozen','expanding','rolling6'],penalties=['l1','l2'],months=['2018-09','2018-10','2018-11','2018-12','2019-01','2019-02'],C=[.01,.1,1.],selection='last three completed calendar months inside each past training range; mean BA then Brier then larger C; rolling inner folds use only outer six-month range',task='same complete hourly samples, prices and full-body binary stemmed chi2-500 features; all no-news windows retained',threshold=.5,source_sha256={str(p.relative_to(O.parent)):sha(p) for p in paths})
    (B/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.time();d=load_data();d.to_pickle(R/'data.pkl');preds=[];logs=[]
    for stock,sd in d.groupby('symbol'):
        for penalty in protocol['penalties']:
            for policy in protocol['policies']:
                for month in protocol['months']:
                    test=sd[sd.start_utc.dt.strftime('%Y-%m').eq(month)].copy();name=f'{stock}_{penalty}_{policy}_{month}';folder=R/name;folder.mkdir()
                    if policy=='frozen':
                        spec=old['models'][stock]['body_'+penalty];model=joblib.load(O.parent/spec['path']);train=sd[sd.split.eq('train')];C=float(spec['C']);grid=None
                    else:
                        train=train_rows(sd,month,policy)
                        factory=lambda c:baseline('paper_stem_body',c).set_params(model__l1_ratio=int(penalty=='l1'))
                        C,grid=select(train,month,policy,factory,protocol['C']);model=factory(C).fit(train,train.label);grid.to_csv(folder/'cv.csv',index=False)
                    assert train.end_utc.max()<test.cutoff_utc.min()
                    p=model.predict_proba(test)[:,1];joblib.dump(model,folder/'model.joblib')
                    record=dict(symbol=stock,penalty=penalty,policy=policy,month=month,C=C,n_train=len(train),train_end=str(train.end_utc.max()),first_cutoff=str(test.cutoff_utc.min()),train_keys=row_keys(train),test_keys=row_keys(test),training=metrics(train,model.predict_proba(train)[:,1]),evaluation=metrics(test,p))
                    (folder/'manifest.json').write_text(json.dumps(record,indent=2));logs.append(record)
                    names=model['features'].get_feature_names_out();coef=model['model'].coef_[0]
                    pd.DataFrame(dict(feature=names,coefficient=coef)).to_csv(folder/'coefficients.csv',index=False)
                    q=test[['symbol','start_utc','cutoff_utc','label','target_return','has_news','news_count','split']].copy();q['penalty']=penalty;q['policy']=policy;q['month']=month;q['p']=p;preds.append(q)
                    print(name,'C',C,'BA',round(record['evaluation']['balanced_accuracy'],4),'Brier',round(record['evaluation']['brier'],4),flush=True)
    pd.concat(preds,ignore_index=True).to_csv(R/'predictions.csv',index=False)
    verify_files(O.parent,protocol['source_sha256']);(R/'results.json').write_text(json.dumps(dict(models=logs,seconds=time.time()-start),indent=2));(B/'artifact_sha256.json').write_text(json.dumps(snapshot(B),indent=2));print('E13 done',round(time.time()-start),flush=True)
