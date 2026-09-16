import sys,json,time,random,copy
import numpy as np,pandas as pd,torch,joblib
from common import B,O,prepared,fold_indices,inner_split,metric_details,scores,require_empty,sha
sys.path.insert(0,str(O/'stock_temporal'))
from models import TemporalNet,fit_scaler,transform,predict,train_net
D,M=prepared();P=json.loads((B/'protocol.json').read_text());N=P['temporal'];R=B/'results/temporal';require_empty(R);R.mkdir(exist_ok=True);(R/'started.json').write_text('{"status":"training"}')
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);started=time.time();result=dict(protocol=P,stocks={},test_evaluated=False)


def select_epoch(kind,hidden,X,y,length,a,b,seed,wd):
    scaler=fit_scaler(X[a]);aa=transform(X[a],scaler);bb=transform(X[b],scaler)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);net=TemporalNet(kind,hidden=hidden);opt=torch.optim.AdamW(net.parameters(),lr=N['learning_rate'],weight_decay=wd)
    xx=torch.tensor(aa);yy=torch.tensor(y[a],dtype=torch.float32);ll=torch.tensor(length[a]);trace=[];best=float('inf');epoch_best=0;stale=0
    for epoch in range(1,N['max_epochs']+1):
        net.train()
        for ids in torch.randperm(len(a)).split(N['batch_size']):
            opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(net(xx[ids],ll[ids]),yy[ids]);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        p=predict(net,bb,length[b]);q=np.clip(p,1e-7,1-1e-7);bce=float(-np.mean(y[b]*np.log(q)+(1-y[b])*np.log(1-q)))
        trace.append(dict(epoch=epoch,inner_bce=bce,inner_brier=float(np.mean((p-y[b])**2)),training_brier=float(np.mean((predict(net,aa,length[a])-y[a])**2))))
        if bce<best-N['min_delta']:best=bce;epoch_best=epoch;stale=0
        else:stale+=1
        if stale>=N['patience']:break
    assert epoch_best>0
    return epoch_best,trace


for s,d in D.groupby('symbol'):
    d=d.reset_index(drop=True);z=np.load(O/f'stock_temporal/results/{s}/data.npz');X=z['X'];y=z['y'];length=z['lengths'];np.testing.assert_array_equal(y,d.label);tr=np.flatnonzero(d.split.eq('train'));va=np.flatnonzero(d.split.eq('validation'));F=R/s;F.mkdir();pred=d.iloc[va][['symbol','start_utc','label','has_news']].copy();result['stocks'][s]={}
    for variant in N['variants']:
        kind=variant.split('_')[0];hidden=int(variant.rsplit('h',1)[-1]);grid=[];traces=[]
        for wd in N['weight_decay']:
            for month,a,b in fold_indices(d):
                ia,ib=inner_split(d,a,N['inner_days'])
                for seed in N['seeds']:
                    epochs,trace=select_epoch(kind,hidden,X,y,length,ia,ib,seed,wd);sc=fit_scaler(X[a]);aa=transform(X[a],sc);bb=transform(X[b],sc)
                    config=dict(GRU_hidden=hidden,epochs=epochs,learning_rate=N['learning_rate'],batch_size=N['batch_size'],gradient_clip=1.)
                    net,_=train_net(kind,aa,y[a],length[a],seed,wd,config);p=predict(net,bb,length[b]);grid.append(dict(weight_decay=wd,month=month,seed=seed,epochs=epochs,n_train=len(a),inner_train_end=d.end_utc.iloc[ia].max(),inner_validation_start=d.cutoff_utc.iloc[ib].min(),outer_validation_start=d.cutoff_utc.iloc[b].min(),**scores(y[b],p)))
                    traces.append(dict(weight_decay=wd,month=month,seed=seed,chosen_epoch=epochs,trace=trace))
            print(s,variant,'wd',wd,'folds done',round(time.time()-started),'s',flush=True)
        g=pd.DataFrame(grid);best=g.groupby('weight_decay')[['balanced_accuracy','brier']].mean().reset_index().sort_values(['balanced_accuracy','brier','weight_decay'],ascending=[False,True,False]).iloc[0];wd=float(best.weight_decay);runs={}
        sc=fit_scaler(X[tr]);aa=transform(X[tr],sc);bb=transform(X[va],sc);joblib.dump(sc,F/f'{variant}_scaler.joblib');ia,ib=inner_split(d,tr,N['inner_days'])
        for seed in N['seeds']:
            epochs,trace=select_epoch(kind,hidden,X,y,length,ia,ib,seed,wd);config=dict(GRU_hidden=hidden,epochs=epochs,learning_rate=N['learning_rate'],batch_size=N['batch_size'],gradient_clip=1.)
            net,train_trace=train_net(kind,aa,y[tr],length[tr],seed,wd,config);p=predict(net,bb,length[va]);pred[f'{variant}_{seed}']=p;torch.save(net.state_dict(),F/f'{variant}_{seed}.pt')
            runs[str(seed)]=dict(epochs=epochs,early_stop_trace=trace,refit_trace=train_trace,training=scores(y[tr],predict(net,aa,length[tr])),validation=metric_details(d.iloc[va],p))
        g.to_csv(F/f'{variant}_grid.csv',index=False);(F/f'{variant}_inner_traces.json').write_text(json.dumps(traces,indent=2))
        means={m:float(np.mean([r['validation']['overall'][m] for r in runs.values()])) for m in ['balanced_accuracy','brier','mcc']};stds={m:float(np.std([r['validation']['overall'][m] for r in runs.values()],ddof=1)) for m in means}
        result['stocks'][s][variant]=dict(hidden=hidden,weight_decay=wd,training_cv_BA=float(best.balanced_accuracy),seed_mean=means,seed_std=stds,seeds=runs)
        print(s,variant,'VAL',means,'epochs',[r['epochs'] for r in runs.values()],flush=True)
        (R/'results.json').write_text(json.dumps(result,indent=2))
    pred.to_csv(F/'validation_predictions.csv',index=False)
result['elapsed_seconds']=time.time()-started;result['prepared_sha256']=sha(B/'results/prepared.json');(R/'results.json').write_text(json.dumps(result,indent=2));print('Temporal finished',round(result['elapsed_seconds']),'seconds',flush=True)
