import sys,json,time,gc
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib
from model import fit,probabilities,WindowModel,forward_rows,fit_transform
B=Path(__file__).resolve().parent;O=B.parent;sys.path.insert(0,str(O/'stock_robust'))
from common import verify_files,require_empty,sha,fold_indices,inner_split,metric_details,scores
P=json.loads((B/'protocol.json').read_text());M=json.loads((B/'results/prepared.json').read_text());C=Path(M['cache_dir']);verify_files(O,M['source_sha256']);verify_files(C,M['cache_sha256']);verify_files(B/'results',M['artifact_sha256']);R=B/'results/training';require_empty(R);R.mkdir(exist_ok=True);(R/'started.json').write_text('{"status":"training"}')
torch.set_num_threads(2);d=pd.read_csv(B/'results/manifest.csv');z=np.load(C/'data.npz');data={k:z[k] for k in ['bags','numeric','pooled','has','stock','y']};data['prefix']=torch.load(C/'prefix.pt',weights_only=True,map_location='cpu');data['tail_state']=torch.load(C/'tail_state.pt',weights_only=True,map_location='cpu');data['config']=json.loads((C/'config.json').read_text())
tr=np.flatnonzero(d.split.eq('train'));va=np.flatnonzero(d.split.eq('validation'));assert len(tr)==2000 and len(va)==504
result=dict(protocol=P,models={},test_evaluated=False);pred=d.iloc[va][['symbol','start_utc','label','has_news']].copy();start=time.time()


def free(net):
    del net;gc.collect()
    if torch.backends.mps.is_available():torch.mps.empty_cache()


for kind in ['frozen','lora']:
    F=R/kind;F.mkdir();grid=[];traces=[]
    for month,a,b in fold_indices(d):
        ia,ib=inner_split(d,a,P['training']['inner_days'])
        for seed in P['training']['seeds']:
            net,transform,epochs,trace=fit(kind,data,ia,ib,seed,P);free(net)
            net,transform,_,refit_trace=fit(kind,data,a,None,seed,P,fixed_epochs=epochs);p=probabilities(net,data,b)
            for s in ['AAPL','AMZN']:
                mask=d.symbol.iloc[b].eq(s).to_numpy();grid.append(dict(month=month,seed=seed,epochs=epochs,symbol=s,inner_train_end=d.end_utc.iloc[ia].max(),inner_monitor_start=d.cutoff_utc.iloc[ib].min(),outer_start=d.cutoff_utc.iloc[b].min(),**scores(data['y'][b][mask],p[mask])))
            traces.append(dict(month=month,seed=seed,epochs=epochs,inner_trace=trace,refit_trace=refit_trace));free(net)
            print('E10',kind,'fold',month,'seed',seed,'epochs',epochs,'elapsed',round(time.time()-start),'s',flush=True)
    pd.DataFrame(grid).to_csv(F/'training_cv.csv',index=False);(F/'fold_traces.json').write_text(json.dumps(traces,indent=2));runs={};ia,ib=inner_split(d,tr,P['training']['inner_days'])
    for seed in P['training']['seeds']:
        net,transform,epochs,trace=fit(kind,data,ia,ib,seed,P);free(net)
        net,transform,_,refit_trace=fit(kind,data,tr,None,seed,P,fixed_epochs=epochs);p=probabilities(net,data,va);p_train=probabilities(net,data,tr);pred[f'{kind}_{seed}']=p
        torch.save(net.state_dict(),F/f'{seed}.pt');joblib.dump(transform,F/f'{seed}_transform.joblib')
        # Only adapters/head can change; compare every frozen tail tensor.
        check={"trainable_parameters":sum(v.numel() for v in net.parameters() if v.requires_grad)}
        if kind=='lora':
            changed=[]
            for key,value in net.tail.state_dict().items():
                if key.endswith(('.A','.B')):
                    if key.endswith('.B'):changed.append(float(value.abs().max().cpu()))
                    continue
                original_key=key.replace('.query.base.','.query.').replace('.value.base.','.value.')
                assert torch.equal(value.cpu(),data['tail_state'][original_key]),key
            check['max_abs_adapter_B']=max(changed);assert max(changed)>0
        runs[str(seed)]=dict(epochs=epochs,inner_trace=trace,refit_trace=refit_trace,checks=check,stocks={s:dict(training=scores(data['y'][tr][d.symbol.iloc[tr].eq(s)],p_train[d.symbol.iloc[tr].eq(s)]),validation=metric_details(d.iloc[va].loc[mask],p[mask.to_numpy()])) for s in ['AAPL','AMZN'] if (mask:=d.symbol.iloc[va].eq(s)).any()})
        print('E10',kind,'FINAL',seed,'epochs',epochs,'BA',[runs[str(seed)]['stocks'][s]['validation']['overall']['balanced_accuracy'] for s in ['AAPL','AMZN']],'elapsed',round(time.time()-start),'s',flush=True);free(net)
    result['models'][kind]=dict(seeds=runs,training_cv=pd.DataFrame(grid).groupby('symbol')[['balanced_accuracy','brier']].mean().to_dict('index'))
    pred.to_csv(R/'validation_predictions.csv',index=False);(R/'results.json').write_text(json.dumps(result,indent=2))
result['elapsed_seconds']=time.time()-start;result['prepared_sha256']=sha(B/'results/prepared.json');(R/'results.json').write_text(json.dumps(result,indent=2));print('E10 complete. Final test not evaluated.',flush=True)
