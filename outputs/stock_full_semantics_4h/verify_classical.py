"""No-fit replay and selection reconstruction, then publish exposed random scores."""
from classical import *
def main():
 check_sources();d,*_=load();outer=pd.read_csv(BASE/'outer_folds.csv');frames=[];error=0.;models=0
 seal=DEST/'seal.json';assert json.loads(seal.read_text())['code']==sha(HERE/'classical.py')
 for seed in [573,574,575]:
  for stock in ['AAPL','AMZN']:
   for fold in range(10):
    root=DEST/f'{seed}_{stock}_{fold}';complete=json.loads((root/'complete.json').read_text());assert complete['seal']==sha(seal)
    for n,h in complete['files'].items():assert sha(root/n)==h
    pred=pd.read_csv(root/'predictions.csv',float_precision='round_trip');records=pd.read_csv(root/'inner_metrics.csv');ev=outer[(outer.seed==seed)&(outer.symbol==stock)&(outer.fold==fold)]['index'].to_numpy(int)
    assert pred.row_id.tolist()==d.iloc[ev].row_id.tolist();assert pred.label.tolist()==d.iloc[ev].label.tolist()
    for info in json.loads((root/'models.json').read_text()):
     method=info['method'];r=records[records.method==method];candidates=CAND[method]
     selected=min(candidates,key=lambda v:(-r[r.value==v].BA.mean(),r[r.value==v].Brier.mean(),candidates.index(v)))
     assert selected==info['selected'];assert info['evaluation']==ev.tolist()
     assert set(info['train'])==set(np.flatnonzero(d.symbol==stock))-set(ev)
     m=joblib.load(root/(method+'.joblib'));p=m.predict_proba(d.iloc[ev].stem_body.fillna(''))[:,1]
     error=max(error,float(np.max(abs(pred[method]-p))));assert error<=1e-12;models+=1
    baseline=pd.read_csv(BASE/'baseline'/f'{seed}_{stock}_{fold}'/'outer_predictions_SEALED.csv',float_precision='round_trip')
    assert baseline.row_id.tolist()==pred.row_id.tolist()
    for method in ['PAPER','PRICE','FULL','FINBERT','MODERN']:pred[method]=baseline[method]
    frames.append(pred)
 allp=pd.concat(frames,ignore_index=True);assert len(allp)==1607*3
 metrics=[]
 for (stock,seed),g in allp.groupby(['symbol','seed']):
  for method in list(CAND)+['PAPER','PRICE','FULL','FINBERT','MODERN']:
   base=g.FULL.to_numpy()>=.5;q=g[method].to_numpy()>=.5;y=g.label.to_numpy()
   metrics.append(dict(symbol=stock,seed=seed,method=method,**metric(y,g[method]),changed=int(sum(q!=base)),repaired=int(sum((base!=y)&(q==y))),introduced=int(sum((base==y)&(q!=y)))))
 pd.DataFrame(metrics).to_csv(OUT/'CLASSICAL_METRICS.csv',index=False);allp.to_csv(OUT/'CLASSICAL_PREDICTIONS.csv',index=False,float_format='%.17g')
 dump(OUT/'CLASSICAL_VERIFICATION.json',dict(status='PASS',selected_models=models,probabilities_replayed=len(allp)*3,max_reload_error=error,checks=['source seal','code seal','all artifact hashes','canonical row order and labels','outer membership','training membership disjoint','inner C/depth reconstruction','independent saved-model probability replay'],fit_calls=0))
 print('PASS',models,error,flush=True)
if __name__=='__main__':main()
