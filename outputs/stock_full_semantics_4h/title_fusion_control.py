"""Finite no-fit title-fusion control; selection OOF is not meta-training."""
from train_semantics import *
def main():
 check_sources();d,*_=load();splits=json.loads((PRIVATE/'inner_folds.json').read_text());rows=[];decisions=[];fallback=0
 for seed in SEEDS:
  for stock in ['AAPL','AMZN']:
   for fold in range(10):
    root=PRIVATE/'baseline'/f'{seed}_{stock}_{fold}';seal=json.loads((root/'complete.json').read_text())
    for n in ['outer_predictions_SEALED.csv','FULL_selection_oof_NOT_META_TRAIN.csv','FINBERT_selection_oof_NOT_META_TRAIN.csv','MODERN_selection_oof_NOT_META_TRAIN.csv']:assert sha(root/n)==seal['artifacts'][n]
    base=pd.read_csv(root/'FULL_selection_oof_NOT_META_TRAIN.csv',float_precision='round_trip').set_index('index');outer=pd.read_csv(root/'outer_predictions_SEALED.csv',float_precision='round_trip')
    for name in ['FINBERT','MODERN']:
     signal=pd.read_csv(root/(name+'_selection_oof_NOT_META_TRAIN.csv'),float_precision='round_trip').set_index('index');candidates=[]
     for w in WEIGHTS:
      metrics=[]
      for s in splits:
       if (s['seed'],s['symbol'],s['fold'])!=(seed,stock,fold):continue
       ix=s['validation'];metrics.append(metric(d.iloc[ix].label,blend(base.loc[ix].p,signal.loc[ix].p,d.iloc[ix].has_news.to_numpy().astype(bool),w)))
      candidates.append(dict(weight=w,BA=float(np.mean([r['BA'] for r in metrics])),Brier=float(np.mean([r['Brier'] for r in metrics]))))
     w=min(candidates,key=lambda r:(-r['BA'],r['Brier'],r['weight']))['weight'];gate=outer.has_news.to_numpy().astype(bool);p=blend(outer.FULL,outer[name],gate,w)
     expected=outer.FULL.to_numpy().copy()
     if w:expected[gate]=(1-w)*expected[gate]+w*outer[name].to_numpy()[gate]
     assert np.array_equal(p,expected);fallback+=int(np.count_nonzero(p[~gate]!=outer.FULL.to_numpy()[~gate]))
     part=outer[['row_id','symbol','day','label']].copy();part['seed']=seed;part['fold']=fold;part['encoder']=name;part['FULL']=outer.FULL;part['p']=p;rows.append(part);decisions.append(dict(seed=seed,symbol=stock,fold=fold,encoder=name,weight=w,candidates=candidates))
 allp=pd.concat(rows,ignore_index=True);metrics=[]
 for (name,stock,seed),g in allp.groupby(['encoder','symbol','seed']):
  y=g.label.to_numpy();b=g.FULL.to_numpy()>=.5;q=g.p.to_numpy()>=.5;metrics.append(dict(encoder=name,symbol=stock,seed=seed,**metric(y,g.p),changed=int(sum(q!=b)),repaired=int(sum((b!=y)&(q==y))),introduced=int(sum((b==y)&(q!=y)))))
 pd.DataFrame(metrics).to_csv(PUBLIC/'TITLE_FUSION_METRICS.csv',index=False);allp.to_csv(PUBLIC/'TITLE_FUSION_PREDICTIONS.csv',index=False,float_format='%.17g');dump(PUBLIC/'TITLE_FUSION_SELECTIONS.json',decisions);dump(PUBLIC/'TITLE_FUSION_AUDIT.json',dict(status='PASS_FIT_FREE_ARITHMETIC_REPLAY',fit_calls=0,rows=len(allp),fallback_mismatches=fallback,protocol_sha256=sha(PUBLIC/'TITLE_FUSION_CONTROL_PROTOCOL.json'),code_sha256=sha(__file__),not_an_independent_encoder_replay=True))
if __name__=='__main__':main()
