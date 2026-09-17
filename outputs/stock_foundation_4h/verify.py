"""Independent recomputation, temporal mutation, safe-cache resume and weight replay."""
import json,tempfile,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer,AutoModel
import experiment as ex
import chronos_experiment as ce
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef,brier_score_loss

def main():
 d,*_=ex.inputs();p=pd.read_csv(ex.OUT/'all_predictions.csv');m=pd.read_csv(ex.OUT/'all_metrics.csv');assert len(p)==1607 and p.key.is_unique
 assert np.array_equal(d.key,p.key) and np.array_equal(d.label,p.label)
 for r in m.itertuples():
  g=p[(p.phase==r.phase)&(p.symbol==r.symbol)];prob=g[r.method];assert len(g)==r.n
  for a,b in [(balanced_accuracy_score(g.label,prob>=.5),r.BA),(matthews_corrcoef(g.label,prob>=.5),r.MCC),(brier_score_loss(g.label,prob),r.Brier)]:assert abs(a-b)<1e-12
 usable=p.phase!='warmup';no=usable&(p.has_original_news==0)
 for col in ['A01','A1','E01','E1','MODERN']:assert np.array_equal(p.loc[no,col],p.loc[no,'R1'])
 old=pd.read_csv(ex.OUT/'finbert_match_predictions.csv');parity=float(np.max(np.abs(old.loc[usable,'M_F2']-p.loc[usable,'F2'])));assert parity<1e-10
 count=0
 for stage in ['aggregation','finbert_match','modern','chronos']:
  evidence=json.loads((ex.OUT/f'{stage}_training.json').read_text());cv=pd.read_csv(ex.OUT/f'{stage}_cv.csv')
  for r in evidence['fits']:
   assert pd.Timestamp(r['train_end'])<pd.Timestamp(r['cutoff_min']);assert ex.sha(ex.PRIVATE/'models'/r['name'])==r['sha256'];assert r['reload_error']<1e-12;count+=1
  for r in evidence['choices']:
   boundary='2018-09' if r['month']=='final' else r['month'];assert all(x<boundary for x in r['selection_months'])
   if stage!='aggregation':
    prior=cv[(cv.symbol==r['symbol'])&(cv.method==r['method'])&(cv.month<boundary)];assert ex.choose_c(prior.to_dict('records'))==r['C']
 a=pd.read_csv(ex.OUT/'chronos_alignment.csv');assert (pd.to_datetime(a.latest_context_end,utc=True)<=pd.to_datetime(a.cutoff,utc=True)).all();assert ((a.close_step-a.open_step)==47).all()
 context,alignment,features=ce.infer(d) # Default np.load now safely reuses the repaired cache.
 assert len(features)==1607 and np.isfinite(features).all()
 # A forbidden five-minute gap bar is perturbed; allowed inputs must be unchanged.
 index=int(np.flatnonzero(a.open_step.to_numpy()==2)[0]);one=d.iloc[[index]].reset_index(drop=True);before,_,_=ce.prepare(one);original=ce.load_bars
 def mutated(symbol):
  path,b=original(symbol);b=b.copy()
  if symbol==one.symbol.iloc[0]:b.loc[one.cutoff_utc.iloc[0],['open','close']]*=100
  return path,b
 ce.load_bars=mutated;after,_,_=ce.prepare(one);ce.load_bars=original;assert np.array_equal(before,after,equal_nan=True)
 # Reject a mismatched fingerprint without altering the real experiment directory.
 original_private=ex.PRIVATE
 with tempfile.TemporaryDirectory() as td:
  ex.PRIVATE=Path(td);(ex.PRIVATE/'source_hashes.json').write_text('{}')
  try:ex.init()
  except AssertionError:pass
  else:raise AssertionError('cache mismatch accepted')
 ex.PRIVATE=original_private
 # Reload the frozen ModernBERT checkpoint, reproduce the first original batch.
 torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu';mp=ex.W/'models/modern';tok=AutoTokenizer.from_pretrained(mp,local_files_only=True);model=AutoModel.from_pretrained(mp,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager',reference_compile=False).eval().to(device)
 z=np.load(ex.PRIVATE/'modern_embeddings.npz');manifest=json.loads((ex.ROOT/'outputs/stock_integrated_4h/prepared/embedding_inputs.json').read_text());titles=dict(manifest['keys_and_titles']);x=tok([titles[k] for k in z['keys'][:32]],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
 with torch.inference_mode():
  mask=x['attention_mask'].unsqueeze(-1);h=model(**x).last_hidden_state;v=((h*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu().numpy()
 err=float(np.max(np.abs(v-z['embeddings'][:32])));assert err<1e-5
 r=dict(status='PASS',rows=len(d),metric_rows=len(m),classifier_fits=count,finbert_reproduction_error=parity,modern_weight_replay_error=err,checks=['original labels and unique keys','BA/MCC/Brier independent recomputation','all model weight hashes','all training boundaries','C selected from earlier folds only','no-news exact fallback','Chronos target steps 48/49','forbidden gap-bar mutation leaves inputs unchanged','cache mismatch rejected','safe Unicode cache resume','frozen ModernBERT weight replay'],implementation_repairs='Chronos local object-key cache migrated to Unicode with original artifacts preserved; numerical forecasts unchanged')
 ex.dump(ex.OUT/'verification.json',r);print(json.dumps(r,indent=2))
if __name__=='__main__':main()
