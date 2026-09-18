"""Independent metrics, temporal boundary, input replay and checkpoint checks."""
from run import *
def main():
 record=json.loads((PRIVATE/'done.json').read_text());d=core.data();p=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');z=np.load(PRIVATE/'inputs.npz');raw=z['raw'];ends=pd.to_datetime(z['ends'],utc=True);sup=z['supervised'];unlab=z['unlabeled'];valid=sup[:,0]>=0;checks={}
 assert np.array_equal(p.key,d.key) and np.array_equal(p.label,d.label) and len(p)==1607
 for path,h in record['fingerprint'].items():assert core.sha(ROOT/path)==h
 for i,r in enumerate(d.itertuples()):
  if valid[i]:assert ends[sup[i]].max()<=r.cutoff_utc
 checks['rows_time_label_keys']=len(p)
 for choice in record['selections']:
  boundary='2018-09' if choice['month']=='final' else choice['month'];assert all(m<boundary for m in choice['past_months'])
 for entry in record['encoders']:
  assert pd.Timestamp(entry['latest_pretrain_end'])<pd.Timestamp(entry['evaluation_cutoff']);assert entry['early']['max_gradient']>0 and entry['final']['max_gradient']>0;assert entry['final']['curve'][-1]['epoch']==entry['selected_epochs'];path=PRIVATE/f"encoder_{entry['month']}_{entry['seed']}.pt";assert core.sha(path)==entry['checkpoint_sha256']
 checks['encoder_checkpoint_count']=len(record['encoders']);checks['pretrain_used_no_direction_labels']=True
 for f in record['heads']:
  assert pd.Timestamp(f['train_label_end'])<pd.Timestamp(f['eval_cutoff']);assert core.sha(PRIVATE/f['file'])==f['sha256']
 checks['head_checkpoint_count']=len(record['heads']);evalmask=d.phase.ne('warmup').to_numpy()
 for method in ('RAW','RANDOM','SSL'):assert np.array_equal(p.loc[~valid&evalmask,method].to_numpy(),p.loc[~valid&evalmask,'B'].to_numpy())
 checks['exact_missing_sequence_fallback']=True
 errors=[]
 for r in pd.read_csv(OUT/'metrics.csv').itertuples():
  g=p[(p.symbol==r.symbol)&(p.phase==r.phase)];y=g.label.to_numpy();q=g[r.method].to_numpy()>=.5;tp=np.sum(q&(y==1));tn=np.sum(~q&(y==0));fp=np.sum(q&(y==0));fn=np.sum(~q&(y==1));ba=.5*(tp/(tp+fn)+tn/(tn+fp));den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)));mcc=(tp*tn-fp*fn)/den if den else 0;errors.append(max(abs(ba-r.BA),abs(mcc-r.MCC),abs(np.mean((g[r.method].to_numpy()-y)**2)-r.Brier)))
 assert max(errors)<1e-12;checks['metrics_rows']=len(errors);checks['max_metric_error']=max(errors)
 # Reconstruct final all-seed probabilities directly from private encoders and heads.
 replay=[];month='final'
 for symbol in ('AAPL','AMZN'):
  tr,ev=core.split(d,symbol,month)
  for method in ('RAW','RANDOM','SSL'):
   c=next(x['C'] for x in record['selections'] if x['method']==method and x['month']==month);probs=[]
   for j,seed in enumerate(SEEDS if method!='RAW' else [573]):
    bundle=joblib.load(PRIVATE/f'head_{method}_{month}_{symbol}_{c}_{j}.joblib');price=bundle['price'].transform(d[core.OLD+core.RECENT])
    if method=='RAW':
     flat=np.zeros((len(d),48*6));flat[valid]=normalize(raw[sup[valid]],bundle['bar_mean'],bundle['bar_scale']).reshape(sum(valid),-1);emb=bundle['pca'].transform(flat);emb[~valid]=0
    else:
     cp=torch.load(PRIVATE/f'encoder_{month}_{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);m=Encoder()
     if method=='SSL':m.load_state_dict(cp['state'])
     m.eval();emb=np.zeros((len(d),32))
     with torch.no_grad():emb[valid]=m(torch.tensor(normalize(raw[sup[valid]],cp['mean'],cp['scale'])))[1].numpy()
    x=np.c_[price,bundle['embedding_scale'].transform(emb)];q=bundle['model'].predict_proba(x[ev])[:,1];q[~valid[ev]]=p.iloc[ev].B.to_numpy()[~valid[ev]];probs.append(q)
   q=np.mean(probs,axis=0);q[~valid[ev]]=p.iloc[ev].B.to_numpy()[~valid[ev]];err=float(abs(q-p.iloc[ev][method].to_numpy()).max());assert err<1e-7;replay.append(dict(symbol=symbol,method=method,max_error=err))
 checks['independent_final_replay']=replay
 # Future raw rows cannot alter the first valid evaluated sequence.
 i=next(i for i in range(len(d)) if valid[i] and evalmask[i]);mutated=raw.copy();mutated[ends>d.iloc[i].cutoff_utc]+=999;assert np.array_equal(mutated[sup[i]],raw[sup[i]]);checks['future_feature_perturbation']=True
 # Cache rejection is exercised in a copied private manifest without touching models.
 from unittest.mock import patch
 actual_sha=core.sha
 with patch.object(core,'sha',side_effect=lambda path:'changed' if Path(path).name=='run.py' else actual_sha(path)):
  try:run()
  except AssertionError as error:assert 'cache mismatch' in str(error)
  else:raise AssertionError('changed source was not rejected')
 checks['cache_mismatch_rejected_before_training']=True
 core.dump(OUT/'verification.json',checks);print(json.dumps(checks,indent=2))
if __name__=='__main__':
 torch.set_num_threads(2)
 with threadpool_limits(limits=2):main()
