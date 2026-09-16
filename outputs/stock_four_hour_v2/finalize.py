"""Final verification and evidence packaging; does not train models."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np,pandas as pd,joblib
from run import ROOT,BASE,ARMS
B=Path(__file__).parent
checks={}
source=ROOT/'outputs/stock_horizons/runs/v1/data.pkl';original=pd.read_pickle(source);orig4=original[original.horizon=='4h'];origkeys=set(orig4.symbol+'|'+orig4.start_utc.astype(str))
for name in ['v1','lag5']:
 out=B/'runs'/name;d=pd.read_pickle(out/'main4h.pkl');assert len(d)==1607 and set(d.key)==origkeys
 prov=json.loads((out/'provenance.json').read_text())
 for row in prov:
  for kind in ['fresh','stale']:
   r=row[kind]
   for v in r['used'].values():assert pd.Timestamp(v['last_end'])+pd.Timedelta(r['lag'])<=pd.Timestamp(r['asof'])<=pd.Timestamp(r['cutoff'])
  if pd.Timestamp(row['fresh']['cutoff']).tz_convert('America/New_York').strftime('%H:%M')=='09:25':
   assert not {'opening_gap','session_return'}&set(row['fresh']['used'])
 pred=pd.read_csv(out/'predictions.csv');fits=json.loads((out/'fits.json').read_text());assert all(x['iterations']<3000 for x in fits)
 assert np.isfinite(pred.p).all() and not pred.duplicated(['dataset','key','phase','arm']).any()
 # Reproduce every saved outer/frozen evaluation against the corresponding saved matrix.
 errors=[]
 for f in fits:
  parts=f['name'].split('_');dataset=parts[0]
  frame=pd.read_pickle(out/f'{dataset}.pkl').set_index('key').loc[f['eval_keys']]
  model=joblib.load(out/'models'/f"{f['name']}.joblib")
  if '_inner_' not in f['name']:
   arm=parts[-1];phase='frozen_replay' if '_2018-09_' in f['name'] else 'outer'
   expected=pred[(pred.dataset==dataset)&(pred.phase==phase)&(pred.arm==arm)].set_index('key').loc[f['eval_keys']].p.to_numpy()
   errors.append(float(np.max(np.abs(model.predict_proba(frame[f['features']])[:,1]-expected))))
 assert max(errors)<1e-12
 checks[name]={'rows':len(d),'all_provenance_past_only':True,'model_probability_max_error':max(errors),'fits':len(fits),'converged':True}
 if name=='lag5':
  target=out/'code_snapshot';target.mkdir(exist_ok=True)
  expected=json.loads((out/'manifest.json').read_text())['code_sha256']
  for fn,h in expected.items():
   f=B/fn
   if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest()==h:shutil.copyfile(f,target/fn)
 # Cases include all predictions, and deterministic balanced-by-outcome sample (up to 8 per cell).
 q=pred[pred.dataset=='main4h'].pivot(index=['key','symbol','day','phase','label'],columns='arm',values='p').reset_index()
 oc=(q.O>=.5)==q.label;rc=(q.R>=.5)==q.label
 q['case_type']=np.select([~oc&rc,oc&~rc,~oc&~rc],['fixed','introduced_error','both_wrong'],default='both_correct')
 q=q.merge(d[['key','cutoff_utc','history_end','news_count']+ARMS['R']],on='key',validate='one_to_one');q.to_csv(out/'cases_all.csv',index=False)
 panel=q.assign(sample_hash=q.key.map(lambda x:hashlib.sha256(x.encode()).hexdigest())).sort_values('sample_hash').groupby(['symbol','phase','case_type']).head(8)
 panel.to_csv(out/'case_panel.csv',index=False);checks[name]['case_panel_rows']=len(panel)
 q.groupby(['symbol','phase','case_type']).size().rename('count').to_csv(out/'case_counts.csv')
 d.groupby(['symbol','split'])[[v for v in ARMS['R'] if v.endswith('_valid')]].mean().to_csv(out/'availability_coverage.csv')
files=[source,ROOT/'work/stock-data/audit/xnys_schedule.csv',ROOT/'work/stock-data/raw/CHARTS/APPLE5.csv',ROOT/'work/stock-data/raw/CHARTS/AMAZON5.csv']
checks['current_source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
checks['interpretation']='Hashes captured at final verification; historical files were read only throughout this run.'
(B/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
