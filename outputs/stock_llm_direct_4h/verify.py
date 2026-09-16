"""Independent run integrity checks, no model calls or source mutation."""
import argparse,json,math
from pathlib import Path
from common import B,W,VARIANTS,rows,sha,dump,messages,validate_boundary,parse_forecast

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--baseline',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 seal=json.loads((a.inputs/'manifest.json').read_text());assert sha(a.inputs/'inputs.jsonl')==seal['input_sha'];assert sha(a.inputs/'features.pkl')==seal['features_sha'];assert sha(a.inputs/'coverage.csv')==seal['coverage_sha']
 for p,h in seal['sources'].items():assert sha(p)==h
 inputs=rows(a.inputs/'inputs.jsonl');assert len(inputs)==1607 and len({r['key'] for r in inputs})==1607
 for r in inputs:validate_boundary(r)
 ev=[r for r in inputs if r['interval_start']>='2018-09-01'];assert len(ev)==609;index={r['key']:r for r in ev}
 manifest=json.loads((a.run/'manifest.json').read_text());summary=json.loads((a.run/'summary.json').read_text());assert summary['completed'] and manifest['input_sha']==seal['input_sha'];assert manifest['keys']==list(index)
 for f,h in manifest['code'].items():assert sha(a.run/'code'/f)==h
 for f,h in manifest['model_files'].items():assert sha(W/'model_compare/model'/f)==h
 is_choice=manifest.get('interface')=='binary_choice_logprobs'
 if is_choice:
  from choice import choice_messages
 n=0
 for v in VARIANTS:
  rr=rows(a.run/f'{v}.jsonl');assert [r['key'] for r in rr]==list(index)
  for r in rr:
   inp=index[r['key']];assert r['messages']==(choice_messages(inp,v) if is_choice else messages(inp,v))
   if is_choice:
    lp=[r['logp_up'],r['logp_down']];m=max(lp);z=m+math.log(sum(math.exp(x-m) for x in lp));p=math.exp(lp[0]-z)
    assert abs(p-r['p'])<1e-12 and 0<=r['choice_mass']<=1.00001
    assert r['parsed']['direction']==('UP' if p>=.5 else 'DOWN') and r['valid']
   else:
    forecast=parse_forecast(r['raw'],inp,v)
    for field in ['p','valid','error','parsed','evidence_valid']:assert r[field]==forecast[field]
   assert 0<=r['p']<=1 and math.isfinite(r['p']) and r['prompt_tokens']+manifest['max_output']<=6144
   assert r['finish_reason'] in (['scored'] if is_choice else ['stop','length']);n+=1
  assert sum(r['valid'] for r in rr)==summary['variants'][v]['valid']
 from datetime import datetime
 fits=json.loads((a.baseline/'fits.json').read_text())
 for r in fits:
  assert datetime.fromisoformat(r['max_label_end'])<datetime.fromisoformat(r['cutoff'])
  assert not set(r['train_keys'])&set(r['eval_keys'])
 b=json.loads((a.baseline/'summary.json').read_text());assert len(fits)==b['actual_fits']==60 and b['input_sha']==seal['features_sha']
 for r in b['selected']:assert sha(a.baseline/f"{r['symbol']}_{r['variant']}.joblib")==r['model_sha'] and r['reload_max_error']<=1e-12
 dump(a.out,dict(status='PASS',input_windows=1607,forecast_windows=609,forecasts_verified=n,lr_fits_verified=len(fits),sources_unchanged=True,prompt_whitelist_reproduced=True,output_validation_reproduced=True,future_time_checks=True,model_reload_verified=True,code_snapshots_verified=True,llm_model_hashes_verified=True,no_claim_of_pretraining_decontamination=True))
 print('PASS',n,'forecasts;',len(fits),'past-only LR fits')
if __name__=='__main__':main()
