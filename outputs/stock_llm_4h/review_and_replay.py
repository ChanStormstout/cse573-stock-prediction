"""Prepare local review materials and exercise the gate on unchanged four-hour outputs."""
from common import *
from forecast import review_gate,FactCorrection
import pandas as pd,numpy as np

def main():
 out=new_run(B/'runs/integration_v2');rs={r['id']:r for r in rows(B/'data/pilot_v3/inputs.jsonl')};labels=rows(B/'data/pilot_v3/labels.jsonl');review=[];expected={}
 candidate=B/'runs/tuned_balanced_v1/predictions.jsonl';preds={p['id']:p for p in rows(candidate)};candidate_sha=file_sha(candidate)
 for r in labels:
  if r['split']!='check':continue
  for kind in ['rating','target_price']:
   rid=r['id']+'|'+kind;prediction=preds[r['id']];obj=prediction.get('parsed') or {};content={'symbol':r['symbol'],'sentences':rs[r['id']]['sentences'],'candidate_prediction_sha256':candidate_sha,'candidate_valid':prediction['valid'],'candidate_raw':prediction['raw'],'proposed':[e for e in obj.get('events',[]) if e.get('kind')==kind]};fp=digest(json.dumps(content,sort_keys=True).encode());expected[rid]=fp
   review.append({'id':rid,'kind':kind,'input_sha256':fp,'event_group':r['event_group'],'reviewer':'','critical_correct':'','event_present':'','missed':'','uncertain':'','notes':'','payload':json.dumps(content,ensure_ascii=False)})
 pd.DataFrame(review).to_csv(out/'independent_review.csv',index=False);dump(out/'expected_review.json',expected)
 quality=review_gate(review,expected);dump(out/'quality.json',quality)
 # Also prove that rerunning an unanswered review does not bypass the stop.
 assert review_gate(review,expected)==quality and quality['status']=='WAITING_OR_FAILED'
 source=ROOT/'outputs/stock_adaptive_4h/runs/zero_bias_v1/predictions.csv';d=pd.read_csv(source)
 base=d.title.to_numpy();got=FactCorrection().predict(np.zeros((len(d),12)),base,np.zeros(len(d),bool));assert np.array_equal(got,base)
 prepared=pd.read_pickle(ROOT/'outputs/stock_integrated_4h/prepared/data.pkl')
 dump(out/'status.json',{'quality':quality,'original_task_windows':len(prepared),'saved_prediction_rows_checked':len(d),'development_and_later_rows_checked':int(d.phase.eq('frozen').sum()),'candidate_predictions_sha256':candidate_sha,'price_plus_title_probability_exactly_preserved':bool(np.array_equal(base,got)),'original_predictions_sha256':file_sha(source),'original_inputs_sha256':file_sha(ROOT/'outputs/stock_integrated_4h/prepared/data.pkl'),'new_four_hour_event_training_executed':False,'reason':'No independent type acceptance; only22 unique training annotations, all AMZN provisional samples no-event. Full corpus extraction and B1–B3 forecast fitting intentionally not started.','status':'INTERFACE_IMPLEMENTED_QUALITY_STOP_ENFORCED','not_a_performance_experiment':True})
 print((out/'status.json').read_text())
if __name__=='__main__':main()
