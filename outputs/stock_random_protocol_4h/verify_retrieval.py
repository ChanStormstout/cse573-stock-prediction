"""Independently inspect all case memberships/times; no model or outcome scoring."""
import json
import pandas as pd,numpy as np
from common import PRIVATE,OUT,SOURCES,sha,dump

def main():
 src=PRIVATE/'analogy';d=pd.read_pickle(SOURCES['rows']).reset_index(drop=True)
 manifest=json.loads((src/'manifest.json').read_text());docs=json.loads((src/'documents.json').read_text());jobs=json.loads((src/'jobs.json').read_text())
 scopes={s['scope']:s for s in json.loads((src/'scopes.json').read_text())};prompts={r['prompt_hash']:r for r in map(json.loads,(src/'prompts.jsonl').read_text().splitlines())}
 checks=dict(artifacts_unchanged=all(sha(src/n)==h for n,h in manifest['artifacts'].items()),query_membership=True,case_label_membership=True,strictly_matured=True,past_available_within_30_days=True,same_stock=True,at_most_three_distinct_records=True,no_target_answer_in_current=True,source_time_matches=True)
 ncases=0;prompt_labels=0;seen=set()
 for job in jobs:
  scope=scopes[job['scope']];i=job['index'];q=d.iloc[i];ids=job['cases']
  checks['query_membership'] &= i in scope['validation'] and i not in scope['train'] and (job['scope'],i) not in seen;seen.add((job['scope'],i))
  checks['at_most_three_distinct_records'] &= len(ids)<=3 and len({docs[j]['record_key'] for j in ids})==len(ids)
  for j in ids:
   r=d.iloc[j];ncases+=1
   checks['case_label_membership'] &= j in scope['train'] and j not in scope['validation']
   checks['strictly_matured'] &= pd.Timestamp(r.end_utc)<pd.Timestamp(q.cutoff_utc)
   age=(pd.Timestamp(docs[i]['available'])-pd.Timestamp(docs[j]['available'])).total_seconds()/86400
   checks['past_available_within_30_days'] &= 0<age<=30 and pd.Timestamp(docs[j]['available'])<=pd.Timestamp(q.cutoff_utc)
   checks['same_stock'] &= r.symbol==q.symbol
  for variant,h in job['prompts'].items():
   payload=json.loads(prompts[h]['messages'][1]['content']);current=payload['CURRENT']
   checks['no_target_answer_in_current'] &= set(current)=={'company','cutoff','target_start','target_end','title','evidence','available','published','price'}
   checks['source_time_matches'] &= pd.Timestamp(current['cutoff'])==pd.Timestamp(q.cutoff_utc) and pd.Timestamp(current['target_start'])==pd.Timestamp(q.start_utc) and pd.Timestamp(current['target_end'])==pd.Timestamp(q.end_utc)
   hist=payload.get('HISTORICAL_CASES',[])
   checks['case_label_membership'] &= len(hist)==(len(ids) if variant=='ANALOGY' else 0)
   for j,case in zip(ids,hist):
    from common import digest
    checks['case_label_membership'] &= case['case_id']==digest(d.iloc[j].key)[:24]
    # Raw independently reconstructed returns were checked at preparation; additionally bind label direction here.
    checks['case_label_membership'] &= int(case['four_hour_return_pct']>0)==int(d.iloc[j].label);prompt_labels+=1
 checks['all_scopes_exhaustive']=len(jobs)==sum(len(s['validation']) for s in scopes.values())
 result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},query_evaluations=len(jobs),case_references=ncases,prompt_case_outcomes_checked=prompt_labels,estimator_fits=0,predictive_metrics_computed=False)
 dump(OUT/'ANALOGY_VERIFICATION.json',result);print(json.dumps(result,indent=2))
 if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
