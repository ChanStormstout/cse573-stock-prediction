"""Compare matched extraction outputs; no forecast labels or fitting."""
import argparse, collections, csv, json
from pathlib import Path
from labels import rows, dump, write_rows, sha, ROOT
import sys
sys.path.insert(0,str(ROOT/'outputs/stock_llm_4h'))
from common import evaluate, signature, check_panel

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--rule',type=Path,required=True);p.add_argument('--frozen',type=Path,required=True);p.add_argument('--tuned',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();check_panel(a.data)
 labels=rows(a.data/'labels.jsonl');inputs={r['id']:r for r in rows(a.data/'inputs.jsonl')};ls={r['id']:r for r in labels};models={k:rows(getattr(a,k)/'predictions.jsonl') for k in ['rule','frozen','tuned']}
 expected={r['id'] for r in labels if r['split']=='check'}
 for k,ps in models.items():
  ids=[r['id'] for r in ps if r['id'] in expected]
  if set(ids)!=expected or len(set(ids))!=len(ids):raise ValueError((k,'incomplete/duplicate check predictions'))
 a.out.mkdir(parents=True,exist_ok=False);table=[]
 for k,ps in models.items():
  for cohort in ['all_check','original_check','enriched_check']:
   for symbol in ['ALL','AAPL','AMZN']:
    sub=[r for r in labels if r['split']=='check' and (symbol=='ALL' or r['symbol']==symbol) and (cohort=='all_check' or r['id'].startswith('N' if cohort=='original_check' else 'Q'))]
    if not sub:continue
    m=evaluate(ps,sub)['check']
    predmap={r['id']:r for r in ps};presence_tp=presence_fp=presence_fn=0
    for r in sub:
     prediction=predmap[r['id']];actual=bool(r['answer']['events']);positive=bool(prediction['valid'] and prediction['parsed']['events'])
     presence_tp+=actual and positive;presence_fp+=not actual and positive;presence_fn+=actual and not positive
    m.update(event_presence_tp=presence_tp,event_presence_fp=presence_fp,event_presence_fn=presence_fn)
    table.append({'model':k,'cohort':cohort,'symbol':symbol,**m})
 with (a.out/'metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 cases=[];pred={k:{r['id']:r for r in v} for k,v in models.items()}
 for key in sorted(expected):
  gold={signature(e) for e in ls[key]['answer']['events']};states={}
  for k in models:
   r=pred[k][key];got={signature(e) for e in r['parsed']['events']} if r['valid'] else set()
   states[k]={'exact':r['valid'] and gold==got,'matched':len(gold&got),'extra':len(got-gold),'missed':len(gold-got),'valid':r['valid']}
  cohort='both_correct' if states['frozen']['exact'] and states['tuned']['exact'] else 'fixed' if states['tuned']['exact'] else 'regressed' if states['frozen']['exact'] else 'both_wrong'
  cases.append({'id':key,'symbol':ls[key]['symbol'],'cohort':cohort,'has_gold_event':bool(gold),'models':states})
 write_rows(a.out/'case_index.jsonl',cases)
 # Raw source stays in local work only.
 panel=[c for c in cases if c['has_gold_event'] or c['cohort'] in {'fixed','regressed'}]
 write_rows(a.out/'case_review.jsonl',[{**c,'input':inputs[c['id']],'gold':ls[c['id']]['answer'],'predictions':{k:pred[k][c['id']] for k in models}} for c in panel])
 dump(a.out/'summary.json',{'evaluation':'EXTRACTION_NOT_STOCK_PREDICTION','independent_human_review_passed':False,'input_sha256':sha((a.data/'inputs.jsonl').read_bytes()),'n_check':len(expected),'case_groups':dict(collections.Counter(c['cohort'] for c in cases)),'positive_check_articles':sum(c['has_gold_event'] for c in cases),'warning':'Selected-input, model-provisional labels; enriched challenge is not natural prevalence; event grouping incomplete. Semantic evidence support still needs review. No four-hour BA result in this file.'})
 print(json.dumps(table,indent=2))
if __name__=='__main__':main()
