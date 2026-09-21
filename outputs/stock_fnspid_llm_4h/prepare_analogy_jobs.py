#!/usr/bin/env python3
"""Build time-safe case-conditioned L4 jobs after L1 article scoring."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;PUBLIC=HERE/'v1'
PRIVATE=ROOT/'work/stock-data/fnspid_llm_4h/v1';SOURCE=ROOT/'work/stock-data/fnspid_augmented_4h/v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n')
def load_jsonl(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
SYSTEM=("You forecast the next four-hour direction by comparing only the supplied current news with strictly earlier historical cases. "
        "Historical outcomes are examples, not rules. Return exactly UP or DOWN with no explanation.")

def main():
    out=PRIVATE/'analogy_jobs.jsonl'
    if out.exists():raise FileExistsError(out)
    scores=load_jsonl(PRIVATE/'article_scores.jsonl');score={(x['article_key'],x['task']):x['p_A'] for x in scores}
    if len(scores)!=3966:raise RuntimeError('article inference incomplete')
    d=pd.read_pickle(SOURCE/'inputs.pkl').sort_values(['start_utc','symbol']).reset_index(drop=True)
    z=np.load(SOURCE/'combined_embeddings.npz');keys=z['keys'].astype(str);lookup={k:i for i,k in enumerate(keys)};emb=z['finbert'].astype(float)
    ids=[[lookup[k] for k in str(x).split('|') if k] for x in d.aug_news_record_keys.fillna('')]
    means=np.asarray([emb[ii].mean(0) if ii else np.zeros(emb.shape[1]) for ii in ids]);means/=np.maximum(np.linalg.norm(means,axis=1,keepdims=True),1e-12)
    course=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl');course['article_key']=course.archive+'::'+course.member;titles=dict(zip(course.article_key,course.title.fillna('').astype(str)))
    ext=pd.read_parquet(SOURCE/'fnspid_articles.parquet');titles.update(dict(zip(ext.article_key,ext.title.fillna('').astype(str))))
    jobs=[];case_records=[];violations=[]
    for i,r in enumerate(d.itertuples(index=False)):
      if r.phase=='warmup':continue
      current_keys=[k for k in str(r.aug_news_record_keys).split('|') if k and (not k.startswith('FNSPID::') or score[(k,'quality')]>=.60)]
      if not current_keys:continue
      bank=(d.symbol.eq(r.symbol)&(d.end_utc<r.cutoff_utc))
      if r.month>='2018-09':bank&=d.month.lt('2018-09')
      pool=np.flatnonzero(bank.to_numpy())
      if not len(pool):continue
      sims=means[pool]@means[i];top=pool[np.argsort(-sims,kind='stable')[:3]]
      current='; '.join(titles[k][:220] for k in current_keys[-6:])
      cases=[]
      for rank,j in enumerate(top,1):
        if not (d.iloc[j].end_utc<r.cutoff_utc):violations.append((r.key,d.iloc[j].key))
        ckeys=[k for k in str(d.iloc[j].aug_news_record_keys).split('|') if k]
        cases.append(f"CASE {rank} | direction={'UP' if d.iloc[j].label else 'DOWN'} | similarity={float(means[j]@means[i]):.3f} | news="+'; '.join(titles[k][:160] for k in ckeys[-3:]))
        case_records.append({'query_key':r.key,'case_key':d.iloc[j].key,'query_cutoff':str(r.cutoff_utc),'case_end':str(d.iloc[j].end_utc),'case_label':int(d.iloc[j].label),'similarity':float(means[j]@means[i])})
      prompt=f"TARGET={r.symbol}\nCURRENT NEWS\n{current}\n\nSTRICTLY EARLIER CASES\n"+'\n'.join(cases)+"\n\nPredict the direction for CURRENT. ANSWER:"
      jobs.append({'job_id':r.key+'::analogy','key':r.key,'symbol':r.symbol,'task':'analogy','system':SYSTEM,'prompt':prompt,'options':['UP','DOWN'],'case_keys':[d.iloc[j].key for j in top]})
    if violations:raise RuntimeError(f'future case leakage {violations[:3]}')
    with out.open('x') as f:
      for x in jobs:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    pd.DataFrame(case_records).to_csv(PRIVATE/'analogy_cases.csv',index=False)
    audit={'status':'PREPARED','jobs':len(jobs),'evaluated_rows':int((d.phase!='warmup').sum()),'fallback_rows':int((d.phase!='warmup').sum())-len(jobs),'case_rows':len(case_records),'future_case_violations':0,'query_labels_in_prompt':False,'article_scores_sha256':sha(PRIVATE/'article_scores.jsonl'),'private_jobs_sha256':sha(out),'private_cases_sha256':sha(PRIVATE/'analogy_cases.csv')}
    dump(PUBLIC/'ANALOGY_JOB_AUDIT.json',audit);print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
