"""Diagnostic only: same frozen LLM on full selected input vs assistant gold evidence.
Gold-evidence inputs are NOT a deployable method and never feed stock forecasting.
"""
import os,time
from common import *
os.environ['HF_HOME']=str(ROOT/'work/stock-data/llm-cache')
import mlx.core as mx
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler

def main():
 out=new_run(B/'runs/input_probe_v1');allr={r['id']:r for r in rows(B/'data/pilot_v3/inputs.jsonl')};labels=rows(B/'data/pilot_v3/labels.jsonl');ids=['L03','L04','L05','L08','L09','L10','L14'];labs=[r for r in labels if r['id'] in ids]
 dump(out/'protocol.json',{'ids':ids,'input':'Only assistant-annotated evidence sentences; labels are NOT in the prompt','oracle':True,'outcome_labels':False,'model':MODEL,'revision':REVISION,'created_before_inference':True})
 model,tok=load(MODEL,revision=REVISION);mx.random.seed(573);preds=[]
 for lab in labs:
  r=dict(allr[lab['id']]);keep={s for e in lab['answer']['events'] for s in e['evidence_ids']};r['sentences']={k:v for k,v in r['sentences'].items() if k in keep}
  prompt=tok.apply_chat_template(messages(r),tokenize=False,add_generation_prompt=True,enable_thinking=False);answer=generate(model,tok,prompt=prompt,max_tokens=512,sampler=make_sampler(temp=0),verbose=False);obj=parse(answer);valid,why=validate(obj,r);preds.append({'id':r['id'],'raw':answer,'parsed':obj,'valid':valid,'errors':why});print(r['id'],valid,flush=True)
 write_rows(out/'predictions.jsonl',preds);dump(out/'metrics.json',evaluate(preds,labs))
if __name__=='__main__':main()
