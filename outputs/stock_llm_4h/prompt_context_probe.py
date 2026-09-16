"""User-requested 2x2 prompt/context diagnostic on the fixed eight development inputs."""
import os,time
from common import *
os.environ['HF_HOME']=str(ROOT/'work/stock-data/llm-cache')
import mlx.core as mx
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler
SHORT='''Read the numbered news sentences about TARGET. Extract current actions by the PRIMARY analyst only. Ignore other companies, old analyst lists, share holdings and product upgrades. A static Buy rating is not an upgrade. A reported price target without an explicit change has action unknown. Keep rating changes and price-target changes as separate events. If no supported event, return {"events":[]}.
Reply with one JSON object and no other text. Format:
{"events":[{"kind":"rating","action":"maintain","old":null,"new":"Buy","unit":"rating","evidence_ids":["S1"]}]}
kind: rating or target_price. action: raise, lower, maintain, initiate, unknown. unit: rating or USD. old/new: exact strings from evidence, or null if absent. For target_price copy the number without $. Cite only sentences supporting the current target-company action. Never invent values or predict stock prices.'''

def main():
 out=new_run(B/'runs/prompt_context_v1');labs=[r for r in rows(B/'data/pilot_v3/labels.jsonl') if r['split']=='valid'];ids={r['id'] for r in labs}
 dump(out/'protocol.json',{'scope':'Eight already-exposed DEVELOPMENT cases only; no check-driven prompt selection','factors':{'context':['selected_context','current_context'],'prompt':['original_with_three_examples','short_without_chat_examples']},'same_model':MODEL,'revision':REVISION,'same_schema':True,'short_prompt':SHORT,'temperature':0,'max_tokens':512,'thinking':False,'ids':sorted(ids),'reuse_original_cells':'Previously recorded frozen_v1 and frozen_current_v1 outputs; identical inputs/model/decoding. Prompt reduction includes removing examples, not an isolated token-length causal claim.'})
 cells={};model,tok=load(MODEL,revision=REVISION)
 for context,data,oldrun in [('selected_context','pilot_v3','frozen_v1'),('current_context','pilot_current_v1','frozen_current_v1')]:
  old=[r for r in rows(B/'runs'/oldrun/'predictions.jsonl') if r['id'] in ids];cells[context+'|original']=evaluate(old,labs)['valid']
  inputs=[r for r in rows(B/'data'/data/'inputs.jsonl') if r['id'] in ids];preds=[];mx.random.seed(573)
  for r in inputs:
   user='TARGET: '+r['symbol']+'\n'+'\n'.join(k+': '+v for k,v in r['sentences'].items())
   m=[{'role':'system','content':SHORT},{'role':'user','content':user}];prompt=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   tick=time.time();answer=generate(model,tok,prompt=prompt,max_tokens=512,sampler=make_sampler(temp=0),verbose=False);obj=parse(answer);ok,why=validate(obj,r);preds.append({'id':r['id'],'raw':answer,'parsed':obj,'valid':ok,'errors':why,'seconds':time.time()-tick,'prompt_tokens':len(tok.encode(prompt,add_special_tokens=False))});print(context,r['id'],ok,flush=True)
  write_rows(out/(context+'_short_predictions.jsonl'),preds);cells[context+'|short']=evaluate(preds,labs)['valid']
 dump(out/'metrics.json',cells);print(json.dumps(cells,indent=2),flush=True)
if __name__=='__main__':main()
