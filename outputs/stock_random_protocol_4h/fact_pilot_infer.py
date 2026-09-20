"""Frozen local pair-reading pilot. Mechanical validity is NOT semantic gold."""
import json,re,time,os,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'v1';PRIVATE=ROOT/'work/stock-data/random_protocol_4h/v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def validate(answer,card):
 errors=[]
 if not isinstance(answer,dict):return ['not_object']
 fields={'current_fact','prior_fact','relation','current_evidence','prior_evidence'}
 if set(answer)!=fields:errors.append('schema_fields')
 if answer.get('relation') not in {'repeat','further_change','correction_or_denial','different_actor','background','unknown'}:errors.append('relation')
 for which in ['current','prior']:
  fact=answer.get(which+'_fact');evidence=answer.get(which+'_evidence')
  if fact is None:
   if evidence is not None and (not isinstance(evidence,str) or evidence not in card[which+'_passage']):errors.append(which+'_evidence')
   continue
  required={'target','kind','action','actor','old','new','unit','current_event'}
  if not isinstance(fact,dict) or set(fact)!=required:errors.append(which+'_fact_schema');continue
  if fact['target']!=card['symbol']:errors.append(which+'_target')
  if fact['kind'] not in {'rating','target_price'}:errors.append(which+'_type')
  if fact['action'] not in {'raise','lower','maintain','initiate','unknown'}:errors.append(which+'_action')
  if not isinstance(evidence,str) or len(evidence)<8 or evidence not in card[which+'_passage']:errors.append(which+'_unsupported_quote')
  for key in ['old','new']:
   val=fact[key]
   if val is not None:
    if type(val) not in (int,float):errors.append(which+'_'+key+'_type')
    else:
     numbers=[float(x.replace(',','')) for x in re.findall(r'\d[\d,]*(?:\.\d+)?',evidence or '')]
     if not any(abs(n-val)<1e-9 for n in numbers):errors.append(which+'_'+key+'_unsupported_number')
  actor=fact['actor']
  if actor is not None and (not isinstance(actor,str) or actor.lower() not in (evidence or '').lower()):errors.append(which+'_actor_not_in_evidence')
  if fact['current_event'] is not None and type(fact['current_event']) is not bool:errors.append(which+'_current_event_type')
 return errors

def main():
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
 src=PRIVATE/'fact_pilot';manifest=json.loads((src/'manifest.json').read_text());assert sha(src/'cards.jsonl')==manifest['cards']
 modelpath=ROOT/'work/stock-data/model_compare/model';pin=json.loads((ROOT/'work/stock-data/model_compare/comparison_v1/manifest.json').read_text())
 for n,h in pin['model_files'].items():assert sha(modelpath/n)==h
 seal={'manifest':sha(src/'manifest.json'),'code':sha(__file__),'model_revision':pin['revision'],'model_files':pin['model_files'],'max_tokens':512,'temperature':0,'thinking':False,'max_input_tokens':6000}
 if (src/'inference_seal.json').exists():assert json.loads((src/'inference_seal.json').read_text())==seal
 else:dump(src/'inference_seal.json',seal)
 cards=list(map(json.loads,(src/'cards.jsonl').read_text().splitlines()));dest=src/'outputs.jsonl';old=list(map(json.loads,dest.read_text().splitlines())) if dest.exists() else []
 assert [r['pair_id'] for r in old]==[c['pair_id'] for c in cards[:len(old)]]
 if len(old)<len(cards):
  import mlx.core as mx
  from mlx_lm import load,generate
  from mlx_lm.sample_utils import make_sampler
  model,tok=load(str(modelpath));model.eval();mx.random.seed(573)
  with dest.open('a') as f:
   for card in cards[len(old):]:
    prompt=tok.apply_chat_template(card['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert len(tok.encode(prompt,add_special_tokens=False))<=6000,'no silent truncation'
    start=time.monotonic();text=generate(model,tok,prompt=prompt,max_tokens=512,sampler=make_sampler(temp=0),verbose=False)
    stripped=text.strip();stripped=re.sub(r'^```(?:json)?\s*|\s*```$','',stripped)
    try:answer=json.loads(stripped);errors=validate(answer,card)
    except (ValueError,TypeError):answer=None;errors=['unparseable_json']
    row=dict(pair_id=card['pair_id'],output=text,parsed=answer,mechanical_errors=errors,seconds=time.monotonic()-start)
    f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();mx.clear_cache();print(card['pair_id'],'mechanical errors',errors,flush=True)
 rows=list(map(json.loads,dest.read_text().splitlines()));valid=sum(not r['mechanical_errors'] for r in rows)
 result=dict(status='PILOT_COMPLETE_PENDING_ASSISTANT_AND_INDEPENDENT_SEMANTIC_REVIEW',calls=len(rows),mechanically_valid=valid,mechanically_valid_fraction=valid/len(rows) if rows else None,semantic_precision=None,independent_gold=False,accepted_for_prediction=False,output_hash=sha(dest),seconds=sum(r['seconds'] for r in rows),model_revision=pin['revision'])
 dump(OUT/'FACT_PILOT_EXECUTION.json',result);dump(src/'complete.json',result);print(json.dumps(result))
if __name__=='__main__':main()
