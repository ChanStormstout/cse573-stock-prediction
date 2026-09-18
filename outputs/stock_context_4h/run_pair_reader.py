"""One frozen, outcome-blind Qwen relation pass; private text/output only."""
from __future__ import annotations
import hashlib,json,os,re,zipfile
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'audit_v2';PRIVATE=ROOT/'work/stock-data/context_increment_4h/audit_v2';MODEL=ROOT/'work/stock-data/model_compare/model';EXEC=ROOT/'outputs/stock_llm_model_compare/EXECUTION.json'
PROMPT='''Compare only supplied evidence for TARGET. Never predict direction, reaction, importance, priced-in status, or first disclosure. Do not merge different actors or periods. A maintained rating and target-price change are separate facts. Numeric old/new values must occur literally in cited evidence. If insufficient, unknown. Output JSON only: {"pair_relation":"repeat_same_fact|new_or_changed_fact|explicit_correction_or_denial|background_or_recap|different_actor_object_or_period|unrelated_or_insufficient|unknown","current_evidence_ids":[],"past_evidence_ids":[],"changes":[{"actor":null,"object":null,"old_value":null,"new_value":null,"unit":null,"period":null}]}'''
ENUM=set(PROMPT.split('"pair_relation":"')[1].split('"')[0].split('|'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def packets():
 d=pd.read_json(PRIVATE/'news_pairs_private.jsonl',lines=True);ix=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl');look={f'{r.archive}::{r.member}':r for r in ix.itertuples(index=False)};zs={};out=[]
 for r in d.itertuples():
  def art(k,prefix):
   x=look[k];z=zs.setdefault(x.archive,zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/x.archive));o=json.loads(z.read(x.member));s=[str(o.get('title') or x.title or '')]+[q.strip() for q in re.split(r'(?<=[.!?])\s+|\n+',str(o.get('text') or '')) if q.strip()];return [{'id':f'{prefix}{i}','text':q} for i,q in enumerate(s)]
  out.append({'pair_id':r.pair_id,'target':r.target,'split':r.split,'stratum':r.stratum,'family_id':r.family_id,'current_key':r.current_key,'past_key':r.past_key,'current_available_utc':r.current_available_utc,'past_available_utc':r.past_available_utc,'current_sentences':art(r.current_key,'C'),'past_sentences':art(r.past_key,'P')})
 with (PRIVATE/'pair_passages.jsonl').open('w') as f:
  for x in out:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 (OUT/'passage_manifest.json').write_text(json.dumps({'row_count':len(out),'pair_ids':[x['pair_id'] for x in out],'private_sha256':sha(PRIVATE/'pair_passages.jsonl'),'no_raw_text':True},indent=2)+'\n')
 cards=[{'pair_id':x['pair_id'],'target':x['target'],'current_sentences':x['current_sentences'],'past_sentences':x['past_sentences']} for x in out if x['split']=='locked_check'];(PRIVATE/'locked_review_cards.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in cards));(OUT/'locked_review_cards_manifest.json').write_text(json.dumps({'card_count':len(cards),'sha256':sha(PRIVATE/'locked_review_cards.jsonl'),'schema':['pair_id','target','current_sentences','past_sentences']},indent=2)+'\n');return out
def valid(o,row):
 e=[];c={x['id']:x['text'] for x in row['current_sentences']};p={x['id']:x['text'] for x in row['past_sentences']}
 if not isinstance(o,dict) or o.get('pair_relation') not in ENUM:e+=['enum']
 cited=[]
 for k,d in [('current_evidence_ids',c),('past_evidence_ids',p)]:
  if not isinstance(o.get(k),list) or any(x not in d for x in o.get(k,[])):e+=[k]
  else:cited += [d[x] for x in o[k]]
 for ch in o.get('changes',[]):
  for k in ('old_value','new_value'):
   if ch.get(k) is not None and str(ch[k]) not in ' '.join(cited):e+=['numeric_literal']
 return e
def main():
 rows=packets();ex=json.loads(EXEC.read_text());actual={p.name:sha(p) for p in MODEL.glob('*') if p.is_file()};good=actual==ex['model_files'];(OUT/'QWEN_PROVENANCE.json').write_text(json.dumps({'status':'VERIFIED_EXISTING_FROZEN_QWEN_AVAILABLE' if good else 'HASH_MISMATCH_STOPPED','revision':ex['revision'],'files_matching':sum(actual.get(k)==v for k,v in ex['model_files'].items())},indent=2)+'\n')
 if not good:raise RuntimeError('model hashes changed')
 (HERE/'RELATION_PROMPT.md').write_text(PROMPT+'\n');ph=sha(HERE/'RELATION_PROMPT.md');os.environ.update({'HF_HOME':str(ROOT/'work/stock-data/model_compare/hf-home'),'HF_HUB_OFFLINE':'1'})
 import mlx.core as mx
 from mlx_lm import load,stream_generate
 from mlx_lm.sample_utils import make_sampler
 model,tok=load(str(MODEL));model.eval();mx.random.seed(573);outs=[]
 with (PRIVATE/'qwen_relations.jsonl').open('w') as f:
  for n,r in enumerate(rows,1):
   content='TARGET='+r['target']+'\nCURRENT\n'+'\n'.join(x['id']+': '+x['text'] for x in r['current_sentences'])+'\nPAST\n'+'\n'.join(x['id']+': '+x['text'] for x in r['past_sentences']);msg=[{'role':'system','content':PROMPT},{'role':'user','content':content}];t=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(t,add_special_tokens=False)
   if len(ids)+512>4096:raise RuntimeError('refuse truncation')
   raw=''.join(z.text for z in stream_generate(model,tok,prompt=ids,max_tokens=512,sampler=make_sampler(temp=0)));m=re.search(r'\{.*\}',raw,re.S)
   try:o=json.loads(m.group(0) if m else raw)
   except Exception:o={}
   q={'pair_id':r['pair_id'],'parsed':o,'errors':valid(o,r),'prompt_sha256':ph};outs.append(q);f.write(json.dumps(q)+'\n');f.flush();print(n,len(rows),flush=True);mx.clear_cache()
 v=pd.DataFrame([{'pair_id':x['pair_id'],'valid':not x['errors'],'errors':'|'.join(x['errors'])} for x in outs]);v.to_csv(OUT/'relation_output_validity.csv',index=False);j=pd.DataFrame([{'pair_id':x['pair_id'],'relation':x['parsed'].get('pair_relation','INVALID')} for x in outs]).merge(pd.DataFrame(rows)[['pair_id','target','stratum','split']],on='pair_id');(OUT/'relation_summary.json').write_text(json.dumps({'status':'PROVISIONAL_MODEL_RELATIONS_NOT_GOLD','calls':len(outs),'valid_count':int(v.valid.sum()),'counts':j.groupby(['target','stratum','split','relation']).size().to_dict(),'prompt_sha256':ph},indent=2,default=str)+'\n');(OUT/'PAIR_READER_STATUS.json').write_text(json.dumps({'status':'PROVISIONAL_MODEL_RELATIONS_NOT_GOLD','llm_calls':len(outs),'mechanical_validation_pass_rate':float(v.valid.mean())},indent=2)+'\n')
if __name__=='__main__':main()
