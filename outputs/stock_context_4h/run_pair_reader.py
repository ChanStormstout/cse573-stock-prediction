"""One guarded, frozen, outcome-blind Qwen relation run."""
from __future__ import annotations
import argparse,hashlib,json,os,re,zipfile
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'audit_v2';PRIVATE=ROOT/'work/stock-data/context_increment_4h/audit_v2';MODEL=ROOT/'work/stock-data/model_compare/model';EXEC=ROOT/'outputs/stock_llm_model_compare/EXECUTION.json';PROMPT_PATH=HERE/'RELATION_PROMPT.md'
ENUM=('repeat_same_fact','new_or_changed_fact','explicit_correction_or_denial','background_or_recap','different_actor_object_or_period','unrelated_or_insufficient','unknown')
def split_sentences(body):
    out=[]
    for sentence in [x.strip() for x in re.split(r'(?<=[.!?])\s+|\n+',body or '') if x.strip()]:
        while len(sentence)>1600:
            cut=sentence.rfind(' ',0,1601)
            if cut<=0:cut=1600
            out.append(sentence[:cut].strip());sentence=sentence[cut:].strip()
        if sentence:out.append(sentence)
    return out
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def numerical_surfaces(s): return re.findall(r'(?<![\w.])\$?\d+(?:\.\d+)?(?:\s*(?:%|billion|million|bn|m))?(?![\w.])',str(s),re.I)
def validate(obj,row):
    errors=[]
    if not isinstance(obj,dict):return ['root_not_dict']
    if obj.get('pair_relation') not in ENUM:errors.append('invalid_pair_relation')
    texts={'C':{x['id']:x['text'] for x in row['current_sentences']},'P':{x['id']:x['text'] for x in row['past_sentences']}}; cited=[]
    for prefix,key in [('C','current_evidence_ids'),('P','past_evidence_ids')]:
        values=obj.get(key)
        if not isinstance(values,list) or not all(isinstance(x,str) for x in values):errors.append(key+'_not_list_str');continue
        if len(values)!=len(set(values)):errors.append(key+'_duplicate')
        for x in values:
            if not x.startswith(prefix) or x not in texts[prefix]:errors.append(key+'_bad_id')
            else:cited.append(texts[prefix][x])
    changes=obj.get('changes')
    if not isinstance(changes,list):errors.append('changes_not_list');changes=[]
    evidence=' '.join(cited).casefold()
    for c in changes:
        if not isinstance(c,dict):errors.append('change_not_dict');continue
        for field in ('old_value','new_value'):
            value=c.get(field)
            if value is None:continue
            value=str(value); surfaces=numerical_surfaces(value)
            if surfaces:
                if not all(re.search(r'(?<![\w.])'+re.escape(q)+r'(?![\w.])',evidence,re.I) for q in surfaces):errors.append(field+'_ungrounded_numeric')
            elif value.casefold() not in evidence: errors.append(field+'_ungrounded_text')
    if pd.Timestamp(row['current_available_utc'])<=pd.Timestamp(row['past_available_utc']):errors.append('time_order')
    return sorted(set(errors))
def packets():
    if (PRIVATE/'pair_passages.jsonl').exists():raise FileExistsError('refuse to overwrite packet evidence')
    pairs=pd.read_json(PRIVATE/'news_pairs_private.jsonl',lines=True);ix=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl');lookup={f'{r.archive}::{r.member}':r for r in ix.itertuples(index=False)};zips={};rows=[]
    try:
      for r in pairs.itertuples():
        def article(key,prefix,evidence_id,context_ids):
          x=lookup[key]
          if x.archive not in zips:zips[x.archive]=zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/x.archive)
          obj=json.loads(zips[x.archive].read(x.member));ss=[str(obj.get('title') or x.title or '')]+split_sentences(str(obj.get('text') or ''))
          # Frozen evidence packet: title plus association sentence and its local context.
          wanted=sorted(set([0,int(evidence_id)]+[int(v) for v in context_ids]))
          return [{'id':f'{prefix}{i}','text':ss[i]} for i in wanted if 0<=i<len(ss)]
        rows.append({'pair_id':r.pair_id,'target':r.target,'split':r.split,'stratum':r.stratum,'current_key':r.current_key,'past_key':r.past_key,'current_available_utc':r.current_available_utc,'past_available_utc':r.past_available_utc,'current_sentences':article(r.current_key,'C',r.association_sentence_id_current,r.context_sentence_ids_current),'past_sentences':article(r.past_key,'P',r.association_sentence_id_past,r.context_sentence_ids_past)})
    finally:
      for z in zips.values():z.close()
    with (PRIVATE/'pair_passages.jsonl').open('x') as f:
      for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (OUT/'passage_manifest.json').write_text(json.dumps({'pair_count':len(rows),'pair_ids':[r['pair_id'] for r in rows],'private_sha256':sha(PRIVATE/'pair_passages.jsonl'),'no_raw_text':True},indent=2)+'\n')
    cards=[{'pair_id':r['pair_id'],'target':r['target'],'current_sentences':r['current_sentences'],'past_sentences':r['past_sentences']} for r in rows if r['split']=='locked_check'];(PRIVATE/'locked_review_cards.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cards));(OUT/'locked_review_cards_manifest.json').write_text(json.dumps({'card_count':len(cards),'sha256':sha(PRIVATE/'locked_review_cards.jsonl'),'schema':['pair_id','target','current_sentences','past_sentences']},indent=2)+'\n');return rows
def main():
    args=argparse.ArgumentParser();args.add_argument('--preflight',action='store_true');opt=args.parse_args()
    if (PRIVATE/'qwen_relations.jsonl').exists() or (OUT/'relation_summary.json').exists():raise FileExistsError('one-run guard: relation output exists')
    if not PROMPT_PATH.exists():raise FileNotFoundError('frozen prompt missing')
    prompt=PROMPT_PATH.read_text();assert all(x in prompt for x in ENUM)
    rows=packets();ex=json.loads(EXEC.read_text());actual={p.name:sha(p) for p in MODEL.glob('*') if p.is_file()};good=actual==ex['model_files'];(OUT/'QWEN_PROVENANCE.json').write_text(json.dumps({'status':'VERIFIED_EXISTING_FROZEN_QWEN_AVAILABLE' if good else 'HASH_MISMATCH_STOPPED','revision':ex['revision'],'files_expected':len(ex['model_files']),'files_matching':sum(actual.get(k)==v for k,v in ex['model_files'].items())},indent=2)+'\n')
    if not good:raise RuntimeError('model hashes changed')
    os.environ.update({'HF_HOME':str(ROOT/'work/stock-data/model_compare/hf-home'),'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx
    model,tok=load(str(MODEL));model.eval();mx.random.seed(573); rendered=[]
    for r in rows:
      content='TARGET='+r['target']+'\nCURRENT\n'+'\n'.join(x['id']+': '+x['text'] for x in r['current_sentences'])+'\nPAST\n'+'\n'.join(x['id']+': '+x['text'] for x in r['past_sentences']);messages=[{'role':'system','content':prompt},{'role':'user','content':content}];text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False);rendered.append((r,messages,tok.encode(text,add_special_tokens=False)))
    audit={'pair_count':len(rows),'max_prompt_tokens':max(len(x[2]) for x in rendered),'over_limit_count':sum(len(x[2])+512>4096 for x in rendered)};(OUT/'token_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    if audit['over_limit_count']:raise RuntimeError('pre-inference token audit failed')
    if opt.preflight:
        (OUT/'preflight.json').write_text(json.dumps({'status':'PASS','target_association':'PASS','family_isolation':'PASS','cross_split_family_count':0,'qwen_hashes':'PASS','token_audit':audit},indent=2)+'\n')
        return
    outputs=[]
    with (PRIVATE/'qwen_relations.jsonl').open('x') as f:
      for n,(row,msg,ids) in enumerate(rendered,1):
        last=None;raw=''
        for event in stream_generate(model,tok,prompt=ids,max_tokens=512,sampler=make_sampler(temp=0)):raw+=event.text;last=event
        m=re.search(r'\{.*\}',raw,re.S)
        try:parsed=json.loads(m.group(0) if m else raw)
        except Exception:parsed=None
        rec={'pair_id':row['pair_id'],'raw_output':raw,'parsed_output':parsed,'validation_errors':validate(parsed,row),'prompt_sha256':sha(PROMPT_PATH),'prompt_token_count':len(ids),'output_token_count':getattr(last,'generation_tokens',None),'finish_reason':getattr(last,'finish_reason',None)};outputs.append(rec);f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();print(n,len(rendered),flush=True);mx.clear_cache()
    ids=[x['pair_id'] for x in outputs]; assert len(ids)==len(set(ids)) and set(ids)=={x['pair_id'] for x in rows}
    validity=pd.DataFrame([{'pair_id':x['pair_id'],'valid':not x['validation_errors'],'errors':'|'.join(x['validation_errors'])} for x in outputs]);validity.to_csv(OUT/'relation_output_validity.csv',index=False);relations=pd.DataFrame([{'pair_id':x['pair_id'],'relation':(x['parsed_output'] or {}).get('pair_relation','INVALID')} for x in outputs]).merge(pd.DataFrame(rows)[['pair_id','target','stratum','split']],on='pair_id');counts=relations.groupby(['target','stratum','split','relation']).size().reset_index(name='count').to_dict('records');(OUT/'relation_summary.json').write_text(json.dumps({'status':'PROVISIONAL_MODEL_RELATIONS_NOT_GOLD','calls':len(outputs),'valid_count':int(validity.valid.sum()),'counts':counts,'prompt_sha256':sha(PROMPT_PATH)},indent=2)+'\n');(OUT/'PAIR_READER_STATUS.json').write_text(json.dumps({'status':'PROVISIONAL_MODEL_RELATIONS_NOT_GOLD','llm_calls':len(outputs),'mechanical_validation_pass_rate':float(validity.valid.mean())},indent=2)+'\n')
if __name__=='__main__':main()
