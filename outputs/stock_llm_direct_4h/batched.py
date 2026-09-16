"""Native batch-of-four inference; same prompts and generation budget."""
import argparse,json,os,shutil,time
from pathlib import Path
from common import B,W,VARIANTS,messages,parse_forecast,validate_boundary,rows,sha,dump
os.environ['HF_HOME']=str(W/'model_compare/hf-home');os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',action='store_true');ap.add_argument('--benchmark',action='store_true');ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
 seal=json.loads((a.inputs/'manifest.json').read_text());assert sha(a.inputs/'inputs.jsonl')==seal['input_sha'];data=[r for r in rows(a.inputs/'inputs.jsonl') if r['interval_start']>='2018-09-01'];assert len(data)==609
 if a.benchmark:data=data[:8]
 variants=['price'] if a.benchmark else list(VARIANTS)
 model_dir=W/'model_compare/model';pinned=json.loads((W/'model_compare/comparison_v1/manifest.json').read_text())
 for f,h in pinned['model_files'].items():assert sha(model_dir/f)==h
 manifest=dict(model=pinned['model'],revision=pinned['revision'],model_files=pinned['model_files'],input_sha=seal['input_sha'],keys=[r['key'] for r in data],variants=variants,batch_size=4,temperature=0,seed=573,thinking=False,max_output=192,max_input_plus_output=6144,training=False,smoke=a.benchmark,benchmark=a.benchmark,code={f:sha(B/f) for f in ['run.py','batched.py','common.py','PROTOCOL.md','BATCH_ENGINE.md']})
 if a.out.exists():
  if not a.resume:raise FileExistsError(a.out)
  if json.loads((a.out/'manifest.json').read_text())!=manifest:raise ValueError('Resume fingerprint mismatch')
  if (a.out/'summary.json').exists():raise ValueError('Already complete')
 else:
  a.out.mkdir(parents=True);dump(a.out/'manifest.json',manifest);(a.out/'code').mkdir()
  for f in manifest['code']:shutil.copy2(B/f,a.out/'code'/f)
 import mlx.core as mx
 from mlx_lm import load
 from mlx_lm.generate import BatchGenerator
 from mlx_lm.sample_utils import make_sampler
 model,tok=load(str(model_dir));model.eval();mx.random.seed(573);prepared={};maximum={}
 for v in variants:
  prepared[v]=[]
  for r in data:
   validate_boundary(r);msgs=messages(r,v);text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False);tokens=tok.encode(text,add_special_tokens=False)
   if len(tokens)+192>6144:raise ValueError('Refuse truncation')
   prepared[v].append((msgs,tokens))
  maximum[v]=max(len(t) for _,t in prepared[v])
 dump(a.out/'input_check.json',dict(n=len(data),max_prompt_tokens=maximum,no_truncation=True,no_labels_loaded=True));print('BATCH LOADED',maximum,flush=True)
 summary={};start=time.time()
 for v in variants:
  path=a.out/f'{v}.jsonl';done=rows(path) if path.exists() else []
  if [r['key'] for r in done]!=[r['key'] for r in data[:len(done)]]:raise ValueError('Resume prefix mismatch')
  with path.open('a') as log:
   for offset in range(len(done),len(data),4):
    rr=data[offset:offset+4];pack=prepared[v][offset:offset+4];began=time.time()
    gen=BatchGenerator(model,stop_tokens=[[t] for t in tok.eos_token_ids],sampler=make_sampler(temp=0),completion_batch_size=4,prefill_batch_size=4,prefill_step_size=512)
    uids=gen.insert([t for _,t in pack],[192]*len(pack));tokens_out={u:[] for u in uids};fin={};steps={u:0 for u in uids}
    try:
     while responses:=gen.next_generated():
      for r in responses:
       steps[r.uid]+=1
       if r.finish_reason!='stop':tokens_out[r.uid].append(r.token)
       if r.finish_reason is not None:fin[r.uid]=r.finish_reason
    finally:gen.close()
    elapsed=time.time()-began
    for r,(msgs,prompt),u in zip(rr,pack,uids):
     raw=tok.decode(tokens_out[u]);x=dict(key=r['key'],variant=v,**parse_forecast(raw,r,v),raw=raw,messages=msgs,seconds=elapsed/len(rr),batch_seconds=elapsed,batch_first_key=rr[0]['key'],prompt_tokens=len(prompt),output_tokens=steps[u],finish_reason=fin[u],peak_mlx_gb=mx.get_peak_memory()/1e9)
     log.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');log.flush();done.append(x)
    mx.clear_cache()
    if len(done)%20==0 or len(done)==len(data):print(v,len(done),len(data),'valid',sum(r['valid'] for r in done),'sec',round(sum(r['seconds'] for r in done),1),flush=True)
  summary[v]=dict(n=len(done),valid=sum(r['valid'] for r in done),seconds=sum(r['seconds'] for r in done),peak_mlx_gb=max(r['peak_mlx_gb'] for r in done),output_tokens=sum(r['output_tokens'] for r in done),length_stops=sum(r['finish_reason']=='length' for r in done));dump(a.out/'progress.json',summary)
 if a.benchmark:
  original=rows(W/'direct_4h/run_v1/price.jsonl')[:8];new=rows(a.out/'price.jsonl');assert len(original)==len(new)==8
  checks=[]
  for old,newer in zip(original,new):
   assert old['key']==newer['key'];checks.append(dict(key=old['key'],raw_equal=old['raw']==newer['raw'],probability_equal=old['p']==newer['p'],validity_equal=old['valid']==newer['valid'],direction_equal=(old['parsed'] or {}).get('direction')==(newer['parsed'] or {}).get('direction')))
  dump(a.out/'parity.json',dict(checks=checks,passed=all(c['probability_equal'] and c['validity_equal'] and c['direction_equal'] for c in checks),sequential_seconds=sum(r['seconds'] for r in original),batch_seconds=sum(r['seconds'] for r in rows(a.out/'price.jsonl'))))
 dump(a.out/'summary.json',dict(completed=True,variants=summary,training=False,process_seconds=time.time()-start))
if __name__=='__main__':main()
