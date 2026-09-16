"""Offline frozen direct forecasts; no labels loaded by inference process."""
import argparse,json,os,shutil,time
from pathlib import Path
from common import B,W,VARIANTS,messages,parse_forecast,validate_boundary,rows,sha,dump
os.environ['HF_HOME']=str(W/'model_compare/hf-home');os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
 seal=json.loads((a.inputs/'manifest.json').read_text());assert sha(a.inputs/'inputs.jsonl')==seal['input_sha']
 data=rows(a.inputs/'inputs.jsonl');data=[r for r in data if r['interval_start']>='2018-09-01'];assert len(data)==609
 if a.smoke:
  # Interface-only synthetic facts, no actual stock outcome or news.
  import copy
  r=copy.deepcopy(data[0]);r['key']='SYNTHETIC';r['symbol']='EXAMPLE';r['news']=[];r['news_summary']={'eligible_articles':0,'included_articles':0,'omitted_articles':0}
  for z in r['price_rows']:z.update(log_return_pct=0.01,range_pct=.2)
  r['price_summary'].update(mean_log_return_pct=.01,std_log_return_pct=0.)
  data=[r]
 model_dir=W/'model_compare/model'
 # Hash manifest from already pinned weights; recheck files before loading.
 pinned=json.loads((W/'model_compare/comparison_v1/manifest.json').read_text())
 for name,h in pinned['model_files'].items():assert sha(model_dir/name)==h
 fixed={'model':pinned['model'],'revision':pinned['revision'],'model_files':pinned['model_files'],'input_sha':seal['input_sha'],'code':{f:sha(B/f) for f in ['run.py','common.py','PROTOCOL.md']},'keys':[r['key'] for r in data],'variants':VARIANTS,'max_input_plus_output':6144,'max_output':192,'temperature':0,'seed':573,'thinking':False,'training':False,'smoke':a.smoke}
 # JSON roundtrip canonicalizes tuples for resume equality.
 fixed=json.loads(json.dumps(fixed))
 if a.out.exists():
  if not a.resume:raise FileExistsError(a.out)
  if json.loads((a.out/'manifest.json').read_text())!=fixed:raise ValueError('Resume fingerprint mismatch')
  if (a.out/'summary.json').exists():raise ValueError('Already complete')
 else:
  a.out.mkdir(parents=True);dump(a.out/'manifest.json',fixed);(a.out/'code').mkdir()
  for name in fixed['code']:shutil.copy2(B/name,a.out/'code'/name)
 import mlx.core as mx
 from mlx_lm import load,stream_generate
 from mlx_lm.sample_utils import make_sampler
 start=time.time();model,tok=load(str(model_dir));model.eval();mx.random.seed(573)
 # Validate ALL prompt lengths before any market call. No silent truncation.
 maximum={}
 for v in VARIANTS:
  lens=[]
  for r in data:
   validate_boundary(r);text=tok.apply_chat_template(messages(r,v),tokenize=False,add_generation_prompt=True,enable_thinking=False);lens.append(len(tok.encode(text,add_special_tokens=False)))
  maximum[v]=max(lens)
  if maximum[v]+192>6144:raise ValueError(f'Prompt budget exceeded {v} {maximum[v]}')
 dump(a.out/'input_check.json',{'n':len(data),'max_prompt_tokens':maximum,'no_truncation':True,'no_labels_loaded':True});print('LOADED',maximum,flush=True)
 summary={}
 for v in VARIANTS:
  path=a.out/f'{v}.jsonl';done=rows(path) if path.exists() else []
  if [x['key'] for x in done]!=[x['key'] for x in data[:len(done)]]:raise ValueError('Corrupt resume prefix')
  with path.open('a') as f:
   for i,r in enumerate(data[len(done):],len(done)+1):
    msgs=messages(r,v);text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False);tokens=tok.encode(text,add_special_tokens=False)
    began=time.time();parts=[];last=None
    for response in stream_generate(model,tok,prompt=tokens,max_tokens=192,sampler=make_sampler(temp=0)):
     parts.append(response.text);last=response
    raw=''.join(parts);x=dict(key=r['key'],variant=v,**parse_forecast(raw,r,v),raw=raw,messages=msgs,seconds=time.time()-began,prompt_tokens=len(tokens),output_tokens=last.generation_tokens,finish_reason=last.finish_reason,peak_mlx_gb=mx.get_peak_memory()/1e9)
    f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n');f.flush();done.append(x);mx.clear_cache()
    if i%10==0 or i==len(data):print(v,i,len(data),'valid',sum(x['valid'] for x in done),'seconds',round(sum(x['seconds'] for x in done),1),flush=True)
  summary[v]={'n':len(done),'valid':sum(x['valid'] for x in done),'seconds':sum(x['seconds'] for x in done),'peak_mlx_gb':max(x['peak_mlx_gb'] for x in done),'output_tokens':sum(x['output_tokens'] for x in done),'length_stops':sum(x['finish_reason']=='length' for x in done)}
  dump(a.out/'progress.json',summary)
 dump(a.out/'summary.json',dict(completed=True,variants=summary,training=False,process_seconds=time.time()-start))
if __name__=='__main__':
 import sys
 if '--choice' in sys.argv:
  from choice import main as choice_main
  choice_main()
 elif '--batch' in sys.argv:
  from batched import main as batch_main
  batch_main()
 else:main()
