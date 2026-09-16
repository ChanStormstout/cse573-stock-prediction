"""Matched frozen/adapter fact inference, no price-label access."""
import os,time,argparse
from common import *
os.environ['HF_HOME']=str(ROOT/'work/stock-data/llm-cache')
import mlx.core as mx
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--adapter',type=Path);a.add_argument('--data',type=Path,default=B/'data/pilot_v3');a.add_argument('--split',default='all');a.add_argument('--expanded-contract',action='store_true');a.add_argument('--max-length',type=int,default=2048);args=a.parse_args();out=new_run(args.out)
 if args.expanded_contract:
  import sys
  sys.path.insert(0,str(ROOT/'outputs/stock_llm_annotation'))
  import model_contract
  globals()['messages']=model_contract.messages
  globals()['SYSTEM']=model_contract.SYSTEM
 if args.max_length not in (2048,3072):raise ValueError('Unregistered token budget')
 check_panel(args.data)
 rs=rows(args.data/'inputs.jsonl');labels=rows(args.data/'labels.jsonl');ls={r['id']:r for r in labels}
 if args.split!='all':
  splits=set(args.split.split(','))
  if not splits<={'train','valid','check','diagnostic'}:raise ValueError('Unknown split')
  rs=[r for r in rs if ls[r['id']]['split'] in splits]
 config={'model':MODEL,'revision':REVISION,'adapter':str(args.adapter) if args.adapter else None,'input_sha256':file_sha(args.data/'inputs.jsonl'),'labels_sha256':file_sha(args.data/'labels.jsonl'),'system_sha256':digest(SYSTEM.encode()),'code_sha256':file_sha(__file__),'common_sha256':file_sha(B/'common.py'),'seed':573,'temperature':0,'max_tokens':512,'thinking':False,'quality':'NOT_INDEPENDENT','data_split':args.split,'expanded_contract':args.expanded_contract,'max_length':args.max_length}
 if args.adapter:config['adapter_sha256']=file_sha(args.adapter/'adapters.safetensors')
 dump(out/'manifest.json',config);model,tok=load(MODEL,revision=REVISION,adapter_path=str(args.adapter) if args.adapter else None);mx.random.seed(573);preds=[];start=time.time()
 for r in rs:
  prompt=tok.apply_chat_template(messages(r),tokenize=False,add_generation_prompt=True,enable_thinking=False)
  tokens=tok.encode(prompt,add_special_tokens=False)
  if len(tokens)+512>args.max_length:raise ValueError(f"Input budget exceeded {r['id']}: {len(tokens)}")
  t=time.time();answer=generate(model,tok,prompt=prompt,max_tokens=512,sampler=make_sampler(temp=0),verbose=False);obj=parse(answer);valid,errors=validate(obj,r)
  preds.append({'id':r['id'],'raw':answer,'parsed':obj,'valid':valid,'errors':errors,'seconds':time.time()-t,'prompt_tokens':len(tokens),'output_tokens':len(tok.encode(answer,add_special_tokens=False))});write_rows(out/'predictions.jsonl',preds)
  print(r['id'],'valid',valid,'seconds',round(time.time()-t,2),'events',len(obj.get('events',[])) if isinstance(obj,dict) else None,flush=True)
 dump(out/'metrics.json',evaluate(preds,labels));dump(out/'cost.json',{'seconds':time.time()-start,'peak_memory_gb':mx.get_peak_memory()/1e9,'n':len(preds)});print((out/'metrics.json').read_text(),flush=True)
if __name__=='__main__':main()
