"""Pinned local frozen inference. Never fits, changes labels, or overwrites runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from contracts import ROOT, messages, validate_gate, compile_items
sys.path.insert(0,str(ROOT/'outputs/stock_llm_4h'))
from common import rows, dump, file_sha, check_panel, parse, validate, evaluate

BASE=ROOT/'work/stock-data/model_compare'
os.environ['HF_HOME']=str(BASE/'hf-home')
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'work/stock-data/annotation/student_v3';check_panel(data)
    inputs=rows(data/'inputs.jsonl');labels=rows(data/'labels.jsonl')
    eligible={r['id'] for r in labels if r['split'] in {'valid','check'}}
    inputs=sorted([r for r in inputs if r['id'] in eligible],key=lambda r:(r['split']!='valid',r['id']))
    if args.smoke:
        inputs=[dict(id='SYNTHETIC',symbol='AMZN',sentences={'S0':'Firm K reiterated Buy on Amazon and cut its price target from $240 to $225.'})]
    model_dir=BASE/'model'
    manifest={'model':'mlx-community/Qwen3.5-9B-4bit','revision':'8b2b98c00a6b4d291155e4890773ca8f769aee53',
              'variants':['original','prompt','staged'],'seed':573,'temperature':0,'thinking':False,
              'max_output_per_call':512,'max_total_per_call':4096,'training':False,'smoke_only':args.smoke,
              'input_sha256':file_sha(data/'inputs.jsonl'),'labels_sha256':file_sha(data/'labels.jsonl'),
              'code':{f:file_sha(Path(__file__).parent/f) for f in ['run.py','contracts.py','PROTOCOL.md','SMOKE_AMENDMENT.md']},
              'model_files':{f.name:file_sha(f) for f in sorted(model_dir.glob('*')) if f.is_file()},
              'ids':[r['id'] for r in inputs],'evaluation_status':'EXPOSED_REGRESSION_MODEL_PROVISIONAL'}
    dump(args.out/'manifest.json',manifest)
    (args.out/'code').mkdir()
    for f in manifest['code']:shutil.copy2(Path(__file__).parent/f,args.out/'code'/f)
    import mlx.core as mx
    from mlx_lm import load,stream_generate
    from mlx_lm.sample_utils import make_sampler
    start=time.time();model,tok=load(str(model_dir));model.eval();mx.random.seed(573)
    print('LOADED',round(time.time()-start,1),'seconds',flush=True)
    def call(msgs):
        text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tokens=tok.encode(text,add_special_tokens=False)
        if len(tokens)+512>4096:raise ValueError('Refuse truncation')
        began=time.time();parts=[];last=None
        for response in stream_generate(model,tok,prompt=tokens,max_tokens=512,sampler=make_sampler(temp=0)):
            parts.append(response.text);last=response
        answer=''.join(parts)
        return {'messages':msgs,'raw':answer,'parsed':parse(answer),'seconds':time.time()-began,
                'prompt_tokens':len(tokens),'output_tokens':last.generation_tokens,
                'finish_reason':last.finish_reason,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'tokens_per_second':last.generation_tps}
    summary={}
    for variant in manifest['variants']:
        out=args.out/variant;out.mkdir();preds=[];begin=time.time()
        with (out/'predictions.jsonl').open('x') as log:
            for n,row in enumerate(inputs,1):
                calls=[];rejected=[];component_errors=[]
                if variant!='staged':
                    c=call(messages(row,variant));calls.append(c);obj=c['parsed']
                else:
                    c=call(messages(row,'gate'));calls.append(c)
                    try:
                        ids=validate_gate(c['parsed'],row)
                        if ids:
                            c=call(messages(row,'actions',ids));calls.append(c)
                            obj,rejected=compile_items(c['parsed'],row,ids)
                        else:obj={'events':[]}
                    except (ValueError,TypeError) as exc:
                        component_errors.append(str(exc));obj=None
                valid,errors=validate(obj,row)
                pred={'id':row['id'],'parsed':obj,'valid':valid,'errors':errors,
                      'calls':calls,'rejected':rejected,'component_errors':component_errors,
                      'seconds':sum(c['seconds'] for c in calls)}
                preds.append(pred);log.write(json.dumps(pred,ensure_ascii=False)+'\n');log.flush()
                print(variant,n,len(inputs),row['id'],'valid',valid,'events',len(obj.get('events',[])) if obj else -1,
                      'sec',round(pred['seconds'],1),flush=True)
                mx.clear_cache()
        metric=evaluate(preds,labels) if not args.smoke else {}
        dump(out/'metrics.json',metric)
        summary[variant]={'n':len(preds),'seconds':time.time()-begin,'peak_mlx_gb':mx.get_peak_memory()/1e9,
                          'calls':sum(len(r['calls']) for r in preds),'output_tokens':sum(c['output_tokens'] for r in preds for c in r['calls']),
                          'length_stops':sum(c['finish_reason']=='length' for r in preds for c in r['calls'])}
        dump(args.out/'progress.json',summary)
    dump(args.out/'summary.json',{'completed':True,'variants':summary,'training':False,'seconds':time.time()-start})

if __name__=='__main__':main()
