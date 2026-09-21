#!/usr/bin/env python3
"""Pinned, resumable one-token binary inference for article or analogy jobs."""
from __future__ import annotations
import argparse,hashlib,json,math,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];MODEL=ROOT/'work/stock-data/model_compare/model'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main(inp,out,batch_size,resume):
    expected=json.loads((Path(__file__).parent/'v1/ARTICLE_JOB_AUDIT.json').read_text())
    actual={p.name:sha(p) for p in MODEL.glob('*') if p.is_file()}
    if actual!=expected['model_files']:raise RuntimeError('model fingerprint mismatch')
    jobs=rows(inp);ids=[x['job_id'] for x in jobs]
    if len(ids)!=len(set(ids)):raise RuntimeError('duplicate jobs')
    os.environ.update({'HF_HOME':str(ROOT/'work/stock-data/model_compare/hf-home'),'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    model,tok=load(str(MODEL));model.eval();mx.random.seed(573)
    answer=[tok.encode(x,add_special_tokens=False) for x in ('A','B')]
    if any(len(x)!=1 for x in answer):raise RuntimeError(f'A/B not single tokens: {answer}')
    answer=[x[0] for x in answer]
    prepared=[]
    for j in jobs:
        msg=[{'role':'system','content':j['system']},{'role':'user','content':j['prompt']}]
        text=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        token=tok.encode(text,add_special_tokens=False)
        if len(token)+1>6144:raise RuntimeError('prompt truncation')
        prepared.append((j,token))
    prepared.sort(key=lambda x:(len(x[1]),x[0]['job_id']))
    done=rows(out) if Path(out).exists() else []
    if done and not resume:raise FileExistsError(out)
    if [x['job_id'] for x in done]!=[x[0]['job_id'] for x in prepared[:len(done)]]:raise RuntimeError('resume prefix mismatch')
    began=time.time()
    with Path(out).open('a') as stream:
      for off in range(len(done),len(prepared),batch_size):
        batch=prepared[off:off+batch_size]
        gen=BatchGenerator(model,stop_tokens=[[x] for x in tok.eos_token_ids],sampler=make_sampler(temp=0),completion_batch_size=batch_size,prefill_batch_size=batch_size,prefill_step_size=512)
        uids=gen.insert([x[1] for x in batch],[1]*len(batch));scores={}
        try:
          while response:=gen.next_generated():
            for x in response:
              lp=x.logprobs.astype(mx.float32);lp=lp-mx.logsumexp(lp);sel=lp[mx.array(answer)];mx.eval(sel);scores[x.uid]=sel.tolist()
        finally:gen.close()
        for (j,tokens),uid in zip(batch,uids):
          a,b=scores[uid];mxv=max(a,b);pa=math.exp(a-mxv)/(math.exp(a-mxv)+math.exp(b-mxv));rec={'job_id':j['job_id'],'article_key':j.get('article_key'),'key':j.get('key'),'task':j['task'],'p_A':pa,'prompt_tokens':len(tokens)};stream.write(json.dumps(rec)+'\n');stream.flush();done.append(rec)
        mx.clear_cache()
        if len(done)%200<batch_size or len(done)==len(prepared):print(len(done),'/',len(prepared),'seconds',round(time.time()-began,1),flush=True)
    summary={'status':'COMPLETE','jobs':len(done),'input_sha256':sha(inp),'output_sha256':sha(out),'max_prompt_tokens':max(x['prompt_tokens'] for x in done),'model_revision':expected['model_revision'],'training':False}
    Path(str(out)+'.summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--resume',action='store_true');a=p.parse_args();main(a.input,a.output,a.batch_size,a.resume)
