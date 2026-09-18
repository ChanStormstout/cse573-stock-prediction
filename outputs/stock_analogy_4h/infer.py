"""Frozen local MLX inference; append-only logs with strict recovery fingerprint."""
import argparse
import math
import os
import time
from common import *

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

def main(limit=None):
    seal=check_prepared();pin=json.loads(PIN.read_text())
    for name,h in pin['model_files'].items():
        if sha(MODEL/name)!=h:raise ValueError('model changed: '+name)
    stamp=dict(prepared_manifest_sha256=sha(PRIVATE/'manifest.json'),model=pin,
               inference_code=sha(Path(__file__)),seed=573,batch=4,thinking=False,gradient_training=False,
               output='UP/DOWN conditional token probabilities',max_prompt_tokens=6000)
    mf=PRIVATE/'inference_manifest.json'
    if mf.exists():
        if json.loads(mf.read_text())!=stamp:raise ValueError('inference resume mismatch')
    else:dump(mf,stamp)
    if (PRIVATE/'completed.json').exists():print('matching inference complete');return
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    model,tok=load(str(MODEL));model.eval();mx.random.seed(573)
    ids=[tok.encode(w,add_special_tokens=False) for w in ['UP','DOWN']]
    assert all(len(v)==1 for v in ids);ids=[v[0] for v in ids]
    assert [tok.decode([v]) for v in ids]==['UP','DOWN']
    items=[]
    for r in jsonl(PRIVATE/'prompts.jsonl'):
        txt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tokens=tok.encode(txt,add_special_tokens=False)
        if len(tokens)>6000:raise ValueError('would truncate registered input')
        items.append((r,tokens))
    items.sort(key=lambda x:(len(x[1]),x[0]['variant'],x[0]['key']))
    dump(OUT/'token_audit.json',dict(max_tokens=max(len(t) for _,t in items),
        median_tokens=float(np.median([len(t) for _,t in items])),prompts=len(items),truncated=0))
    done=jsonl(PRIVATE/'generated.jsonl')
    assert [(r['variant'],r['key']) for r in done]==[(r['variant'],r['key']) for r,_ in items[:len(done)]]
    stop=len(items) if limit is None else min(len(items),len(done)+limit)
    began=time.monotonic()
    with (PRIVATE/'generated.jsonl').open('a') as f:
        for offset in range(len(done),stop,4):
            batch=items[offset:min(offset+4,stop)];tick=time.monotonic()
            gen=BatchGenerator(model,stop_tokens=[[t] for t in tok.eos_token_ids],sampler=make_sampler(temp=0),
                               completion_batch_size=4,prefill_batch_size=4,prefill_step_size=512)
            uids=gen.insert([t for _,t in batch],[1]*len(batch));sc={}
            try:
                while responses:=gen.next_generated():
                    for a in responses:
                        full=a.logprobs.astype(mx.float32);full=full-mx.logsumexp(full)
                        lp=full[mx.array(ids)];mx.eval(lp);sc[a.uid]=(lp.tolist(),int(a.token))
            finally:gen.close()
            elapsed=time.monotonic()-tick
            for (r,t),uid in zip(batch,uids):
                lp,top=sc[uid];assert all(math.isfinite(v) for v in lp)
                maximum=max(lp);z=maximum+math.log(sum(math.exp(v-maximum) for v in lp));p=math.exp(lp[0]-z)
                result=dict(key=r['key'],variant=r['variant'],p=p,logp_up=lp[0],logp_down=lp[1],
                    choice_mass=math.exp(z),unconstrained_token=tok.decode([top]),prompt_tokens=len(t),
                    prompt_sha256=digest(json.dumps(r['messages'],ensure_ascii=False)),seconds=elapsed/len(batch),
                    peak_mlx_gb=mx.get_peak_memory()/1e9)
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
            mx.clear_cache()
            if offset%40==0 or offset+4>=stop:
                print('inferred',min(offset+4,stop),'/',len(items),'session seconds',round(time.monotonic()-began,1),flush=True)
    if stop==len(items):
        check_prepared();rows=jsonl(PRIVATE/'generated.jsonl')
        evidence=dict(model=pin['model'],revision=pin['revision'],calls=len(rows),gradient_training=False,
            device='local MLX Metal',seconds=sum(r['seconds'] for r in rows),peak_mlx_gb=max(r['peak_mlx_gb'] for r in rows),
            seed=573,output_sha256=sha(PRIVATE/'generated.jsonl'),manifest_sha256=sha(mf),
            per_variant={v:sum(r['variant']==v for r in rows) for v in ['P0','P2','P3']})
        dump(PRIVATE/'completed.json',evidence);dump(OUT/'execution.json',evidence)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int);a=p.parse_args();main(a.limit)
