"""Reload the pinned frozen model and repeat the first production batch."""
import os
import math
import time
from common import *
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

def main():
    check_prepared();assert (PRIVATE/'completed.json').exists()
    pin=json.loads(PIN.read_text())
    for f,h in pin['model_files'].items():assert sha(MODEL/f)==h
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler
    model,tok=load(str(MODEL));model.eval();mx.random.seed(573)
    items=[]
    for r in jsonl(PRIVATE/'prompts.jsonl'):
        s=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        items.append((r,tok.encode(s,add_special_tokens=False)))
    items.sort(key=lambda x:(len(x[1]),x[0]['variant'],x[0]['key']));batch=items[:4]
    ids=[tok.encode(w,add_special_tokens=False)[0] for w in ['UP','DOWN']]
    before=time.monotonic()
    gen=BatchGenerator(model,stop_tokens=[[t] for t in tok.eos_token_ids],sampler=make_sampler(temp=0),
                       completion_batch_size=4,prefill_batch_size=4,prefill_step_size=512)
    uids=gen.insert([t for _,t in batch],[1]*len(batch));scores={}
    try:
        while responses:=gen.next_generated():
            for r in responses:
                full=r.logprobs.astype(mx.float32);full=full-mx.logsumexp(full);lp=full[mx.array(ids)];mx.eval(lp)
                up,down=lp.tolist();scores[r.uid]=float(1/(1+math.exp(down-up)))
    finally:gen.close()
    old=jsonl(PRIVATE/'generated.jsonl')[:4]
    errors=[]
    for (r,_),uid,prev in zip(batch,uids,old):
        assert (r['key'],r['variant'])==(prev['key'],prev['variant'])
        errors.append(abs(prev['p']-scores[uid]))
    assert max(errors)<1e-5
    report=dict(reloaded_model=True,verified_calls=4,max_probability_error=max(errors),seconds=time.monotonic()-before,
                tolerance=1e-5,model_revision=pin['revision'],direction_equal=all((r['p']>=.5)==(scores[u]>=.5) for r,u in zip(old,uids)))
    dump(OUT/'model_reload.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
