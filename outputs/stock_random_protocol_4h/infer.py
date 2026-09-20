"""Frozen, resumable MLX calls for exact prompt hashes. No stock-model training."""
import argparse,json,math,os,time,hashlib
from pathlib import Path
# The MLX runtime intentionally does not import the sklearn training environment.
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=HERE/'v1';PRIVATE=ROOT/'work/stock-data/random_protocol_4h/v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);args=ap.parse_args()
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
 src=PRIVATE/'analogy';mf=json.loads((src/'manifest.json').read_text())
 for n,h in mf['artifacts'].items():
  if sha(src/n)!=h:raise ValueError('changed prepared artifact: '+n)
 modelpath=ROOT/'work/stock-data/model_compare/model';pin=json.loads((ROOT/'work/stock-data/model_compare/comparison_v1/manifest.json').read_text())
 for n,h in pin['model_files'].items():
  if sha(modelpath/n)!=h:raise ValueError('model fingerprint mismatch: '+n)
 seal=dict(manifest=sha(src/'manifest.json'),model_revision=pin['revision'],model_files=pin['model_files'],code=sha(__file__),batch=4,thinking=False,seed=573,max_prompt_tokens=6000,output='conditional UP/DOWN token probability; not calibrated confidence')
 if (src/'inference_seal.json').exists():assert json.loads((src/'inference_seal.json').read_text())==seal
 else:dump(src/'inference_seal.json',seal)
 if (src/'inference_complete.json').exists():
  complete=json.loads((src/'inference_complete.json').read_text());assert complete['output_sha256']==sha(src/'generated.jsonl');print('matching frozen inference already complete');return
 import mlx.core as mx
 from mlx_lm import load
 from mlx_lm.generate import BatchGenerator
 from mlx_lm.sample_utils import make_sampler
 model,tok=load(str(modelpath));model.eval();mx.random.seed(573)
 choices=[tok.encode(w,add_special_tokens=False) for w in ['UP','DOWN']];assert all(len(v)==1 for v in choices);choices=[v[0] for v in choices]
 tasks=[]
 for row in map(json.loads,(src/'prompts.jsonl').read_text().splitlines()):
  assert row['prompt_hash']==digest(json.dumps(row['messages'],ensure_ascii=False))
  text=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False);tokens=tok.encode(text,add_special_tokens=False)
  if len(tokens)>6000:raise ValueError('prompt over frozen token budget; no truncation: '+row['prompt_hash'])
  tasks.append((row['prompt_hash'],tokens))
 tasks.sort(key=lambda q:(len(q[1]),q[0]));dump(OUT/'LLM_TOKEN_AUDIT.json',dict(prompts=len(tasks),max_tokens=max(len(t) for _,t in tasks),truncated=0,model=pin['model'],model_revision=pin['revision']))
 dest=src/'generated.jsonl';old=list(map(json.loads,dest.read_text().splitlines())) if dest.exists() else []
 assert [r['prompt_hash'] for r in old]==[h for h,_ in tasks[:len(old)]]
 stop=len(tasks) if args.limit is None else min(len(tasks),len(old)+args.limit);began=time.monotonic()
 with dest.open('a') as f:
  for offset in range(len(old),stop,4):
   batch=tasks[offset:min(offset+4,stop)];tick=time.monotonic()
   gen=BatchGenerator(model,stop_tokens=[[t] for t in tok.eos_token_ids],sampler=make_sampler(temp=0),completion_batch_size=4,prefill_batch_size=4,prefill_step_size=512)
   uids=gen.insert([tokens for _,tokens in batch],[1]*len(batch));scores={}
   try:
    while responses:=gen.next_generated():
     for r in responses:
      logp=r.logprobs.astype(mx.float32);logp=logp-mx.logsumexp(logp);v=logp[mx.array(choices)];mx.eval(v);scores[r.uid]=(v.tolist(),int(r.token))
   finally:gen.close()
   elapsed=time.monotonic()-tick
   for (h,tokens),uid in zip(batch,uids):
    lp,top=scores[uid];assert all(math.isfinite(v) for v in lp)
    a=max(lp);z=a+math.log(sum(math.exp(v-a) for v in lp));p=math.exp(lp[0]-z)
    row=dict(prompt_hash=h,p=p,logp_up=lp[0],logp_down=lp[1],choice_mass=math.exp(z),top_token=tok.decode([top]),prompt_tokens=len(tokens),seconds=elapsed/len(batch),peak_mlx_gb=mx.get_peak_memory()/1e9)
    f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
   mx.clear_cache()
   if offset%40==0 or offset+4>=stop:
    progress=dict(status='RUNNING' if offset+len(batch)<len(tasks) else 'COMPLETE',completed=offset+len(batch),total=len(tasks),session_seconds=time.monotonic()-began,model=pin['model'],encoder_training=False)
    dump(OUT/'LLM_PROGRESS.json',progress);print(json.dumps(progress),flush=True)
 if stop==len(tasks):
  rows=list(map(json.loads,dest.read_text().splitlines()));evidence=dict(status='COMPLETE',calls=len(rows),gradient_training=False,seconds=sum(r['seconds'] for r in rows),output_sha256=sha(dest),seal_sha256=sha(src/'inference_seal.json'),peak_mlx_gb=max(r['peak_mlx_gb'] for r in rows))
  dump(src/'inference_complete.json',evidence);dump(OUT/'LLM_EXECUTION.json',evidence)
if __name__=='__main__':main()
