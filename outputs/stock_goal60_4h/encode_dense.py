"""Identical protected candidate tokens for frozen original and existing A1."""
import sys,json,re,zipfile,time,gc
from pathlib import Path
import numpy as np
import torch
from core import ROOT,W,PRIVATE,dump,sha,data
sys.path.insert(0,str(ROOT/'outputs/stock_finbert_event_adapter_4h'))
from adapter import tokenizer,EncodedDataset,article_examples,EventAdapter,load_trainable

def run():
 d=data();raw=__import__('pandas').read_pickle(W/'audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key');pairs=sorted({(r.symbol,k) for r in d.itertuples() for k in str(r.news_record_keys or '').split('|') if k and k!='nan'});keys={k for s,k in pairs};texts={}
 for archive,g in raw.loc[sorted(keys)].groupby('archive'):
  with zipfile.ZipFile(W/'raw/news'/archive) as z:
   for r in g.itertuples():texts[r.Index]=json.loads(z.read(r.member)).get('text','')
 inputs=[]
 for s,k in pairs:
  sentences=[v.strip() for v in re.split(r'(?<=[.!?])\s+|\n+',texts[k]) if v.strip()];pat=r'\b(?:AAPL|Apple)\b' if s=='AAPL' else r'\b(?:AMZN|Amazon)\b';hit=[i for i,t in enumerate(sentences) if re.search(pat,t,re.I)][:2];title=str(raw.at[k,'title']);fallback=not hit
  if fallback:sentences=[title];hit=[0]
  context=sorted({j for i in hit for j in range(max(0,i-1),min(len(sentences),i+2))});inputs.append(dict(id=s+'|'+k,symbol=s,record_key=k,title=title,sentences={f'S{i}':sentences[i] for i in context},candidate_ids=[f'S{i}' for i in hit],title_fallback=fallback))
 dump(PRIVATE/'dense_inputs.json',inputs);tok=tokenizer();ds=EncodedDataset(article_examples(inputs),tok);dump(PRIVATE/'dense_tokens.json',ds.encodings)
 checkpoints={f'A1_{s}':W/f'finbert_event_adapter_4h/v1/adapter_runs/checkpoints/A1_seed{s}.pt' for s in (573,574,575)}
 fp=dict(inputs=sha(PRIVATE/'dense_inputs.json'),tokens=sha(PRIVATE/'dense_tokens.json'),script=sha(__file__),checkpoints={k:sha(p) for k,p in checkpoints.items()})
 ids=[a['id'] for a in inputs];lookup={a:i for i,a in enumerate(ids)};order=sorted(range(len(ds)),key=lambda i:ds.lengths[i]);device='mps' if torch.backends.mps.is_available() else 'cpu';evidence=[]
 for name in ['original']+list(checkpoints):
  path=PRIVATE/f'dense_{name}.npz';manifest=path.with_suffix('.json')
  if path.exists():assert json.loads(manifest.read_text())['fingerprint']==fp;continue
  start=time.monotonic();model=EventAdapter('A0' if name=='original' else 'A1')
  if name!='original':load_trainable(model,torch.load(checkpoints[name],map_location='cpu',weights_only=True)['state'])
  model.eval().requires_grad_(False).to(device);vec=np.zeros((len(ds),768),np.float32)
  with torch.inference_mode():
   for off in range(0,len(order),24):
    ii=order[off:off+24];batch=tok.pad([ds.encodings[i] for i in ii],return_tensors='pt').to(device);v=model.encoder(**batch).last_hidden_state[:,0].float().cpu().numpy();vec[ii]=v
    if off%2400==0:print(name,off,len(ds),flush=True)
  # First average chunks within evidence sentence, then sentences within article.
  groups={}
  for i,r in enumerate(ds.examples):groups.setdefault((r['article_id'],r['sentence_id']),[]).append(i)
  articles={}
  for (a,s),ii in groups.items():articles.setdefault(a,[]).append(vec[ii].mean(0))
  result=np.array([np.mean(articles[a],axis=0) for a in ids]);np.savez_compressed(path,keys=np.array(ids),embeddings=result)
  ev=dict(fingerprint=fp,name=name,device=device,seconds=time.monotonic()-start,articles=len(ids),units=len(ds),title_fallback=sum(a['title_fallback'] for a in inputs),encoder_training=False,sha256=sha(path));dump(manifest,ev);evidence.append(ev);print(name,'complete',ev['seconds'],flush=True);del model;gc.collect()
  if device=='mps':torch.mps.empty_cache()
 dump(PRIVATE/'dense_encoding.json',evidence)
if __name__=='__main__':run()
