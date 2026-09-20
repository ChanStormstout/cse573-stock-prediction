"""Frozen matched paragraph encoders, full token coverage and append-only batches."""
import argparse,sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stock_random_protocol_4h'))
from common import ROOT,sha,dump
import numpy as np,torch
from transformers import AutoTokenizer,AutoModel
HERE=Path(__file__).resolve().parent;OUT=HERE/'v1';SRC=ROOT/'work/stock-data/full_semantics_4h/v1/semantic'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('encoder',choices=['FINBERT','MODERN']);args=ap.parse_args();name=args.encoder
 mf=json.loads((SRC/'manifest.json').read_text())
 for n,h in mf['artifacts'].items():assert sha(SRC/n)==h
 rows=[json.loads(s) for s in (SRC/'chunks.jsonl').read_text().splitlines()];articles=[json.loads(s) for s in (SRC/'articles.jsonl').read_text().splitlines()]
 if name=='FINBERT':
  mp='ProsusAI/finbert';kw=dict(revision=mf['finbert_revision'],cache_dir=str(ROOT/'work/stock-data/finbert-cache'),local_files_only=True)
 else:mp=ROOT/'work/stock-data/foundation_4h/models/modern';kw=dict(local_files_only=True)
 binary_root=(ROOT/'work/stock-data/finbert-cache/models--ProsusAI--finbert/snapshots'/mf['finbert_revision']) if name=='FINBERT' else Path(mp)
 binary_hashes={p.name:sha(p) for p in binary_root.iterdir() if p.is_file()}
 assert any(n.endswith(('.bin','.safetensors')) for n in binary_hashes)
 dst=SRC/name;dst.mkdir(exist_ok=True)
 stamp={'code':sha(__file__),'input':sha(SRC/'manifest.json'),'protocol':sha(OUT/'SEMANTIC_PROTOCOL.json'),'encoder':name,'binary_hashes':binary_hashes,'dtype':'float32','batch':8}
 if (dst/'seal.json').exists():assert json.loads((dst/'seal.json').read_text())==stamp
 else:dump(dst/'seal.json',stamp)
 tok=AutoTokenizer.from_pretrained(mp,**kw);model=AutoModel.from_pretrained(mp,attn_implementation='eager',**({'reference_compile':False} if name=='MODERN' else {}),**kw).eval()
 model.requires_grad_(False);device='mps' if torch.backends.mps.is_available() else 'cpu';model.to(device);torch.set_num_threads(2);torch.manual_seed(573);start=time.monotonic();parts=[];weights=[]
 for i in range(0,len(rows),8):
  target=dst/f'batch_{i:06d}.npz';batch=rows[i:i+8];ids=[r['chunk_id'] for r in batch]
  if target.exists():
   assert json.loads(target.with_suffix('.json').read_text())['sha256']==sha(target)
   z=np.load(target);assert z['keys'].tolist()==ids;parts.append(z['vectors']);weights.extend(z['counts']);continue
  x=tok([r['text'] for r in batch],padding=True,truncation=False,return_special_tokens_mask=True,return_tensors='pt');assert x['input_ids'].shape[1]<=512
  special=x.pop('special_tokens_mask').to(device);x={k:v.to(device) for k,v in x.items()};mask=x['attention_mask']*(1-special)
  with torch.inference_mode():v=model(**x).last_hidden_state;v=(v*mask.unsqueeze(-1)).sum(1)/mask.sum(1).clamp(min=1).unsqueeze(-1)
  vectors=v.cpu().numpy();counts=mask.sum(1).cpu().numpy();assert np.isfinite(vectors).all()
  np.savez_compressed(target,keys=np.array(ids),vectors=vectors,counts=counts);dump(target.with_suffix('.json'),{'sha256':sha(target),'seal_sha256':sha(dst/'seal.json')});parts.append(vectors);weights.extend(counts)
  if i%160==0:print(name,i,len(rows),round(time.monotonic()-start,1),flush=True)
 vectors=np.concatenate(parts);weights=np.asarray(weights);lookup={r['chunk_id']:i for i,r in enumerate(rows)};av=[]
 for a in articles:
  ix=[lookup[k] for k in a['chunks']];av.append(np.average(vectors[ix],axis=0,weights=weights[ix]) if ix else np.zeros(vectors.shape[1]))
 target=SRC/(name+'_articles.npz');assert not target.exists();np.savez_compressed(target,keys=np.array([a['article_id'] for a in articles]),vectors=np.stack(av),has_evidence=np.array([bool(a['chunks']) for a in articles]))
 assert all(p.grad is None and not p.requires_grad for p in model.parameters())
 dump(OUT/(name+'_ENCODING.json'),dict(status='COMPLETE',chunks=len(rows),article_target_pairs=len(articles),device=device,seconds=time.monotonic()-start,encoder_training=False,trainable_parameters=0,gradients_present=False,output_sha256=sha(target),seal=stamp))
if __name__=='__main__':main()
