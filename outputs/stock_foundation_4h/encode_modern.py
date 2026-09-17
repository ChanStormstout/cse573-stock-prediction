"""Matched frozen title encoder, resumable content/config-fingerprinted cache."""
import sys,time,json
from pathlib import Path
import numpy as np,torch
from transformers import AutoModel,AutoTokenizer
from experiment import ROOT,HERE,W,PRIVATE,OUT,init,inputs,sha,dump

def main():
 init();d,keys,_,_,_,_=inputs();manifest=ROOT/'outputs/stock_integrated_4h/prepared/embedding_inputs.json';titles=dict(json.loads(manifest.read_text())['keys_and_titles']);texts=[titles[k] for k in keys];model_path=W/'models/modern'
 fingerprint=dict(manifest=sha(manifest),weights=sha(model_path/'model.safetensors'),config=sha(model_path/'config.json'),tokenizer=sha(model_path/'tokenizer.json'),revision='31d3d96d5839a03dccd030bea40b77c2649a9a01',max_length=256,pooling='attention_mask_mean',dtype='float32',code=sha(Path(__file__)))
 cache=PRIVATE/'modern_cache.json'
 if cache.exists():assert json.loads(cache.read_text())==fingerprint,'cache mismatch'
 else:dump(cache,fingerprint)
 if (PRIVATE/'modern_embeddings.npz').exists():raise FileExistsError('encoder already complete')
 torch.manual_seed(573);torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu';tick=time.monotonic()
 tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True);model=AutoModel.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager',reference_compile=False).eval().to(device);model.requires_grad_(False)
 chunkdir=PRIVATE/'modern_chunks';chunkdir.mkdir(exist_ok=True);arrays=[];tokens=[];calls=0
 for start in range(0,len(keys),32):
  p=chunkdir/f'{start:05d}.npz';tx=texts[start:start+32];kk=keys[start:start+32]
  if p.exists():
   z=np.load(p);assert np.array_equal(z['keys'],kk);v=z['embeddings'];lens=z['lengths']
  else:
   lens=np.array([len(tokenizer(t,truncation=False)['input_ids']) for t in tx]);x=tokenizer(tx,padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
   with torch.inference_mode():
    mask=x['attention_mask'].unsqueeze(-1);h=model(**x).last_hidden_state;v=((h*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu().numpy()
   assert np.isfinite(v).all();np.savez_compressed(p,keys=kk,embeddings=v,lengths=lens);calls+=1
  arrays.append(v);tokens.extend(lens.tolist())
  if start%320==0:print('modern',start+len(tx),'/',len(keys),'seconds',round(time.monotonic()-tick,1),flush=True)
  if time.monotonic()-tick>7200:raise TimeoutError('2-hour encoder budget reached; chunks preserved')
 emb=np.concatenate(arrays);np.savez_compressed(PRIVATE/'modern_embeddings.npz',keys=keys,embeddings=emb)
 evidence=dict(model='clapAI/Fin-ModernBERT',revision=fingerprint['revision'],device=device,dtype='float32',parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),gradients_present=any(p.grad is not None for p in model.parameters()),encoder_training=False,articles=len(keys),new_batches=calls,truncated_articles=int((np.array(tokens)>256).sum()),seconds=time.monotonic()-tick,embeddings_sha256=sha(PRIVATE/'modern_embeddings.npz'),fingerprint=fingerprint,torch=torch.__version__)
 dump(OUT/'modern_inference.json',evidence);print('COMPLETE',evidence,flush=True)
if __name__=='__main__':main()
