from importlib.metadata import version as package_version
import json,hashlib,time,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from experiment import fit_variant
B=Path(__file__).resolve().parent;ROOT=B.parents[1]/'work/stock-data';OUT=B/'results'
sys.path.insert(0,str(B.parent/'stock_review_fixes'));from guards import cache_readable,cache_metadata,sha
if (OUT/'results.json').exists():raise RuntimeError('Completed E05: preserve historical results and use a new experiment version.')
a=pd.read_json(OUT/'article_texts.jsonl',lines=True);original=pd.read_csv(B.parent/'stock_finbert/results/article_probabilities.csv').set_index('record_key')
cache=OUT/'text_probabilities.csv';digest=hashlib.sha256((OUT/'article_texts.jsonl').read_bytes()).hexdigest();meta=OUT/'inference.json'
expected=dict(texts_sha256=digest,upstream_probabilities_sha256=sha(B.parent/'stock_finbert/results/article_probabilities.csv'),upstream_inference_sha256=sha(B.parent/'stock_finbert/results/inference.json'),model='ProsusAI/finbert',revision='4556d13015211d73dccd3fdd39d39232506f3e43',max_length=256,truncation=True,padding='longest',encoding='sequence classification softmax; title from upstream',aggregation='article probabilities; downstream window mean/population std/log1p count',torch=torch.__version__,transformers=package_version('transformers'),tokenizers=package_version('tokenizers'),dtype='float32',device='mps' if torch.backends.mps.is_available() else 'cpu',encoder_script_sha256=sha(__file__))
if cache_readable(cache,meta,expected):pairs=pd.read_csv(cache)
else:
 kw=dict(revision='4556d13015211d73dccd3fdd39d39232506f3e43',cache_dir=str(ROOT/'finbert-cache'),local_files_only=True,trust_remote_code=False)
 tok=AutoTokenizer.from_pretrained('ProsusAI/finbert',**kw);model=AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert',**kw)
 torch.manual_seed(573);torch.set_num_threads(4);device='mps' if torch.backends.mps.is_available() else 'cpu';model.to(device).eval()
 strings=sorted(set(a.lead)|set(a.target));mapping={};start=time.time()
 for i in range(0,len(strings),16):
  batch=tok(strings[i:i+16],padding=True,truncation=True,max_length=256,return_tensors='pt').to(device)
  with torch.inference_mode():p=model(**batch).logits.softmax(-1).cpu().numpy()
  for text,v in zip(strings[i:i+16],p):mapping[text]={model.config.id2label[j].lower():float(x) for j,x in enumerate(v)}
  if i%320==0:print('Encoded',i,'/',len(strings),'elapsed',round(time.time()-start),flush=True)
 rows=[]
 for r in a.itertuples():
  for variant in ['title','lead','target']:
   prob=original.loc[r.record_key,['positive','negative','neutral']].to_dict() if variant=='title' else mapping[getattr(r,variant)]
   rows.append(dict(symbol=r.symbol,record_key=r.record_key,variant=variant,**prob))
 pairs=pd.DataFrame(rows);pairs.to_csv(cache,index=False);meta.write_text(json.dumps(dict(input_sha256=digest,model='ProsusAI/finbert',revision=kw['revision'],device=device,n_unique_new_text=len(strings),elapsed_seconds=time.time()-start,**cache_metadata(cache,expected)),indent=2))
result={'protocol':json.loads((B/'protocol.json').read_text()),'stocks':{}}
for s in ['AAPL','AMZN']:
 d=pd.read_csv(B.parent/f'stock_finbert/results/{s}/development_features.csv').fillna({'news_record_keys':''});texts=a[a.symbol.eq(s)].set_index('record_key');result['stocks'][s]={}
 for variant in ['title','lead','target']:
  probs=pairs[pairs.symbol.eq(s)&pairs.variant.eq(variant)].set_index('record_key');dd=d.copy();combined=[];features=[]
  for r in d.itertuples():
   ids=[k for k in r.news_record_keys.split('|') if k];combined.append(' . '.join(texts.loc[ids,variant]) if ids else '')
   q=probs.loc[ids];f={f'p_{c}':float(q[c].mean()) if ids else 0. for c in ['positive','negative','neutral']};f['p_std']=float((q.positive-q.negative).std(ddof=0)) if ids else 0.;f['count']=float(np.log1p(len(ids)));f['has']=int(bool(ids));features.append(f)
  dd['text']=combined;dd=pd.concat([dd,pd.DataFrame(features)],axis=1)
  for family in ['sparse','sentiment']:
   name=variant+'_'+family;extra=[] if family=='sparse' else ['p_positive','p_negative','p_neutral','p_std','count','has']
   result['stocks'][s][name]=fit_variant(dd,s,name,family,extra,OUT/s)
  (OUT/'results.json').write_text(json.dumps(result,indent=2))
print('E05 complete, test not evaluated.',flush=True)
