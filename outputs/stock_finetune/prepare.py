import json,sys,time,copy
from pathlib import Path
import numpy as np,pandas as pd,torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
B=Path(__file__).resolve().parent;O=B.parent;ROOT=B.parents[1]/'work/stock-data';sys.path.insert(0,str(O/'stock_review_fixes'))
from guards import require_empty,sha,verify_files
from model import Tail,pool_bags
sys.path.insert(0,str(O/'stock_robust'));from common import PRICE
P=json.loads((B/'protocol.json').read_text());R=B/'results';C=ROOT/'e10_cache';require_empty(R);require_empty(C);R.mkdir(exist_ok=True);C.mkdir(exist_ok=True);(R/'preparing.json').write_text('{"status":"preparing"}')
source_paths=[B/'protocol.json',B/'model.py',B/'prepare.py',B/'run.py',O/'stock_review_fixes/guards.py',O/'stock_robust/common.py',O/'stock_baseline/results/results.json',O/'stock_baseline/run_baseline.py',O/'stock_finbert/results/article_probabilities.csv',O/'stock_finbert/results/inference.json']+[O/f'stock_finbert/results/{s}/development_features.csv' for s in ['AAPL','AMZN']]
sources={str(p.relative_to(O)):sha(p) for p in source_paths}
torch.set_num_threads(2);torch.manual_seed(573);a=pd.read_csv(O/'stock_finbert/results/article_probabilities.csv').set_index('record_key');a['at']=pd.to_datetime(a.available_utc,utc=True,format='mixed');frames=[];article_ids={};records=[];bag_rows=[];num=[]
for si,s in enumerate(['AAPL','AMZN']):
    d=pd.read_csv(O/f'stock_finbert/results/{s}/development_features.csv').fillna({'news_record_keys':''});d['stock_index']=si
    assert set(d.split)=={'train','validation'}
    for r in d.itertuples():
        keys=r.news_record_keys.split('|') if r.news_record_keys else [];cut=pd.Timestamp(r.cutoff_utc)
        assert a.loc[keys,'at'].le(cut).all() and a.loc[keys,'at'].gt(cut-pd.Timedelta(hours=4)).all()
        retained=sorted(keys,key=lambda k:(a.loc[k,'at'],k),reverse=True)[:16];bag=[]
        for key in retained:
            pair=(s,key)
            if pair not in article_ids:
                article_ids[pair]=len(records);company='Apple (AAPL)' if s=='AAPL' else 'Amazon (AMZN)';records.append(dict(symbol=s,record_key=key,text=f'Target company: {company}. News: {a.loc[key,"title"]}'))
            bag.append(article_ids[pair])
        bag_rows.append(bag+[-1]*(16-len(bag)));age=float((cut-a.loc[retained,'at']).dt.total_seconds().mean()/3600) if retained else 0.
        num.append([getattr(r,c) for c in PRICE]+[np.log1p(len(keys)),int(bool(keys)),age])
    frames.append(d)
d=pd.concat(frames,ignore_index=True);bags=np.asarray(bag_rows,dtype=np.int64);numeric=np.asarray(num);d[['symbol','start_utc','end_utc','cutoff_utc','label','split','has_news','news_count','stock_index','news_record_keys']].to_csv(R/'manifest.csv',index=False)
kw=dict(revision=P['revision'],cache_dir=str(ROOT/'finbert-cache'),local_files_only=True,trust_remote_code=False)
tok=AutoTokenizer.from_pretrained(P['model'],**kw);base=AutoModelForSequenceClassification.from_pretrained(P['model'],attn_implementation='eager',**kw);base.eval();base.requires_grad_(False);device='mps' if torch.backends.mps.is_available() else 'cpu';base.to(device)
state={}
for i,layer in enumerate(base.bert.encoder.layer[10:]):
    state.update({f'layers.{i}.{k}':v.detach().cpu().clone() for k,v in layer.state_dict().items()})
state.update({'pooler.'+k:v.detach().cpu().clone() for k,v in base.bert.pooler.state_dict().items()});config=base.config.to_dict();tail=Tail(config,state).to(device)
prefix=[];final=[];truncated=0;checks=[];t=time.time()
for start in range(0,len(records),32):
    texts=[r['text'] for r in records[start:start+32]];raw=tok(texts,add_special_tokens=True,truncation=False)['input_ids'];truncated+=sum(len(x)>96 for x in raw)
    batch=tok(texts,padding=True,truncation=True,max_length=96,return_tensors='pt').to(device);mask=batch['attention_mask'];ext=(1-mask[:,None,None,:].float())*torch.finfo(torch.float32).min
    with torch.no_grad():
        h=base.bert.embeddings(input_ids=batch['input_ids'],token_type_ids=batch['token_type_ids'])
        for layer in base.bert.encoder.layer[:10]:h=layer(h,attention_mask=ext)[0]
        cached=h.to(torch.float16).to(torch.float32);v=tail(cached,mask)
        if start==0:
            uncached=base.bert(**batch).pooler_output
            checks.append(dict(max_abs_pooler_error=float((uncached-v).abs().max().cpu()),mean_abs_pooler_error=float((uncached-v).abs().mean().cpu())))
    for j,n in enumerate(mask.sum(1).cpu().tolist()):prefix.append(h[j,:n].detach().cpu().to(torch.float16))
    final.extend(v.detach().cpu().numpy())
    if start%640==0:print('E10 cached',min(start+32,len(records)),len(records),'elapsed',round(time.time()-t),'s',flush=True)
final=np.asarray(final);pooled=pool_bags(final,bags);has=(bags>=0).any(1).astype(int)
torch.save(prefix,C/'prefix.pt');torch.save(state,C/'tail_state.pt');(C/'config.json').write_text(json.dumps(config,indent=2));np.savez_compressed(C/'data.npz',bags=bags,numeric=numeric,pooled=pooled,has=has,stock=d.stock_index.to_numpy(),y=d.label.to_numpy(),article_vectors=final)
pd.DataFrame(records).to_csv(C/'articles.csv',index=False)
summary=dict(protocol=P,status='prepared',source_sha256=sources,cache_sha256={p.name:sha(p) for p in C.iterdir() if p.is_file()},artifact_sha256={'manifest.csv':sha(R/'manifest.csv')},cache_dir=str(C),cache_checks=checks,unique_target_articles=len(records),truncated_titles=truncated,encoding_seconds=time.time()-t,stocks={})
for s in ['AAPL','AMZN']:
    summary['stocks'][s]={}
    for split in ['train','validation']:
        ix=np.flatnonzero(d.symbol.eq(s)&d.split.eq(split));full=int(d.news_count.iloc[ix].sum());ret=int((bags[ix]>=0).sum());summary['stocks'][s][split]=dict(windows=len(ix),original_occurrences=full,retained_occurrences=ret,retained_fraction=ret/full,capped_windows=int((d.news_count.iloc[ix]>16).sum()))
verify_files(O,sources);(R/'prepared.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary['stocks'],indent=2),flush=True);print('Cache reconstruction:',checks,flush=True)
