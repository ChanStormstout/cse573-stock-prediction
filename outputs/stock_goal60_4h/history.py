"""Past-only accepted-news history and observed reaction; no new title filters."""
from core import *
def prepare():
 d=data();raw=pd.read_pickle(W/'audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;sc=schedule();bars={s:load_bars(s) for s in ('AAPL','AMZN')};members=[];react=[];coverage=[];allkeys=set()
 for r in d.itertuples():
  original=[k for k in str(r.news_record_keys or '').split('|') if k and k!='nan'];members.append(dict(current=original,old=[]));react.append({})
 eligible={}
 # Prefix sums over scheduled 5-minute bars preserve missingness and avoid repeated scans.
 grid=pd.DatetimeIndex(np.concatenate([pd.date_range(r.open,r.close-pd.Timedelta('5min'),freq='5min').to_numpy() for r in sc.itertuples()]));grid=pd.to_datetime(grid,utc=True);ticks=grid.asi8;cache={}
 for symbol,b in bars.items():
  g=b.reindex(grid);missing=g[['open','close','high','low']].isna().any(axis=1).to_numpy();rv=np.nan_to_num(np.square(np.log(g.close.to_numpy()/g.open.to_numpy())));cache[symbol]=(g.open.to_numpy(),g.close.to_numpy(),np.r_[0,np.cumsum(missing)],np.r_[0,np.cumsum(rv)])

 for s in ('AAPL','AMZN'):
  pat=r'\b(?:AAPL|Apple)\b' if s=='AAPL' else r'\b(?:AMZN|Amazon)\b'
  g=raw[raw.language.eq('english')&raw.chars.ge(100)&raw.lag_hours.ge(0)&~raw.physician&raw.title.fillna('').str.contains(pat,case=False,regex=True)].sort_values(['available_utc','archive','member']).drop_duplicates('normalized_hash').drop_duplicates('title_norm');eligible[s]=g
 for i,r in enumerate(d.itertuples()):
  j=int(np.flatnonzero(sc.open.dt.date==r.start_utc.date())[0]);lower=sc.iloc[max(0,j-2)].open;g=eligible[r.symbol];g=g[(g.available_utc>=lower)&(g.available_utc<=r.cutoff_utc)];original=set(members[i]['current']);old=[k for k in g.key if k not in original];members[i]['old']=old;allkeys.update(original);allkeys.update(old)
  # Completed scheduled bars only; unknown if any expected interval is absent.
  values=[];unknown=0;overnight=0
  op,cl,miss,rvsum=cache[r.symbol];end=int(np.searchsorted(ticks,(r.cutoff_utc-pd.Timedelta('5min')).value,side='right'))
  for a in g.itertuples():
   start=int(np.searchsorted(ticks,a.available_utc.value,side='left'))
   if start>=end or miss[end]-miss[start]>0:unknown+=1;continue
   overnight+=int(grid[start].date()!=grid[end-1].date());values.append((np.log(cl[end-1]/op[start]),np.sqrt(max(0,rvsum[end]-rvsum[start]))))
  react[i]=dict(reaction_return=float(np.mean([v[0] for v in values])) if values else 0.,reaction_rv=float(np.mean([v[1] for v in values])) if values else 0.,reaction_unknown_fraction=unknown/max(1,len(g)),reaction_overnight_fraction=overnight/max(1,len(g)),reaction_count=np.log1p(len(values)))
  coverage.append(dict(key=r.key,symbol=r.symbol,phase=r.phase,current_n=len(original),old_n=len(old),original_empty=int(not original),filled=int(not original and bool(old)),earliest_allowed=str(lower),latest_used=str(g.available_utc.max()) if len(g) else None,cutoff=str(r.cutoff_utc)))
  assert not len(g) or g.available_utc.max()<=r.cutoff_utc
 dump(PRIVATE/'history_members.json',members);pd.DataFrame(react).to_pickle(PRIVATE/'history_reactions.pkl');pd.DataFrame(coverage).to_csv(PRIVATE/'history_coverage.csv',index=False)
 existing=np.load(ROOT/'outputs/stock_integrated_4h/prepared/articles.npz');oldmap={k:i for i,k in enumerate(existing['keys'])};missing=sorted(allkeys-set(oldmap));rindex=raw.set_index('key');dump(PRIVATE/'history_encode_inputs.json',[dict(key=k,title=str(rindex.at[k,'title'])) for k in missing]);summary=pd.DataFrame(coverage).groupby(['symbol','phase'])[['current_n','old_n','original_empty','filled']].agg({'current_n':'sum','old_n':'sum','original_empty':'sum','filled':'sum'}).reset_index();summary.to_csv(OUT/'history_coverage_summary.csv',index=False);dump(PRIVATE/'history_prepare.json',dict(unique_articles=len(allkeys),extra_titles=len(missing),source=sha(W/'audit/news_index.pkl'),script=sha(__file__)))
 print('history prepared',len(allkeys),'extra',len(missing),flush=True)
def encode():
 import torch
 from transformers import AutoTokenizer,AutoModel
 source=PRIVATE/'history_encode_inputs.json';rows=json.loads(source.read_text());path=PRIVATE/'history_embeddings.npz';fp=dict(source=sha(source),script=sha(__file__),revision='4556d13015211d73dccd3fdd39d39232506f3e43',pooling='attention mean excluding special',max_length=256)
 if path.exists():assert json.loads(path.with_suffix('.json').read_text())['fingerprint']==fp;return
 kwargs=dict(revision=fp['revision'],cache_dir=str(W/'finbert-cache'),local_files_only=True,trust_remote_code=False);tok=AutoTokenizer.from_pretrained('ProsusAI/finbert',**kwargs);model=AutoModel.from_pretrained('ProsusAI/finbert',**kwargs).eval().requires_grad_(False);device='mps' if torch.backends.mps.is_available() else 'cpu';model.to(device);v=[];start=time.monotonic()
 with torch.inference_mode():
  for off in range(0,len(rows),48):
   x=tok([a['title'] for a in rows[off:off+48]],padding=True,truncation=True,max_length=256,return_special_tokens_mask=True,return_tensors='pt').to(device);mask=x['attention_mask']*(1-x.pop('special_tokens_mask'));h=model(**x).last_hidden_state;v.append(((h*mask[:,:,None]).sum(1)/mask.sum(1).clamp(min=1)[:,None]).cpu().numpy())
   if off%2400==0:print('history encode',off,len(rows),flush=True)
 old=np.load(ROOT/'outputs/stock_integrated_4h/prepared/articles.npz');keys=np.r_[old['keys'],np.array([a['key'] for a in rows])];emb=np.concatenate([old['embeddings'],np.concatenate(v)]);np.savez_compressed(path,keys=keys,embeddings=emb);dump(path.with_suffix('.json'),dict(fingerprint=fp,seconds=time.monotonic()-start,device=device,encoded=len(rows),sha256=sha(path)))
if __name__=='__main__':
 import sys
 prepare() if len(sys.argv)==1 or sys.argv[1]=='prepare' else encode()
