from run import *
import re,html,zipfile
from functools import lru_cache
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def main():
 out=B/'prepared';out.mkdir(exist_ok=False)
 source=O/'stock_horizons/runs/v1/data.pkl';d=pd.read_pickle(source);d=d[d.horizon=='4h'].copy();assert len(d)==1607
 d['news_record_keys']=d.article_keys.fillna('');needed=sorted({k for s in d.news_record_keys for k in s.split('|') if k})
 raw=pd.read_pickle(W/'audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key');body={};archive_hashes={}
 for archive,g in raw.loc[needed].groupby('archive'):
  path=W/'raw/news'/archive;archive_hashes[str(path)]=sha(path)
  with zipfile.ZipFile(path) as z:
   for row in g.itertuples():body[row.Index]=json.loads(z.read(row.member)).get('text','')
 stemmer=EnglishStemmer()
 @lru_cache(maxsize=150000)
 def stem(w):return stemmer.stem(w)
 def tokens(s):return {stem(w) for w in re.findall('[a-z]+',s.lower()) if len(w)>1 and w not in ENGLISH_STOP_WORDS}
 bad=re.compile(r'newsletter|privacy policy|terms of use|sign up now|please enter|advertisement|copyright|subscribe|click here',re.I);features={}
 for k in needed:
  text=html.unescape(re.sub(r'<[^>]+>',' ',body[k]));text=' '.join(line for line in text.splitlines() if not bad.search(line));text=re.sub(r'https?://\S+',' ',text);features[k]=tokens(text)|tokens(str(raw.loc[k,'title']))
 d['stem_body']=[' '.join(sorted(set().union(*(features[k] for k in s.split('|') if k)))) for s in d.news_record_keys]
 # Cross-check reusable text features at shared origins; this does not import their labels.
 previous=pd.concat([pd.read_pickle(O/'stock_robust/results/data.pkl'),pd.read_pickle(O/'stock_final_test/results/test_inputs.pkl')]);previous['join_key']=previous.symbol+'|'+pd.to_datetime(previous.start_utc,utc=True).astype(str);previous=previous.set_index('join_key')
 checked=0
 for r in d.itertuples():
  key=r.symbol+'|'+str(pd.Timestamp(r.start_utc))
  if key in previous.index:assert r.stem_body==previous.loc[key,'stem_body'];checked+=1
 # Original OHLC bars independently confirm exact four-hour outcomes for every row.
 for sym,g in d.groupby('symbol'):
  path=W/f"raw/CHARTS/{'APPLE' if sym=='AAPL' else 'AMAZON'}5.csv";bars=pd.read_csv(path,header=None,names=['date','time','open','high','low','close','activity']);bars.index=pd.to_datetime(bars.date+' '+bars.time,format='%Y.%m.%d %H:%M',utc=True)
  for r in g.itertuples():
   x=bars.reindex(pd.date_range(r.start_utc,r.end_utc-pd.Timedelta('5min'),freq='5min'));assert len(x)==48 and not x[['open','close']].isna().any().any();assert int(x.close.iloc[-1]>x.open.iloc[0])==r.label
 base=O/'stock_comprehensive/runs/v1/M08';finger=json.loads((base/'embedding_inputs.json').read_text());old=np.load(base/'articles.npz');old_embeddings=old['embeddings'];lookup={k:i for i,k in enumerate(old['keys'])};oldtitles=dict(finger['keys_and_titles'])
 hashes=json.loads((base/'embedding_fingerprint.json').read_text());assert sha(base/'articles.npz')==hashes[str(base/'articles.npz')];assert sha(base/'embedding_inputs.json')==hashes[str(base/'embedding_inputs.json')]
 titles={k:str(raw.loc[k,'title']) if pd.notna(raw.loc[k,'title']) else '' for k in needed};cached={k:old_embeddings[lookup[k]] for k in needed if k in lookup and oldtitles[k]==titles[k]};missing=[k for k in needed if k not in cached]
 if missing:
  import torch
  from transformers import AutoTokenizer,AutoModelForSequenceClassification
  kw=dict(revision=finger['revision'],cache_dir=str(W/'finbert-cache'),local_files_only=True,trust_remote_code=False);torch.manual_seed(573);torch.set_num_threads(2);device='mps' if torch.backends.mps.is_available() else 'cpu';tok=AutoTokenizer.from_pretrained('ProsusAI/finbert',**kw);model=AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert',**kw).to(device).eval()
  for i in range(0,len(missing),16):
   ks=missing[i:i+16];x=tok([titles[k] for k in ks],padding=True,truncation=True,max_length=256,return_special_tokens_mask=True,return_tensors='pt').to(device);special=x.pop('special_tokens_mask');mask=(x['attention_mask']*(1-special)).unsqueeze(-1)
   with torch.inference_mode():v=((model.bert(**x).last_hidden_state*mask).sum(1)/mask.sum(1).clamp(min=1)).cpu().numpy()
   cached.update(dict(zip(ks,v)))
 np.savez_compressed(out/'articles.npz',keys=np.array(needed),embeddings=np.stack([cached[k] for k in needed]));finger['keys_and_titles']=[[k,titles[k]] for k in needed];finger['additional_frozen_inference_keys']=missing;dump(out/'embedding_inputs.json',finger);d.to_pickle(out/'data.pkl')
 dump(out/'manifest.json',dict(horizon='4h',rows=len(d),split_counts={'|'.join(k):int(v) for k,v in d.groupby(['symbol','split']).size().items()},body_shared_origin_parity=checked,raw_labels_verified=1607,articles=len(needed),new_frozen_embeddings=len(missing),input_hash=sha(source),archive_hashes=archive_hashes,prepared_hashes={x:sha(out/x) for x in ['data.pkl','articles.npz','embedding_inputs.json']},body_content_hashes={k:hashlib.sha256(body[k].encode()).hexdigest() for k in needed}))
 print('Prepared 4h',len(d),'articles',len(needed),'new embeddings',len(missing),'shared text parity',checked,flush=True)
if __name__=='__main__':main()
