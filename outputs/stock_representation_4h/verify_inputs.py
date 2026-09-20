"""Independently rebuild every lexical variant; never fit or inspect outcomes."""
from engine import *
from collections import Counter

def main():
 seal_check();d,*_=load();inp=joblib.load(WORK/'inputs.joblib');raw=pd.read_pickle(SOURCES['raw_index']);raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key');ks=sorted({k for s in d.news_record_keys for k in s.split('|') if k});views={}
 stemmer=EnglishStemmer();sf=lru_cache(maxsize=200000)(stemmer.stem)
 for arc,g in raw.loc[ks].groupby('archive'):
  with zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/arc) as z:
   for k,r in g.iterrows():
    rawtext=json.loads(z.read(r.member)).get('text','');body=html.unescape(re.sub('<[^>]+>',' ',rawtext))
    body=' '.join(line for line in body.splitlines() if not re.search('newsletter|privacy policy|terms of use|sign up now|please enter|advertisement|copyright|subscribe|click here',line,re.I));body=re.sub('https?://\\S+',' ',body)
    text=body+' '+str(r.title);words=[w for w in re.findall('[a-z]+',text.lower()) if len(w)>1];base=[sf(w) for w in words if w not in ENGLISH_STOP_WORDS]
    nums=[]
    for match in re.finditer(r'(?<![\w])[$€£]?\d+(?:[,.]\d+)*(?:%|[bBmM])?(?![\w])',text):
     token=''.join(c if c.isalnum() or c=='_' else {'$':'usd','€':'eur','£':'gbp','%':'pct',',':'comma','.':'dot'}.get(c,'_') for c in match[0].lower());nums.append('num_'+token)
    views[k]=dict(CONTROL=base,FREQUENCY=base,NEGATION=[sf(w) for w in words if w not in ENGLISH_STOP_WORDS-{'no','not','nor'}],NUMERIC=base+nums,NO_STEM=[w for w in words if w not in ENGLISH_STOP_WORDS])
 totals={m:0 for m in ['CONTROL']+VARIANTS}
 for i,r in enumerate(d.itertuples()):
  keys=[k for k in r.news_record_keys.split('|') if k]
  for m in totals:
   vals=[t for k in keys for t in views[k][m]];expected=' '.join(sorted(vals if m=='FREQUENCY' else set(vals)))
   assert inp['documents'][m][i]==expected
   if m=='CONTROL':assert expected==r.stem_body
   totals[m]+=len(expected.split())
  old=set(inp['documents']['CONTROL'][i].split());freq=set(inp['documents']['FREQUENCY'][i].split());assert old==freq
  assert set(inp['documents']['NEGATION'][i].split())-old <= {'not','no','nor'}
  assert {t for t in inp['documents']['NUMERIC'][i].split() if not t.startswith('num_')}==old
 dump(PUB/'INPUT_VERIFICATION.json',dict(status='PASS',articles_reread=len(ks),rows_reconstructed=len(d),variant_row_checks=len(d)*5,tokens=totals,fit_calls=0))
 print('Independent input reconstruction PASS',flush=True)
if __name__=='__main__':main()
