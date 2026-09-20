"""Raw-source reread, independent lexical reconstruction; zero model fits."""
from core import *
import zipfile,collections

def main():
 audit=json.loads((PUBLIC/'INPUT_AUDIT.json').read_text());assert sha(WORK/'inputs.joblib')==audit['input_sha256'];assert sha(WORK/'canonical.pkl')==audit['canonical_sha256']
 inp=joblib.load(WORK/'inputs.joblib');d,*_=load();raw=pd.read_pickle(SOURCES['raw_index']);raw['rk']=raw.archive+'::'+raw.member;raw=raw.set_index('rk');old_manifest=json.loads(SOURCES['canonical_manifest'].read_text())
 n=0;tokens=0;repeats=0
 from nltk.stem.snowball import EnglishStemmer
 from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
 from functools import lru_cache
 stem= lru_cache(maxsize=150000)(EnglishStemmer().stem);oldsets={};views={}
 for arc,g in raw.loc[sorted(inp['articles'])].groupby('archive'):
  with zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/arc) as z:
   for k,r in g.iterrows():
    body=json.loads(z.read(r.member)).get('text','');title='' if pd.isna(r.title) else str(r.title);saved=inp['articles'][k]
    assert digest(body)==saved['raw_body_sha256']==old_manifest['body_content_hashes'][k];assert digest(title)==saved['title_sha256']
    text=re.sub(r'<(?:script|style)\b[^>]*>.*?</(?:script|style)>',' ',body,flags=re.I|re.S)
    text=re.sub(r'</?(?:p|div|br|li|h[1-6]|tr)\b[^>]*>','\n',text,flags=re.I);text=html.unescape(re.sub(r'<[^>]+>',' ',text))
    expected=[]
    for origin,t in [('title',title),('body',text)]:
     for line_index,line in enumerate(t.splitlines()):
      line=' '.join(re.sub(r'https?://\S+',' ',line).split())
      if not line or (origin=='body' and re.search(r'^(?:advertisement|privacy policy|terms of use|copyright|subscribe|sign up now|click here|newsletter)\b',line,re.I)):continue
      for sentence in re.split(r'(?<!\d)[.!?;]+|[.!?;]+(?!\d)',line):
       words=re.findall(r"[$€£]?\d+(?:[,.]\d+)*(?:%|[bBmM])?|[A-Za-z]+(?:['’][A-Za-z]+)?",sentence.lower())
       if words:expected.append({'origin':origin,'line':line_index,'text':sentence.strip(),'tokens':words})
    assert expected==saved['parts'];views[k]=expected
    # Rebuild the old representation rather than infer it from appearance alone.
    cleanbody=html.unescape(re.sub(r'<[^>]+>',' ',body));cleanbody=' '.join(l for l in cleanbody.splitlines() if not re.search(r'newsletter|privacy policy|terms of use|sign up now|please enter|advertisement|copyright|subscribe|click here',l,re.I));cleanbody=re.sub(r'https?://\S+',' ',cleanbody)
    oldsets[k]={stem(w) for txt in [cleanbody,title] for w in re.findall('[a-z]+',txt.lower()) if len(w)>1 and w not in ENGLISH_STOP_WORDS};n+=1
 for i,r in enumerate(d.itertuples()):
  ks=[k for k in r.news_record_keys.split('|') if k];assert ks==inp['mapping'][i]['article_keys'];u=[];b=[]
  for k in ks:
   for s in views[k]:
    w=s['tokens'];u.extend(w);b.extend(w);b.extend(w[j]+' '+w[j+1] for j in range(len(w)-1))
  assert u==inp['docs'][i]['u'] and b==inp['docs'][i]['b'];assert r.stem_body==' '.join(sorted(set().union(*(oldsets[k] for k in ks))))
  tokens+=len(u);repeats+=len(u)-len(set(u))
 dump(PUBLIC/'INPUT_VERIFICATION.json',dict(status='PASS',raw_articles_reread=n,canonical_rows_reconstructed=len(d),old_stem_exact_parity=len(d),ordered_tokens_reconstructed=tokens,repeated_occurrences_preserved=repeats,within_sentence_bigrams_only=True,fit_calls=0))
 print('Independent raw-source input verification PASS',flush=True)
if __name__=='__main__':main()
