"""Actual saved-weight historical inference for the offline course demonstration."""
from core import *
import argparse,semantic,__main__
__main__.SemanticLR=semantic.SemanticLR
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=B/'runs/v1');p.add_argument('--stock',choices=['AAPL','AMZN'],required=True);p.add_argument('--time',required=True);a=p.parse_args();r=a.run.resolve();verify(json.loads((r/'M08/hashes.json').read_text()));d=pd.read_pickle(r/'M08/features.pkl');t=pd.Timestamp(a.time)
if t.tzinfo is None:raise ValueError('An explicit UTC offset is required')
x=d[(d.symbol==a.stock)&(d.start_utc==t)]
if len(x)!=1 or x.split.iloc[0]=='train':raise ValueError('Choose one original development/exposed-replay window')
settings=json.loads((r/'M08/selected.json').read_text())[a.stock];dim=settings['fusion_dim'];price_model=joblib.load(r/'M08'/a.stock/'price/model.joblib');text_model=joblib.load(r/'M08'/a.stock/f'text{dim}/model.joblib');base=float(price_model.predict_proba(x)[0,1]);text=float(text_model.predict_proba(x)[0,1]);w=settings['text_weight'];print(json.dumps({'stock':a.stock,'target_start':str(t),'information_cutoff':str(x.cutoff_utc.iloc[0]),'price_probability':base,'text_probability':text,'text_weight':w,'fusion_probability':(1-w)*base+w*text,'label':int(x.label.iloc[0]),'news_record_keys':x.news_record_keys.iloc[0].split('|'),'status':'Historical replay, not live forecast; event quality pending'},indent=2))
