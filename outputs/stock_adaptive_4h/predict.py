"""Actual frozen inference for known audited historical cutoffs, with labels removed."""
from core import *
import argparse,joblib

def predict(run,symbol,start):
    verification=json.loads((run/'verification.json').read_text())
    if verification['status']!='PASS':raise ValueError('Run has not passed verification')
    check_hashes(verification['inference_sources'])
    bundle=joblib.load(run/f'{symbol}_system.joblib')
    if sha(OLD/'prepared/articles.npz')!=bundle['embedding_sha256']:raise ValueError('Embedding fingerprint mismatch')
    # Historical audit contract: arbitrary new rows are not accepted by this demo.
    input_path=run/'inputs.pkl';source=pd.read_pickle(input_path)
    chosen=source[source.symbol.eq(symbol)&source.start_utc.eq(pd.Timestamp(start))]
    if len(chosen)!=1 or chosen.month.iloc[0]<'2018-09':raise ValueError('Choose one audited frozen-period timestamp with timezone')
    z=np.load(OLD/'prepared/articles.npz');r=predict_bundle(bundle,chosen.drop(columns=['label','target_return']),z['keys'],z['embeddings'].astype(float)).iloc[0]
    method=bundle['selected_method']
    return {'symbol':symbol,'cutoff':str(r.cutoff_utc),'start':str(r.start_utc),'end':str(r.end_utc),'horizon':'4h','baseline_probability':float(r.title),'base_probability':float(r.price),'body_probability':float(r.body),'semantic_probability':float(r.semantic),'static_probability':float(r.static),'gate_probability':float(r.gate),'selected_method':method,'selected_probability':float(r[method]),'weights':{n:float(r['weight_'+n]) for n in ['zero','body','semantic']},'gate_status':r.gate_reason,'news_count':int(r.news_count),'scope':'Frozen exploratory historical inference; labels removed before model call','quality_status':'Independent event review pending; original articles only'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--symbol',choices=['AAPL','AMZN'],required=True);p.add_argument('--start',required=True);a=p.parse_args();print(json.dumps(predict(a.run,a.symbol,a.start),indent=2,ensure_ascii=False))
