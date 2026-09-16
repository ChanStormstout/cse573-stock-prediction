from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,joblib
from scipy.special import expit
B=Path(__file__).resolve().parent;ROOT=B.parents[2];OLD=ROOT/'outputs/stock_integrated_4h'
sys.path.insert(0,str(OLD));import run as legacy
Transform=legacy.Transform  # Historical pickle was saved from run.py as __main__.
OUT=B/'addendum_v2';OUT.mkdir(exist_ok=False)
oof=pd.read_csv(OLD/'runs/v1/oof.csv'); rows=[];choices={}
for symbol,g in oof.groupby('symbol'):
    assert g.month.between('2018-03','2018-08').all()
    base=legacy.monthly(g.label,g.title,g.month.to_numpy())
    candidates=[]
    for kind in legacy.BRANCHES:
        score=legacy.monthly(g.label,g[kind],g.month.to_numpy())
        candidates.append(dict(symbol=symbol,branch=kind,**score,eligible=score['Brier']<=base['Brier']+.002))
    eligible=[r for r in candidates if r['eligible']]
    chosen=sorted(eligible,key=lambda r:(-r['BA'],r['Brier'],legacy.BRANCHES.index(r['branch'])))[0]['branch']
    choices[symbol]=chosen;rows.extend(candidates)
# Commit the training-only choice before loading any later scores.
(OUT/'choices_before_evaluation.json').write_text(json.dumps(choices,indent=2)+'\n')
pd.DataFrame(rows).to_csv(OUT/'single_branch_candidates.csv',index=False)
pred=pd.read_csv(OLD/'runs/v1/predictions.csv');pred=pred[pred.phase=='frozen'];scores=[]
for (symbol,period),g in pred.groupby(['symbol','split']):
    kind=choices[symbol];scores.append(dict(symbol=symbol,period=period,selected=kind,**legacy.metric(g.label,g[kind])))
pd.DataFrame(scores).to_csv(OUT/'single_branch_metrics.csv',index=False)

panel=pd.read_csv(B/'analysis_v1/case_panel_with_outcomes.csv')
d=pd.read_pickle(OLD/'runs/v1/inputs.pkl').set_index('key').loc[panel.key].reset_index()
assert d.symbol.eq('AMZN').all()
model_path=OLD/'runs/v1/models/AMZN_final_body.joblib';pack=joblib.load(model_path)
tf=pack['transform'];model=pack['classifier'];x=tf.transform(d,None,None).toarray();coef=model.coef_[0]
contrib=x*coef;intercept=float(model.intercept_[0]);p=expit(intercept+contrib.sum(axis=1))
np.testing.assert_allclose(p,panel.old_body.to_numpy(),rtol=0,atol=1e-12)
terms=tf.text.get_feature_names_out()[tf.keep];attributions=[]
for i,r in panel.iterrows():
    z=contrib[i,len(legacy.PRICE):];ids=np.argsort(-abs(z));top=[dict(stem=str(terms[j]),logit_contribution=float(z[j])) for j in ids if z[j]!=0][:5]
    attributions.append(dict(case_id=r.case_id,key=r.key,news_count=int(r.news_count),intercept=intercept,price_logit=float(contrib[i,:len(legacy.PRICE)].sum()),text_logit=float(z.sum()),probability=float(p[i]),top_text_contributions=top))
(OUT/'case_attributions.json').write_text(json.dumps(attributions,ensure_ascii=False,indent=2)+'\n')
pd.DataFrame([{k:v for k,v in r.items() if k!='top_text_contributions'} for r in attributions]).to_csv(OUT/'case_attributions.csv',index=False)
assert all(r['text_logit']==0 for r in attributions if r['news_count']==0)
files=[B/'ADDENDUM.md',Path(__file__),OLD/'runs/v1/oof.csv',OLD/'runs/v1/predictions.csv',model_path,B/'analysis_v1/case_panel_with_outcomes.csv']
(OUT/'verification.json').write_text(json.dumps(dict(new_training=False,training_only_choice=choices,probability_reload_max_error=float(np.max(abs(p-panel.old_body))),no_news_text_zero=True,hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}),indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(scores).to_string(index=False))
for r in attributions:
    if r['case_id'] in ['C03','C08','C11','C12','C15','C19']:
        print(json.dumps(r,ensure_ascii=False))
