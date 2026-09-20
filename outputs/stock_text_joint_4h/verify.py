"""No-fit independent selection/member/reload verifier; never changes weights."""
from core import *
from unittest.mock import patch
from sklearn.metrics import balanced_accuracy_score,brier_score_loss
from threadpoolctl import threadpool_limits
import time,collections

def forbidden(*a,**kw):raise AssertionError('Verifier attempted a fit')
def main(block=None):
 checks={};nprotected=protect_check();checks['protected_artifacts_unchanged']=nprotected
 cache_check(json.loads((WORK/'training_seal.json').read_text()));checks['code_input_config_binding']=True
 d,*_=load();src=pd.read_pickle(WORK/'canonical.pkl')
 pd.testing.assert_frame_equal(src,d[src.columns]);checks['canonical_windows_labels_cutoffs']=len(d)
 inp=joblib.load(WORK/'inputs.joblib');docs=inp['docs'];emb,ids,means,gate=semantic(d)
 outer=read_csv(PRIVATE/'outer_folds.csv');inners=json.loads((PRIVATE/'inner_folds.json').read_text())
 rows=[];selections=[];errors=[];reloads=0;rhozero=0;innerrows=0;calchecks=0
 for seed in SEEDS:
  for stock in ['AAPL','AMZN']:
   for fold in range(10):
    name=f'{seed}_{stock}_{fold}'
    if block is not None and name!=block:continue
    root=WORK/'training'/name;base=PRIVATE/'baseline'/name
    complete=json.loads((root/'complete.json').read_text());assert complete['seal']==sha(WORK/'training_seal.json')
    for f,h in complete['files'].items():assert sha(root/f)==h
    fragment=WORK/'verification_fragments'/f'{name}.json'
    if block is None and fragment.exists():
     cached=json.loads(fragment.read_text())
     assert cached['verifier_sha256']==sha(__file__) and cached['complete_sha256']==sha(root/'complete.json') and cached['training_seal_sha256']==sha(WORK/'training_seal.json')
     rows.append(read_csv(root/'predictions.csv'));selections.extend(cached['selections']);errors.append(cached['max_probability_error']);reloads+=cached['models_reloaded'];rhozero+=cached['rhozero'];innerrows+=cached['innerrows'];calchecks+=cached['calchecks'];continue
    p=read_csv(root/'predictions.csv');old=read_csv(base/'outer_predictions_SEALED.csv')
    ev=outer[(outer.seed==seed)&(outer.symbol==stock)&(outer.fold==fold)]['index'].to_numpy(int)
    tr=sorted(set(np.flatnonzero(d.symbol==stock))-set(ev));assert p.row_id.tolist()==d.iloc[ev].row_id.tolist();assert np.array_equal(p.label,d.iloc[ev].label)
    for m in ['PAPER','PRICE','FULL','FINBERT','MODERN']:assert np.array_equal(p[m],old[m])
    records=read_csv(root/'selection.csv');ip=read_csv(root/'inner_predictions.csv');ledger=json.loads((root/'training.json').read_text())
    ss={s['inner_fold']:s for s in inners if (s['seed'],s['symbol'],s['fold'])==(seed,stock,fold)}
    baseoof=read_csv(base/'FULL_selection_oof_NOT_META_TRAIN.csv').set_index('index').p;priceoof=read_csv(base/'PRICE_selection_oof_NOT_META_TRAIN.csv').set_index('index').p
    for (m,c,rho,inf),g in ip.groupby(['method','C','rho','inner_fold'],sort=False):
     b=ss[inf]['validation'];assert g['index'].tolist()==b
     r=records[(records.method==m)&(records.C==c)&(records.rho==rho)&(records.inner_fold==inf)].iloc[0]
     assert abs(balanced_accuracy_score(d.iloc[b].label,g.p>=.5)-r.BA)<1e-12
     assert abs(brier_score_loss(d.iloc[b].label,g.p)-r.Brier)<1e-12
     innerrows+=len(g)
     if m.startswith('B') and rho==0:
      obj=joblib.load(base/f'FULL_inner{int(inf)}_{float(c)}.joblib');q=obj['model'].predict_proba(obj['transform'].transform(d,b,{}))[:,1];nn=d.iloc[b].has_news.to_numpy()==0;q[nn]=priceoof.loc[b].to_numpy()[nn]
      errors.append(float(np.max(abs(q-g.p))));rhozero+=len(g)
    for item in ledger:
     f=item['file'];o=joblib.load(root/f);m=o['method'];c=o['C'];rho=o['rho'];inf=o['inner_fold'];a=o['train'];b=o['evaluation']
     if inf is None:assert a==tr and b==ev.tolist()
     else:assert a==ss[inf]['train'] and b==ss[inf]['validation']
     assert not(set(a)&set(b)) and not(set(a)&set(ev))
     if m.startswith('A'):
      if m=='A3':
       expected=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(a)),d.iloc[a].label))
       for cal,(at,bt) in zip(o['model'].calibrated_classifiers_,expected):
        tf=cal.estimator.named_steps['text'];assert tf.fit_row_ids_==d.iloc[np.array(a)[at]].row_id.tolist();assert not(set(tf.fit_row_ids_)&set(d.iloc[np.array(a)[bt]].row_id));calchecks+=1
      else:assert o['model'].named_steps['text'].fit_row_ids_==d.iloc[a].row_id.tolist()
      q=o['model'].predict_proba([docs[i] for i in b])[:,1]
      nn=d.iloc[b].has_news.to_numpy()==0;fallback=old.PRICE.to_numpy() if inf is None else priceoof.loc[b].to_numpy();q[nn]=fallback[nn]
     else:
      tf=o['transform'];assert tf.train==a;assert tf.full.fit_ids==d.iloc[a].row_id.tolist()
      expectedarticles=sorted({j for i in a for j in ids[i]});assert tf.article_ids==expectedarticles
      assert len(tf.full.keep)<=500 and tf.full.cols==OLD
      # Independently check training-only scales and PCA training mean.
      np.testing.assert_allclose(tf.full.price_scale.mean,np.nanmean(d.iloc[a][OLD],axis=0),atol=1e-12,rtol=0)
      if tf.pca is not None:np.testing.assert_allclose(tf.pca.mean_,emb[expectedarticles].mean(0),atol=1e-6,rtol=0)
      z=tf.pca.transform(means) if tf.pca is not None else means.copy();z[~gate]=0
      np.testing.assert_allclose(tf.scale.mean,np.mean(z[a],axis=0),atol=1e-6,rtol=0)
      x=tf.transform(d,b,means,gate,rho)
      z0=tf.scale.transform(z[b]);z0[~gate[b]]=0
      np.testing.assert_allclose(x[:,-z0.shape[1]:].toarray(),rho*z0,atol=1e-12,rtol=0)
      q=o['model'].predict_proba(x)[:,1];fallback=old.FULL.to_numpy() if inf is None else baseoof.loc[b].to_numpy();q[~gate[b]]=fallback[~gate[b]]
     target=p[m].to_numpy() if inf is None else ip[(ip.method==m)&(ip.C==c)&(ip.rho==rho)&(ip.inner_fold==inf)].p.to_numpy()
     errors.append(float(np.max(abs(q-target))));reloads+=1
    # Independent parameter reconstruction; no call to trainer's choose helper.
    for m in ['A1','A2','A3','B1','B2']:
     rs=records[records.method==m];agg=rs.groupby(['C','rho'])[['BA','Brier']].mean().reset_index().sort_values(['BA','Brier','rho','C'],ascending=[False,True,True,True],kind='stable');chosen=agg.iloc[0]
     selected=[x for x in ledger if x['method']==m and x['inner_fold'] is None]
     if selected:assert selected[0]['C']==chosen.C and selected[0]['rho']==chosen.rho
     else:
      reuse=json.loads((root/(m+'_reuse.json')).read_text());assert chosen.rho==0 and chosen.C==reuse['C'];assert sha(ROOT/reuse['source'])==reuse['sha256'];assert np.array_equal(p[m],p.FULL);rhozero+=len(p)
     if m.startswith('B'):assert np.array_equal(p[m].to_numpy()[~gate[ev]],p.FULL.to_numpy()[~gate[ev]])
     nn=d.iloc[ev].has_news.to_numpy()==0;assert np.array_equal(p[m].to_numpy()[nn],p.PRICE.to_numpy()[nn])
     selections.append(dict(seed=seed,symbol=stock,fold=fold,method=m,C=chosen.C,rho=chosen.rho))
    s=json.loads((root/'shrink.json').read_text());ix=baseoof.index.to_numpy(int);bp=baseoof.to_numpy();has=d.iloc[ix].has_news.to_numpy().astype(bool)
    scores=[]
    for alpha in [.25,.5,.75,1.]:
     q=bp.copy();q[has]=.5+alpha*(bp[has]-.5);scores.append((float(np.mean((q-d.iloc[ix].label.to_numpy())**2)),-alpha))
    alpha=-min(scores)[1];assert alpha==s['selected'];q=old.FULL.to_numpy().copy();has=d.iloc[ev].has_news.to_numpy().astype(bool);q[has]=.5+alpha*(q[has]-.5)
    assert np.array_equal(q,p.FULL_SHRINK) and np.array_equal(q>=.5,old.FULL.to_numpy()>=.5)
    classical=read_csv(CLASSIC/name/'predictions.csv')
    for m in ['TFIDF_LR','TFIDF_SVM','TFIDF_RF']:
     assert np.array_equal(p[m],classical[m]);assert np.array_equal(p[m+'_PRICE_FALLBACK'],np.where(has,classical[m],old.PRICE))
    rows.append(p);print(name,'verified',flush=True)
 assert len(rows)==(60 if block is None else 1) and max(errors)<=1e-12
 predictions=pd.concat(rows,ignore_index=True)
 if block is not None:
  dump(WORK/'verification_fragments'/f'{block}.json',dict(status='PARTIAL_BLOCK_PASS',block=block,verifier_sha256=sha(__file__),complete_sha256=sha(root/'complete.json'),training_seal_sha256=sha(WORK/'training_seal.json'),models_reloaded=reloads,max_probability_error=max(errors),selections=selections,rhozero=rhozero,innerrows=innerrows,calchecks=calchecks,fit_calls=0));return
 assert len(predictions)==4821
 for seed,g in predictions.groupby('seed'):assert set(g.row_id)==set(d.row_id) and not g.row_id.duplicated().any()
 predictions.to_csv(PUBLIC/'predictions.csv',index=False,float_format='%.17g');pd.DataFrame(selections).to_csv(PUBLIC/'selection_records.csv',index=False)
 checks.update(inner_metrics_reconstructed=innerrows,checkpoint_models_reloaded=reloads,max_probability_error=max(errors),rho0_replayed_probabilities=rhozero,SVM_calibration_pipeline_memberships=calchecks,selection_reconstructed=True,exact_no_evidence_fallback=True,exact_no_news_fallback=True,shrink_preserves_direction=True,semantic_scale_then_rho=True,fit_calls=0)
 dump(PUBLIC/'VERIFICATION.json',dict(status='PASS',checks=checks,note='Independent no-fit membership/selection/reload verification, not independent retraining'))
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--block');args=ap.parse_args()
 with threadpool_limits(2),patch.object(LogisticRegression,'fit',forbidden),patch.object(LinearSVC,'fit',forbidden),patch.object(CalibratedClassifierCV,'fit',forbidden),patch.object(TfidfVectorizer,'fit_transform',forbidden),patch.object(PCA,'fit',forbidden):main(args.block)
