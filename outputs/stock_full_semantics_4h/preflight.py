"""Fit-free semantic input and existing base-OOF contracts."""
from train_semantics import *
def main():
 check_sources();d,*_=load();emb,ids,means=semantic_input(d,'FINBERT');inner=json.loads((PRIVATE/'inner_folds.json').read_text());outer=pd.read_csv(PRIVATE/'outer_folds.csv');checks=[];article_counts=[]
 with np.load(SRC/'FINBERT_articles.npz') as z:array_parity=np.array_equal(z['vectors'],emb)
 for seed in SEEDS:
  for stock in ['AAPL','AMZN']:
   for fold in range(10):
    root=PRIVATE/'baseline'/f'{seed}_{stock}_{fold}';base=pd.read_csv(root/'FULL_selection_oof_NOT_META_TRAIN.csv').set_index('index');splits=[s for s in inner if (s['seed'],s['symbol'],s['fold'])==(seed,stock,fold)];ev=outer[(outer.seed==seed)&(outer.symbol==stock)&(outer.fold==fold)]['index'].to_numpy();validation=[i for s in splits for i in s['validation']]
    checks.append(len(validation)==len(set(validation)) and set(validation)==set(base.index) and len(splits)==3)
    for s in splits:
     a=s['train'];b=s['validation'];n=len({j for i in a for j in ids[i]});article_counts.append(n);checks.append(not(set(a)&set(b)) and not(set(a)&set(ev)) and n>=16 and d.iloc[a].label.nunique()==2)
 result={'status':'PASS' if all(checks) and array_parity and np.isfinite(emb).all() and np.isfinite(means).all() else 'FAIL','rows':len(d),'outer_blocks':60,'inner_splits':len(inner),'minimum_training_article_pairs_for_PCA16':min(article_counts),'price_feature_count':len(OLD),'finite_FinBERT_vectors':bool(np.isfinite(emb).all()),'materialized_array_exact_parity':bool(array_parity),'fold_and_base_oof_checks_passed':sum(checks),'fold_and_base_oof_checks_total':len(checks),'fit_calls':0,'Modern_status':'COMPLETE' if (PUBLIC/'MODERN_ENCODING.json').exists() else 'ENCODING_IN_PROGRESS','code_sha256':sha(__file__)}
 assert result['status']=='PASS';dump(PUBLIC/'SEMANTIC_PREFLIGHT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
