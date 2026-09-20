"""Selected-model lexical retention and case-margin diagnostics, no fitting."""
from engine import *
def main():
 assert json.loads((PUB/'preprocess_VERIFICATION.json').read_text())['status']=='PASS'
 inp=joblib.load(WORK/'inputs.joblib');d,*_=load();rows=[];case_details=[]
 for b in inp['random']:
  root=WORK/'preprocess'/b['name'];old=joblib.load(CLASSIC/b['name']/'TFIDF_SVM.joblib')
  models={'CONTROL':old}
  for m in VARIANTS:models[m]=joblib.load(root/(m+'_selected.joblib'))['model']
  for m,model in models.items():
   for ci,cl in enumerate(model.calibrated_classifiers_):
    v=cl.estimator.named_steps['text'] if m!='CONTROL' else cl.estimator.steps[0][1];names=v.get_feature_names_out();lr=cl.estimator.steps[-1][1]
    rows.append(dict(block=b['name'],symbol=b['stock'],seed=b['seed'],method=m,calibration_fold=ci,features=len(names),numeric_features=int(sum(t.startswith('num_') for t in names)),negation_features=int(sum(t in {'no','not','nor'} for t in names)),coefficient_norm=float(np.linalg.norm(lr.coef_)),max_iterations=int(lr.n_iter_)))
 pd.DataFrame(rows).to_csv(PUB/'feature_diagnostics.csv',index=False)
 # Fixed case panel: retain only model margin contributions privately.
 panel=read(PUB/'case_index.csv');panel=panel[panel.stage=='preprocess'];byid=d.reset_index().set_index('row_id');outer=read(PRIVATE/'outer_folds.csv')
 for r in panel.itertuples():
  i=int(byid.loc[r.row_id,'index']);fold=int(outer[(outer.seed==573)&(outer['index']==i)].fold.iloc[0]);name=f'573_{r.symbol}_{fold}'
  obj=joblib.load(WORK/'preprocess'/name/(r.method+'_selected.joblib'));model=obj['model'];doc=inp['documents'][r.method][i];contrib={}
  for cl in model.calibrated_classifiers_:
   v=cl.estimator.named_steps['text'];x=v.transform([doc]);coef=cl.estimator.steps[-1][1].coef_[0];names=v.get_feature_names_out()
   for j,val in zip(x.indices,x.data):contrib[names[j]]=contrib.get(names[j],0)+float(val*coef[j])/3
  strongest=sorted(contrib.items(),key=lambda kv:-abs(kv[1]))[:12]
  case_details.append(dict(row_id=r.row_id,method=r.method,category=r.category,margin_contributions=strongest,interpretation='average uncalibrated SVM margin contribution; not probability attribution or causal effect'))
 dump(WORK/'case_margin_details.json',case_details)
 p=read(PUB/'preprocess_selected_C.csv');old={}
 for block in p.block.unique():old[block]=next(r['selected'] for r in json.loads((CLASSIC/block/'models.json').read_text()) if r['method']=='TFIDF_SVM')
 p['CONTROL_C']=p.block.map(old);p['changed_C']=p.C!=p.CONTROL_C;p.to_csv(PUB/'preprocess_C_comparison.csv',index=False)
 print('Feature diagnostics saved; raw term contributions remain private',flush=True)
if __name__=='__main__':main()
