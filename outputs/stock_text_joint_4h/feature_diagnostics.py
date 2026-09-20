"""Describe retained feature types, never use outer outcomes to select features."""
from core import *

def main():
 assert json.loads((PUBLIC/'VERIFICATION.json').read_text())['status']=='PASS'
 rows=[];shrinkrows=[]
 for root in sorted((WORK/'training').iterdir()):
  seed,symbol,fold=root.name.split('_');ledger=json.loads((root/'training.json').read_text())
  s=json.loads((root/'shrink.json').read_text());shrinkrows.append(dict(seed=int(seed),symbol=symbol,fold=int(fold),selected_a=s['selected']))
  for item in ledger:
   if item['inner_fold'] is not None:continue
   bundle=joblib.load(root/item['file']);m=item['method'];model=bundle['model']
   if m.startswith('A'):
    pipelines=[cc.estimator for cc in model.calibrated_classifiers_] if m=='A3' else [model]
    for j,p in enumerate(pipelines):
     terms=p.named_steps['text'].get_feature_names_out().tolist()
     rows.append(dict(seed=int(seed),symbol=symbol,fold=int(fold),method=m,calibration_model=j,terms=len(terms),bigrams=sum(' ' in w for w in terms),numeric_terms=sum(bool(re.search(r'\d',w)) for w in terms),negation_terms=sum(bool(set(w.split())&{'not','no','nor'}) for w in terms),C=item['C'],rho=item['rho'],semantic_dimensions=0))
   else:
    tf=bundle['transform'];rows.append(dict(seed=int(seed),symbol=symbol,fold=int(fold),method=m,calibration_model=-1,terms=len(tf.full.keep),bigrams=0,numeric_terms=0,negation_terms=0,C=item['C'],rho=item['rho'],semantic_dimensions=16 if m=='B1' else 768))
  for m in ['B1','B2']:
   if (root/(m+'_reuse.json')).exists():
    r=json.loads((root/(m+'_reuse.json')).read_text());rows.append(dict(seed=int(seed),symbol=symbol,fold=int(fold),method=m,calibration_model=-1,terms=None,bigrams=0,numeric_terms=0,negation_terms=0,C=r['C'],rho=0.,semantic_dimensions=0))
 pd.DataFrame(rows).to_csv(PUBLIC/'feature_diagnostics.csv',index=False);pd.DataFrame(shrinkrows).to_csv(PUBLIC/'shrink_selections.csv',index=False)
 print('Feature diagnostics complete; no fits')
if __name__=='__main__':main()
