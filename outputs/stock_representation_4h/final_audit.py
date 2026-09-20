"""Unchanged historical artifacts, base-probability parity and convergence."""
from engine import *
def main():
 seal_check();check_sources();inp=joblib.load(WORK/'inputs.joblib');protected=json.loads((WORK/'protected.json').read_text())
 for p,h in protected.items():assert sha(ROOT/p)==h,'Historical artifact changed '+p
 rows=[];worst=0;count=0
 for stage in ['preprocess','grouped','aggregation']:
  assert json.loads((PUB/(stage+'_VERIFICATION.json')).read_text())['status']=='PASS'
  for b in inp['grouped'] if stage=='grouped' else inp['random']:
   root=WORK/stage/b['name'];out=read(root/'predictions.csv');base=PRIVATE/('diagnostic_baseline' if stage=='grouped' else 'baseline')/b['name'];prior=read(base/('predictions_SEALED.csv' if stage=='grouped' else 'outer_predictions_SEALED.csv'))
   assert out.row_id.tolist()==prior.row_id.tolist()
   for m in ['PRICE','PAPER','FULL','FINBERT','MODERN']:np.testing.assert_array_equal(out[m],prior[m])
   for r in json.loads((root/'training.json').read_text()):
    o=joblib.load(root/r['file']);model=o['model']
    classifiers=[c.estimator.steps[-1][1] for c in model.calibrated_classifiers_] if stage!='aggregation' else [model]
    for clf in classifiers:
     n=int(np.max(clf.n_iter_));assert n<clf.max_iter,'Estimator reached iteration limit';worst=max(worst,n);count+=1
  rows.append(dict(stage=stage,blocks=60,base_probabilities_exact=True))
 assert count==11400
 llm=PRIVATE/'analogy/generated.jsonl';assert sha(llm)=='c37cbc45187fd6cbd86a3144cd06e294963df509088471fc57664e817e8d8ae7'
 dump(PUB/'FINAL_AUDIT.json',dict(status='PASS',historical_files_unchanged=len(protected),classifier_convergence_checked=count,max_iterations=worst,prior_probability_parity=rows,LLM_rows=1972,LLM_unchanged=True,fit_calls=0,source_seal_pass=True))
 print('Final preservation/source-parity/convergence audit PASS',flush=True)
if __name__=='__main__':main()
