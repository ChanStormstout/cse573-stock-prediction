"""Evidence audit after the four independent stage verifiers. No fitting."""
from models import *
import ast,subprocess

def main():
 results={stage:json.loads((OUT/f'{stage}_VERIFICATION.json').read_text()) for stage in ['lexical','kernel','table','residual']};assert all(r['status']=='PASS' for r in results.values())
 protected=json.loads((LOCAL/'protected.json').read_text());bad=[p for p,h in protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h];assert not bad,bad
 # Prove the added scheduling wrapper has the same execution body except lock scope.
 code=Path(__file__).parent;original=(code/'run.py').read_text();wrapper=(code/'run_stage.py').read_text();assert wrapper[wrapper.index('def main():'):]==original[original.index('def main():'):].replace("LOCAL/'run.lock'","LOCAL/(args.stage+'.lock')")
 maximum=0;convergence=[];weightpaths=[];baseparity=[];margin_errors=[]
 with threadpool_limits(2):
  for stage in results:
   for root in sorted((LOCAL/stage).iterdir()):
    if not root.is_dir():continue
    for p in root.glob('*.joblib'):
     obj=joblib.load(p)
     if obj['type']=='svm':
      for cal in obj['model'].calibrated_classifiers_:
       mod=cal.estimator.named_steps['classifier']
       if stage=='kernel':
        from kernel_fast import ORIGINAL
        xx=cal.estimator.named_steps['features'].transform(np.array(obj['evaluation'])[:,None]);fast=mod.decision_function(xx);slow=ORIGINAL(mod,xx.toarray());err=float(np.max(abs(fast-slow)));assert err<1e-10; margin_errors.append(err)
       n=int(np.max(mod.n_iter_));maximum=max(maximum,n);convergence.append(dict(stage=stage,block=root.name,method=p.stem,iterations=n,limit=mod.max_iter,converged=bool(n<mod.max_iter)))
     if obj['type']=='table' and 'TABPFN' in p.stem:
      path=Path(obj['model'].path);assert path.is_file() and path.parent==root;weightpaths.append(str(path.relative_to(ROOT)))
    if stage=='lexical':
     got=read(root/'predictions.csv');old=read(previous.CLASSIC/root.name/'predictions.csv');assert got.row_id.tolist()==old.row_id.tolist();baseparity.append(float(np.max(abs(got.UNI_DIRECT-old.TFIDF_SVM))))
 assert all(r['converged'] for r in convergence),'Unconverged selected classifier'
 pd.DataFrame(convergence).to_csv(OUT/'convergence.csv',index=False)
 dump(OUT/'FINAL_AUDIT.json',dict(status='PASS',protected_files=len(protected),protected_mismatches=len(bad),all_stage_verifiers='PASS',max_replay_error=max(r['max_probability_error'] for r in results.values()),selected_classifier_convergence_checks=len(convergence),max_iterations=maximum,TabPFN_weight_files_checked=len(weightpaths),kernel_original_decision_checks=len(margin_errors),kernel_original_decision_max_error=max(margin_errors),old_unigram_control_max_direct_probability_difference=max(baseparity),scheduling_wrapper_exact_body_parity=True,new_encoder_calls=0,new_LLM_calls=0,model_selection_using_outer_results=False,independent_human_extraction_review=False,interpretation='EXPOSED EXPLORATORY RANDOM HISTORICAL BACKTEST'))
 dump(OUT/'EXECUTABLES.json',{str(p.relative_to(ROOT)):sha(p) for p in code.glob('*.py')})
 print('Final evidence audit PASS',flush=True)
if __name__=='__main__':main()
