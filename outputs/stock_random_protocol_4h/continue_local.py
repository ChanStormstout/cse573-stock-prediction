"""Bounded continuation of the already running local inference, no tuning."""
import json,subprocess,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=HERE/'v1';PRIVATE=ROOT/'work/stock-data/random_protocol_4h/v1'
def state(stage,status,**kw):
 (OUT/'CONTINUATION_STATUS.json').write_text(json.dumps(dict(stage=stage,status=status,**kw),indent=2)+'\n')
def run(python,script):
 with (PRIVATE/(script+'.log')).open('x') as log:
  subprocess.run([str(python),str(HERE/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
def main():
 started=time.monotonic();state('waiting_existing_analogy_inference','RUNNING')
 while not (PRIVATE/'analogy/inference_complete.json').exists():
  log=PRIVATE/'inference.log'
  if log.exists() and 'Traceback (most recent call last)' in log.read_text():raise RuntimeError('existing inference failed; outputs preserved, no restart')
  if time.monotonic()-started>8*3600:raise TimeoutError('existing inference not complete within eight hours; no rerun started')
  time.sleep(10)
 py=ROOT/'work/stock-data/finbert-env/bin/python3';mlx=ROOT/'work/stock-data/structured-env/bin/python3'
 state('verify_analogy_inputs','RUNNING');run(py,'verify_retrieval.py')
 state('verify_llm_outputs','RUNNING');run(py,'verify_inference.py')
 state('fit_free_shared_analogy_selection','RUNNING');run(py,'analogy_evaluate.py')
 state('fact_relation_quality_pilot','RUNNING');run(mlx,'fact_pilot_infer.py')
 state('quality_review_and_remaining_full_pipeline','PENDING_REVIEW',outer_metrics_released=False)
if __name__=='__main__':
 try:main()
 except Exception as e:
  state('preserved_failure','FAILED',reason=repr(e));raise
