"""Sequential reproducible entry point; completed matching stages resume safely."""
from pathlib import Path
import os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
python=ROOT/'work/stock-data/finbert-env/bin/python'
steps=[('core.py',[]),('tabular.py',[]),('encode_dense.py',[]),('text_models.py',[]),('text_models.py',['T_original','T_A1']),('history.py',['prepare']),('history.py',['encode']),('history_models.py',[]),('report.py',[]),('checks.py',[]),('verify_history.py',[]),('finalize.py',[])]
if __name__=='__main__':
 for script,args in steps:
  env=dict(os.environ)
  if script=='tabular.py':env.update(PYTHONPATH=str(ROOT/'work/stock-data/goal60_4h/runtime'),TABPFN_DISABLE_TELEMETRY='1')
  subprocess.run([str(python),str(HERE/script)]+args,cwd=ROOT,env=env,check=True)
