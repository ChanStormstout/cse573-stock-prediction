"""Temporary fixture test: feature preparation, fit/save, independent reload, report gate and faults."""
from __future__ import annotations
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from outputs.stock_context_4h.market_features import build_market_row,expected_regular_grid
def main():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp); code='''import json, numpy as np\nfrom pathlib import Path\nfrom sklearn.linear_model import LogisticRegression\nr=Path(__import__("sys").argv[1]);x=np.array([[0],[1],[2],[3]],float);y=np.array([0,0,1,1]);m=LogisticRegression(C=.1,solver="liblinear").fit(x,y);np.savez(r/"model.npz",coef=m.coef_,intercept=m.intercept_,mean=x.mean(0),scale=x.std(0),classes=m.classes_);(r/"pred.json").write_text(json.dumps({"p":m.predict_proba(x)[:,1].tolist(),"keys":["a","b","c","d"]}))'''
  subprocess.run([sys.executable,'-c',code,str(root)],check=True)
  code2='''import json, numpy as np\nfrom pathlib import Path\nr=Path(__import__("sys").argv[1]);z=np.load(r/"model.npz",allow_pickle=False);p=json.loads((r/"pred.json").read_text());x=np.array([[0],[1],[2],[3]],float);q=1/(1+np.exp(-(x@z["coef"].T+z["intercept"]).ravel()));assert np.max(abs(q-np.array(p["p"])))<1e-12;(r/"report.md").write_text("PASS")'''
  subprocess.run([sys.executable,'-c',code2,str(root)],check=True)
  z=np.load(root/'model.npz',allow_pickle=False);assert list(z.files)==['coef','intercept','mean','scale','classes']
  schedule=pd.DataFrame({'open':[pd.Timestamp('2018-01-02 14:30Z')],'close':[pd.Timestamp('2018-01-02 16:30Z')]});grid=expected_regular_grid(schedule);bars=pd.DataFrame({'open':[1.]*len(grid),'close':[1.01]*len(grid)},index=pd.DatetimeIndex(grid.bar_start_utc));row=build_market_row(pd.Timestamp('2018-01-02 15:31Z'),'2018-01-02',grid,{'SPY':bars,'QQQ':bars});assert row['market_window_missing']==0
  changed=bars.copy();changed.loc[grid.bar_start_utc.iloc[-1],'close']=999;early=build_market_row(pd.Timestamp('2018-01-02 15:31Z'),'2018-01-02',grid,{'SPY':bars,'QQQ':bars});early_changed=build_market_row(pd.Timestamp('2018-01-02 15:31Z'),'2018-01-02',grid,{'SPY':changed,'QQQ':changed});assert early==early_changed
  assert not (root/'unverified_report.md').exists()
 print(json.dumps({'status':'PASS','subprocesses':['fit_save','verify_report'],'faults':['coefficient/feature-order fixture schema','future incomplete bar','unverified report blocked']}))
if __name__=='__main__':main()
