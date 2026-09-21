import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np,pandas as pd
from run import fit_model

def main():
 n=80;r=np.random.default_rng(573)
 d=pd.DataFrame({'F1':np.full(n,.6),'label':r.integers(0,2,n),'symbol':np.where(np.arange(n)%2,'AAPL','AMZN')})
 raw={b:pd.DataFrame({b:r.normal(size=n)}) for b in ['D','U','A','L']}
 gates={b:np.zeros(n) for b in raw};gates['D'][:40]=1
 p,a=fit_model(d,raw,gates,'D',.1,np.arange(60),np.arange(60,80))
 assert np.array_equal(p,d.F1.iloc[60:80].to_numpy())
 gates['D'][60:70]=1
 p,a=fit_model(d,raw,gates,'D',.1,np.arange(60),np.arange(60,80))
 assert np.array_equal(p[10:],d.F1.iloc[70:80].to_numpy())
 print('PASS exact gated fallback')
if __name__=='__main__':main()
