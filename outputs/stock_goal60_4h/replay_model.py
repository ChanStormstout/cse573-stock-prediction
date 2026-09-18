"""Reload a private saved classifier/interaction without retraining it."""
import argparse
from core import *
from text_models import TextTransform,Interaction,STATE
from history_models import HistoryTransform
import torch

def replay(method,month,symbol,seed=573):
 d=data();folder=PRIVATE/method;record=json.loads((folder/'done.json').read_text());c=next(r['C'] for r in record['choices'] if r['month']==month);tr,ev=split(d,symbol,month)
 suffix=f'_{seed}' if method.startswith(('T_','F1','F2')) else '';path=folder/f'model_{tr[-1]}_{c}{suffix}.joblib';b=joblib.load(path)
 if isinstance(b,dict):
  x=b['transform'].transform(d,ev)
  if 'weights' in b:
   m=Interaction(b['input_dim']);m.load_state_dict({k:torch.tensor(v) for k,v in b['weights'].items()});m.eval()
   with torch.no_grad():p=torch.sigmoid(m(torch.tensor(x,dtype=torch.float32),torch.tensor(b['transform'].state.transform(d.iloc[ev][STATE]),dtype=torch.float32))).numpy()
  else:p=b['model'].predict_proba(x)[:,1]
 else:
  x=d[OLD+RECENT].to_numpy(float)
  if 'cross' in method:x=np.c_[x,pd.read_pickle(PRIVATE/'cross.pkl').to_numpy()]
  p=b.predict_proba(x[ev])[:,1]
 if method.startswith(('T_','F1','F2')):
  reference=np.load(folder/f'seeds_{tr[-1]}_{c}.npz');j=list(reference['seeds']).index(seed);expected=reference['probabilities'][j]
 else:expected=np.load(folder/f'{symbol}_{month}_{c}_pred.npz')['raw']
 error=float(abs(p-expected).max());assert error<1e-6;print(json.dumps(dict(method=method,symbol=symbol,month=month,seed=seed,n=len(p),max_error=error)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('method');p.add_argument('--month',default='final');p.add_argument('--symbol',default='AMZN');p.add_argument('--seed',type=int,default=573);a=p.parse_args();replay(a.method,a.month,a.symbol,a.seed)
