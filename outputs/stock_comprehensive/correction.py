"""M07 offset-logistic correction; no training without independent field approval."""
from core import *
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

class OffsetCorrection:
    def __init__(self,C=.1):self.C=C
    def fit(self,z,y,base,gate):
        z=np.asarray(z,dtype=float);gate=np.asarray(gate,dtype=bool)
        if z.shape[1]>20:raise ValueError('At most20 event features')
        if gate.sum()<60:raise ValueError('Insufficient event windows')
        self.scaler=StandardScaler(with_mean=False).fit(z[gate]);x=np.c_[self.scaler.transform(z[gate]),np.ones(gate.sum())];y=np.asarray(y)[gate];offset=logit(np.clip(np.asarray(base)[gate],1e-6,1-1e-6))
        def objective(w):
            eta=offset+x@w;loss=np.logaddexp(0,eta).sum()-y@eta+.5*np.sum(w*w)/self.C;grad=x.T@(expit(eta)-y)+w/self.C;return loss,grad
        fit=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
        if not fit.success:raise RuntimeError(fit.message)
        self.w=fit.x;return self
    def predict(self,z,base,gate):
        p=np.asarray(base,dtype=float).copy();gate=np.asarray(gate,dtype=bool)
        if gate.any():p[gate]=expit(logit(np.clip(p[gate],1e-6,1-1e-6))+np.c_[self.scaler.transform(np.asarray(z)[gate]),np.ones(gate.sum())]@self.w)
        # No-event probabilities exactly retain original floating point values.
        return p

def gate_check(review,nonduplicate_events,event_windows):
    if not review.get('independent_approved',False):return 'pending_independent_review'
    if nonduplicate_events<30 or event_windows<60:return 'insufficient_events_or_windows'
    return 'eligible'

def train_from_oof(z,y,base,gate,dates,review,groups):
    status=gate_check(review,len(set(np.asarray(groups)[gate])),sum(gate))
    if status!='eligible':return None,{'status':status}
    rows=[]
    for C in [.01,.1,1]:
        for mo in [6,7,8]:
            lo=pd.Timestamp(f'2018-{mo:02d}-01',tz='UTC');hi=lo+pd.DateOffset(months=1);a=np.asarray(dates<lo);b=np.asarray((dates>=lo)&(dates<hi))
            if sum(np.asarray(gate)[a])<60:continue
            m=OffsetCorrection(C).fit(z[a],np.asarray(y)[a],base[a],gate[a]);rows.append(dict(C=C,month=mo,**metrics(np.asarray(y)[b],m.predict(z[b],base[b],gate[b]))))
    if not rows:return None,{'status':'insufficient_inner_event_windows'}
    grid=pd.DataFrame(rows);chosen=float(grid.groupby('C')[['balanced_accuracy','brier']].mean().sort_values(['balanced_accuracy','brier'],ascending=[False,True]).index[0]);return OffsetCorrection(chosen).fit(z,y,base,gate),{'status':'trained','C':chosen,'grid':rows}

def run(out,runroot):
    quality=json.loads((runroot/'M06/quality_gate.json').read_text());events=pd.read_json(runroot/'M06/events.jsonl',lines=True);d=load(runroot);statuses={}
    for stock,s in d[d.split.eq('train')].groupby('symbol'):
        train_keys={k for x in s.news_record_keys.fillna('') for k in x.split('|') if k};accepted=events[events.symbol.eq(stock)&events.extraction_status.eq('provisional')&events.record_key.isin(train_keys)];keysset=set(accepted.record_key);g=s.news_record_keys.fillna('').apply(lambda x:bool(keysset.intersection(x.split('|'))));statuses[stock]=dict(status=gate_check({'independent_approved':False},accepted.event_group.nunique(),int(g.sum())),provisional_unique_groups=accepted.event_group.nunique(),training_event_windows=int(g.sum()))
    dump(out/'decision.json',{'stocks':statuses,'formal_gate':quality,'training_performed':False,'reason':'Independent review is a required condition of the approved plan, not a permissions request.'})
