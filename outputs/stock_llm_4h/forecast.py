"""Quality-gated four-hour fact correction; original price+text is immutable.
Model fit is implemented but deliberately cannot run without verified review records.
"""
from common import *
import datetime as dt
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit

def instant(value):
 x=dt.datetime.fromisoformat(str(value).replace('Z','+00:00'))
 if x.tzinfo is None:raise ValueError('Timezone required')
 return x

def review_gate(reviews,expected,required_types=('rating','target_price')):
 if set(r['id'] for r in reviews)!=set(expected):raise ValueError('Review input set mismatch')
 accepted=[];details={}
 for kind in required_types:
  rs=[]
  for r in reviews:
   if r['input_sha256']!=expected[r['id']]:raise ValueError('Review fingerprint mismatch')
   if 'payload' in r and digest(json.dumps(json.loads(r['payload']),sort_keys=True).encode())!=r['input_sha256']:raise ValueError('Review payload mismatch')
   if r.get('kind')!=kind:continue
   if not r.get('reviewer','').strip():continue
   if re.search('assistant|codex|chatgpt|gpt',r['reviewer'],re.I):continue
   if r.get('critical_correct') not in (0,1) or r.get('missed') not in (0,1) or r.get('uncertain') not in (0,1) or r.get('event_present') not in (0,1):continue
   rs.append(r)
  grouped={}
  for r in rs:grouped.setdefault(r['event_group'],[]).append(r)
  # One event group one vote, all checked records must be correct. Duplicates cannot add denominator.
  n=len(grouped);correct=sum(all(x['critical_correct']==1 and x['uncertain']==0 for x in g) for g in grouped.values())
  positive_n=sum(any(x['event_present']==1 for x in g) for g in grouped.values())
  details[kind]={'independent_groups':n,'positive_groups':positive_n,'correct_groups':correct,'critical_accuracy':correct/n if n else None,'missed':sum(x['missed'] for x in rs)}
  if n>=30 and positive_n>=30 and correct/n>=.9:accepted.append(kind)
 return {'accepted_types':accepted,'by_type':details,'status':'PASSED_LIMITED_TYPES' if set(required_types)<=set(accepted) else 'WAITING_OR_FAILED','identity_note':'Reviewer identity is declared, not externally authenticated.'}

def aggregate(events,symbol,cutoff,allowed_keys,approved_types,extractor_frozen_at):
 cutoff=instant(cutoff)
 if instant(extractor_frozen_at)>cutoff:raise ValueError('Extractor adapted using later information')
 z=np.zeros(12);total=0.;seen=set()
 for r in events:
  if r['symbol']!=symbol or r['record_key'] not in allowed_keys:continue
  if instant(r['available_utc'])>cutoff:raise ValueError('Future news')
  e=r['event']
  if e['kind'] not in approved_types:continue
  key=(r['event_group'],)+signature(e)
  if key in seen:continue
  seen.add(key)
  pub=instant(r['published_utc']) if r.get('published_utc') else None
  if pub and pub>instant(r['available_utc']):raise ValueError('Publication after availability')
  # Unknown disclosure age receives explicit unknown feature, not an invented date.
  age=max(0,(cutoff-pub).total_seconds()/3600) if pub else None
  weight=2**(-age/4) if age is not None else .5
  f=np.zeros(12);idx=0 if e['kind']=='rating' else 4
  action={'raise':0,'lower':1,'maintain':2,'initiate':3}.get(e['action'])
  if action is not None:f[idx+action]=1
  if e['kind']=='target_price' and e['old'] and e['new']:
   old=float(e['old'].replace(',',''));new=float(e['new'].replace(',',''))
   if old>0:f[8]=np.clip(new/old-1,-1,1)
  f[9]=1;f[10]=float(age is None);f[11]=float(e['action']=='unknown')
  z+=weight*f;total+=weight
 return z/(1+total),bool(seen)

class FactCorrection:
 def fit(self,z,past_base_probability,y,gate,train_end,evaluation_cutoff,quality,nonduplicate_events,C=.1,ledger=None):
  if quality['status']!='PASSED_LIMITED_TYPES':raise ValueError('Independent quality not passed')
  if ledger is None:raise ValueError('Past-only OOF provenance ledger required')
  validate_oof_ledger(ledger,evaluation_cutoff)
  if instant(train_end)>=instant(evaluation_cutoff):raise ValueError('Unmatured future label')
  z=np.asarray(z,float);g=np.asarray(gate,bool)
  if z.shape[1]>20 or nonduplicate_events<30 or g.sum()<60:raise ValueError('Insufficient unique events/event windows or feature budget')
  o=logit(np.clip(past_base_probability,1e-8,1-1e-8));y=np.asarray(y,float);self.scale=np.maximum(np.std(z,axis=0),1e-6);x=(z/self.scale)*g[:,None]
  def obj(w):
   t=o+x@w
   return np.logaddexp(0,t).sum()-y@t+(w@w)/(2*C),x.T@(expit(t)-y)+w/C
  r=minimize(obj,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B')
  if not r.success:raise ValueError(r.message)
  self.coef=r.x;return self
 def predict(self,z,p0,gate,weight=1.):
  if weight not in (0,.25,.5,1):raise ValueError('Unregistered weight')
  p=np.asarray(p0,dtype=float).copy();g=np.asarray(gate,bool)
  if weight==0 or not g.any():return p
  delta=(np.asarray(z)/self.scale)@self.coef
  p[g]=expit(logit(np.clip(p[g],1e-8,1-1e-8))+weight*np.clip(delta[g],-1,1));return p

def validate_oof_ledger(ledger,evaluation_cutoff):
 boundary=instant(evaluation_cutoff)
 for r in ledger:
  c=instant(r['cutoff_utc'])
  if instant(r['base_training_end'])>=c:raise ValueError('Base prediction is not past-only OOF')
  if instant(r['extractor_frozen_at'])>c:raise ValueError('Extractor future adaptation')
  if instant(r['end_utc'])>=boundary:raise ValueError('Future training target')
  if r.get('used_for_base_fit',False):raise ValueError('In-sample base probability')
 return True
