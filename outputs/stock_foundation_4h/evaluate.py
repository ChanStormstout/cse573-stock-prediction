"""Freeze train-only gate decisions, then publish exposed outcomes and intervals."""
import json,hashlib
import numpy as np,pandas as pd
from experiment import ROOT,HERE,PRIVATE,OUT,dump,metric,sha,summarize
import report as legacy_report

def gate(d,new,base):
 rows=[]
 for (sym,month),g in d[d.phase=='train_forward_oof'].groupby(['symbol','month']):
  a=metric(g.label,g[new]);b=metric(g.label,g[base]);rows.append(dict(symbol=sym,month=month,BA=a['BA']-b['BA'],Brier=a['Brier']-b['Brier']))
 r=pd.DataFrame(rows);per=r.groupby('symbol')[['BA','Brier']].mean();monthly=r.groupby('month').BA.mean();passed=len(monthly)>=3 and (monthly>0).sum()>=2 and r.BA.mean()>=.01 and per.BA.min()>=-.01 and per.Brier.max()<=.002 and r.Brier.mean()<=.002
 return dict(method=new,baseline=base,months=len(monthly),positive_months=int((monthly>0).sum()),BA_delta=float(r.BA.mean()),Brier_delta=float(r.Brier.mean()),stocks=per.to_dict('index'),passed=bool(passed))

def fuse(d,method):
 out='FUSE_'+method;records=[]
 for month in sorted(d.loc[d.phase=='train_forward_oof','month'].unique())+['final']:
  past=d[(d.phase=='train_forward_oof')&(d.month<('2018-09' if month=='final' else month))];scores=[]
  for w in [0,.25,.5,.75]:
   rows=[]
   for _,g in past.groupby(['symbol','month']):rows.append(metric(g.label,(1-w)*g.F1+w*g[method]))
   score=pd.DataFrame(rows);scores.append((w,float(score.BA.mean()) if len(score) else 0.,float(score.Brier.mean()) if len(score) else 0.))
  # Preregistered weights/global past-only selection; BA primary, Brier tie-break.
  w=sorted(scores,key=lambda a:(-a[1],a[2],a[0]))[0][0];mask=d.phase.isin(['development','later']) if month=='final' else d.month==month;d.loc[mask,out]=(1-w)*d.loc[mask,'F1']+w*d.loc[mask,method]
  records.append(dict(method=out,month=month,weight=w,selection_months=sorted(past.month.unique().tolist()),candidates=scores))
 return out,records

def main():
 path=OUT/'selection.json'
 if path.exists():raise FileExistsError('selection already frozen')
 d=pd.read_csv(OUT/'aggregation_predictions.csv');m=pd.read_csv(OUT/'modern_predictions.csv').set_index('key');c=pd.read_csv(OUT/'chronos_predictions.csv').set_index('key');d['MODERN']=d.key.map(m.MODERN)
 for col in ['HISTORY_LR','CHRONOS_LR']:d[col]=d.key.map(c[col])
 gates=[gate(d,'MODERN','F2'),gate(d,'CHRONOS_LR','HISTORY_LR')];selection=dict(gates=gates,source_hashes={name:sha(OUT/name) for name in ['aggregation_predictions.csv','modern_predictions.csv','chronos_predictions.csv']},all_evaluation_periods_exposed=True)
 # Freeze decisions before this script reads evaluation-period metrics.
 dump(path,selection);fusion=[];extra=[]
 for g in gates:
  if g['passed']:
   name,records=fuse(d,g['method']);extra.append(name);fusion.extend(records)
 dump(OUT/'fusion_selection.json',dict(records=fusion,reason='only gates that pass receive finite past-only global fusion'))
 methods=['F0','R1','F1','F2','F6','A01','A1','E01','E1','MODERN','HISTORY_LR','CHRONOS_LR']+extra;summarize(d,methods,'all')
 pairs=[('E01','A01'),('E1','A1'),('A01','A1'),('E01','E1'),('MODERN','F2'),('CHRONOS_LR','HISTORY_LR'),('CHRONOS_LR','R1')]+[(x,'F1') for x in extra]
 legacy_report.PAIRS=pairs;legacy_report.intervals(d).to_csv(OUT/'paired_intervals.csv',index=False)
 transitions=[];subgroups=[];cases=[]
 for (phase,symbol),g in d[d.phase!='warmup'].groupby(['phase','symbol']):
  for new,base in pairs:
   a=(g[new]>=.5)==g.label;b=(g[base]>=.5)==g.label;transitions.append(dict(phase=phase,symbol=symbol,method=new,baseline=base,changed_right=int((a&~b).sum()),changed_wrong=int((~a&b).sum()),common_right=int((a&b).sum()),common_wrong=int((~a&~b).sum())))
  for subset,mask in [('news',g.has_original_news==1),('no_news',g.has_original_news==0)]:
   h=g[mask]
   for method in methods:
    if len(h):subgroups.append(dict(phase=phase,symbol=symbol,subset=subset,method=method,**metric(h.label,h[method])))
 for symbol in ('AAPL','AMZN'):
  g=d[(d.phase=='later')&(d.symbol==symbol)].copy();g['hash']=g.key.map(lambda k:hashlib.sha256(k.encode()).hexdigest())
  for new,base in [('MODERN','F2'),('CHRONOS_LR','HISTORY_LR')]:
   a=(g[new]>=.5)==g.label;b=(g[base]>=.5)==g.label
   for category,mask in [('changed_right',a&~b),('changed_wrong',~a&b),('common_right',a&b),('common_wrong',~a&~b)]:
    for r in g[mask].sort_values('hash').head(1).to_dict('records'):cases.append(dict(symbol=symbol,method=new,baseline=base,category=category,key=r['key'],label=r['label'],probability=r[new],base_probability=r[base],has_news=r['has_original_news']))
 pd.DataFrame(transitions).to_csv(OUT/'transitions.csv',index=False);pd.DataFrame(subgroups).to_csv(OUT/'subgroups.csv',index=False);dump(OUT/'cases.json',cases)
 print(json.dumps(selection,indent=2))
if __name__=='__main__':main()
