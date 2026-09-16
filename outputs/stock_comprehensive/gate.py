from core import *
def assess(run):
    rows=[];cv=pd.read_csv(run/'M02/cv.csv');chosen=json.loads((run/'M02/selected.json').read_text());pieces=[]
    for (stock,method),g in cv.groupby(['symbol','method']):pieces.append(g[g.C.eq(chosen[stock+'_'+method])])
    z=pd.concat(pieces)
    for pen in ['l1','l2']:
        for k in [25,100]:
            a=z[z.method.eq(f'k{k}_{pen}')].set_index(['symbol','month']);b=z[z.method.eq(f'k500_{pen}')].set_index(['symbol','month']);diff=a[['balanced_accuracy','brier']]-b[['balanced_accuracy','brier']];mo=diff.groupby('month').mean();rows.append(dict(candidate=f'M02_k{k}_{pen}',baseline=f'k500_{pen}',positive_months=int(mo.balanced_accuracy.gt(0).sum()),mean_BA_gain=mo.balanced_accuracy.mean(),mean_Brier_change=mo.brier.mean(),monthly=mo.to_dict('index')))
    if (run/'M08/selected_oof.csv').exists():
        z=pd.read_csv(run/'M08/selected_oof.csv');z['month']=z.start_utc.str[:7];met=[]
        for (stock,month,name),g in z[z.method.isin(['fusion','price'])].groupby(['symbol','month','method']):met.append(dict(symbol=stock,month=month,method=name,**metrics(g.label,g.p)))
        t=pd.DataFrame(met);a=t[t.method.eq('fusion')].set_index(['symbol','month']);b=t[t.method.eq('price')].set_index(['symbol','month']);mo=(a[['balanced_accuracy','brier']]-b[['balanced_accuracy','brier']]).groupby('month').mean();rows.append(dict(candidate='M08_fusion',baseline='price',positive_months=int(mo.balanced_accuracy.gt(0).sum()),mean_BA_gain=mo.balanced_accuracy.mean(),mean_Brier_change=mo.brier.mean(),monthly=mo.to_dict('index')))
    for r in rows:r['pass']=bool(r['positive_months']>=2 and r['mean_BA_gain']>=.01 and r['mean_Brier_change']<=.002)
    result={'comparisons':rows,'advanced_enter':any(r['pass'] for r in rows),'interpretation':'Selected training-OOF comparisons; budget gate only, not unbiased efficacy estimate'};dump(run/'advanced_gate.json',result);return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();print(json.dumps(assess(a.run),indent=2))
