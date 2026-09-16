"""Separate prequential replay. No same-day outcomes or expert refits."""
from core import *
import argparse,joblib,time,os

def intercept_fit(prior,recent,method):
    parts=[]
    for data,mass in [(prior,20.),(recent,float(recent.day.nunique()))]:
        if data.empty:continue
        z=data.copy();z['weight']=mass/data.day.nunique()/data.groupby('day').key.transform('size');parts.append(z)
    data=pd.concat(parts,ignore_index=True);w=data.weight.to_numpy();w/=w.sum();y=data.label.to_numpy();o=logit(np.clip(data[method].to_numpy(),1e-8,1-1e-8))
    def f(b):
        t=o+b[0];return np.sum(w*(np.logaddexp(0,t)-y*t))+.5*b[0]**2,np.array([np.sum(w*(expit(t)-y))+b[0]])
    r=minimize(f,[0.],jac=True,bounds=[(-.5,.5)],method='L-BFGS-B')
    if not r.success:raise RuntimeError(r.message)
    return float(r.x[0]),int(r.nit)

def replay(oof,frozen,bundles):
    output=[];states=[];fits=0
    for sym,f in frozen.groupby('symbol'):
        prior=oof[oof.symbol==sym].copy();seen=prior.copy();static=np.asarray(bundles[sym]['static'])
        loss=prior[['day']].copy()
        for m in ['price','body','semantic']:loss[m]=(prior.label-prior[m])**2
        initial=loss.groupby('day')[['price','body','semantic']].mean().mean().to_numpy()
        for day,current in f.sort_values('cutoff_utc').groupby('day',sort=True):
            current=current.copy();allowed=seen[(seen.day<day)&(seen.end_utc<current.cutoff_utc.min())]
            temporal_check(allowed,current);days=sorted(allowed.day.unique())[-20:];recent=allowed[allowed.day.isin(days)]
            losses=recent[['day']].copy()
            for m in ['price','body','semantic']:losses[m]=(recent.label-recent[m])**2
            means=losses.groupby('day')[['price','body','semantic']].mean()
            estimate=(20*initial+means.sum().to_numpy())/(20+len(means))
            w=.5*static+.5*softmax(-estimate/.05)
            current['online_weights']=combine(current,w)
            state={'symbol':sym,'day':day,'cutoff':str(current.cutoff_utc.min()),'max_used_label_end':str(allowed.end_utc.max()),'recent_keys':recent.key.tolist(),'prior_keys':prior.key.tolist(),'weights':w.tolist(),'pid':os.getpid(),'intercepts':{}}
            for m in ['price','title']:
                b,n=intercept_fit(prior,recent,m);current['online_'+m]=expit(logit(np.clip(current[m],1e-8,1-1e-8))+b)
                state['intercepts'][m]={'value':b,'optimizer_iterations':n};fits+=1
            for i,m in enumerate(['zero','body','semantic']):current['online_weight_'+m]=w[i]
            states.append(state);output.append(current);seen=pd.concat([seen,current],ignore_index=True)
    return pd.concat(output,ignore_index=True),states,fits

def main(run,out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    paths=[run/'oof.pkl',run/'predictions.pkl',run/'AAPL_system.joblib',run/'AMZN_system.joblib',B/'online.py',B/'core.py',B/'PROTOCOL.md'];source={str(p):sha(p) for p in paths}
    oof=pd.read_pickle(run/'oof.pkl');frozen=pd.read_pickle(run/'predictions.pkl').query("phase=='frozen'").copy();bundles={s:joblib.load(run/f'{s}_system.joblib') for s in ['AAPL','AMZN']}
    result,states,fits=replay(oof,frozen,bundles)
    result.to_pickle(out/'predictions.pkl');result.drop(columns=['stem_body','text','article_keys'],errors='ignore').to_csv(out/'predictions.csv',index=False)
    dump(out/'daily_states.json',states)
    # Metamorphic test: alter October and later labels. September outputs must be identical.
    perturbed=frozen.copy();mask=perturbed.month>='2018-10';perturbed.loc[mask,'label']=1-perturbed.loc[mask,'label']
    probe,_,_=replay(oof,perturbed,bundles)
    a=result[result.month=='2018-09'].set_index('key');b=probe[probe.month=='2018-09'].set_index('key').loc[a.index]
    for m in ['online_weights','online_price','online_title']:np.testing.assert_array_equal(a[m].to_numpy(),b[m].to_numpy())
    empty=result.has_news==0;np.testing.assert_array_equal(result.loc[empty,'online_weights'],result.loc[empty,'price'])
    check_hashes(source);dump(out/'sources.json',source)
    dump(out/'status.json',{'status':'COMPLETE','rows':len(result),'daily_states':len(states),'primary_intercept_fits':fits,'additional_metamorphic_check_fits':fits,'expert_refits':0,'seconds':time.monotonic()-start,'future_label_perturbation_verified':True,'no_same_day_labels':True,'exact_no_news_fallback':True})
    print('ONLINE COMPLETE',len(states),'day-stock updates',fits,'intercept fits',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.run,a.output)
