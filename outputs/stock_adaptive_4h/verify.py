"""Independent saved-run lineage and numerical checks, including historical code snapshots."""
from core import *
import argparse,joblib

def main(run,out):
    fits=json.loads((run/'fits.json').read_text());selection=json.loads((run/'selection.json').read_text());inputs=pd.read_pickle(run/'inputs.pkl');oof=pd.read_pickle(run/'oof.pkl');pred=pd.read_pickle(run/'predictions.pkl')
    expected=json.loads((run/'sources.json').read_text());resolved={}
    for name,h in expected.items():
        p=Path(name);snap=run/'code_snapshot'/p.name
        if p.parent==B and snap.exists() and sha(snap)==h:resolved[str(snap)]=h
        else:resolved[name]=h
    check_hashes(resolved);lookup=inputs.set_index('key')
    for f in fits:
        assert sha(run/'models'/f"{f['name']}.joblib")==f['model_hash']
        temporal_check(lookup.loc[f['train_keys']],lookup.loc[f['eval_keys']])
        if f['layer']=='residual':
            assert f['train_keys']==f['offset_keys'];parents=oof.set_index('key').loc[f['offset_keys']]
            assert parents.month.max()<lookup.loc[f['eval_keys']].month.min()
            model=joblib.load(run/'models'/f"{f['name']}.joblib");assert model['transform'].train_keys_==f['train_keys']
        if f['layer']=='router':
            parents=oof.set_index('key').loc[f['train_keys']];assert parents.residual_ready.all()
    base=pd.read_csv(run/'base_cv.csv');res=pd.read_csv(run/'residual_cv.csv');gcv=pd.read_csv(run/'gate_cv.csv')
    for s in selection:
        month=s['month'];sym=s['symbol'];kind=s['kind']
        assert all(m<month for m in s.get('selection_months',[]))
        if kind in ['price','title']:
            v=base[(base.symbol==sym)&(base.kind==kind)&(base.month<month)&(base.month>='2018-03')]
            c=.1 if v.empty else float(v.groupby('C')[['BA','Brier']].mean().reset_index().sort_values(['BA','Brier','C'],ascending=[False,True,True]).iloc[0].C)
            assert s['C']==c
        elif kind in ['body','semantic','joint_body','joint_semantic']:
            v=res[(res.symbol==sym)&(res.kind==kind)&(res.month<month)]
            if v.empty:c=.1
            else:
                a=v.groupby('C')[['BA','Brier','base_Brier']].mean().reset_index();a=a[a.Brier<=a.base_Brier+.002]
                c=None if a.empty else float(a.sort_values(['BA','Brier','C'],ascending=[False,True,True]).iloc[0].C)
            assert s['C']==c
        elif kind=='static':
            h=oof[(oof.symbol==sym)&(oof.month<month)&oof.residual_ready];w,_=select_static(h);assert s['weights']==w
        elif kind=='router':
            v=gcv[(gcv.month<month)&gcv.trained];candidates=[]
            for (a,rho),g in v.groupby(['alpha','rho']):
                z=g.groupby('symbol')[['BA','Brier','static_Brier']].mean()
                if len(z)==2 and (z.Brier<=z.static_Brier+.002).all():candidates.append((float(z.BA.mean()),float(z.Brier.mean()),float(rho),float(a)))
            if v.empty:alpha,rho=10.,.25
            elif not candidates:alpha,rho=10.,0.
            else:
                _,_,rho,alpha=sorted(candidates,key=lambda x:(-x[0],x[1],x[2],-x[3]))[0]
            assert s['alpha']==alpha and s['rho']==rho
    z=np.load(OLD/'prepared/articles.npz');keys=z['keys'];emb=z['embeddings'].astype(float);maximum=0
    for sym in ['AAPL','AMZN']:
        bundle=joblib.load(run/f'{sym}_system.joblib');x=inputs[(inputs.symbol==sym)&(inputs.month>='2018-09')].copy();target=pred[pred.symbol.eq(sym)&pred.phase.eq('frozen')].set_index('key')
        clean=predict_bundle(bundle,x.drop(columns=['label','target_return']),keys,emb).set_index('key');x.label=1-x.label;x.target_return=-x.target_return
        altered=predict_bundle(bundle,x,keys,emb).set_index('key')
        for name in ['price','title','body','semantic','joint_body','joint_semantic','static','gate','price_cal','static_cal','gate_cal']:
            a=clean[name].to_numpy();b=target.loc[clean.index,name].to_numpy();np.testing.assert_allclose(a,b,atol=1e-12,rtol=0);np.testing.assert_array_equal(a>=.5,b>=.5);np.testing.assert_array_equal(a,altered[name]);maximum=max(maximum,float(np.max(np.abs(a-b))))
        empty=clean.has_news.eq(0)
        for name in ['body','semantic','static','gate']:np.testing.assert_array_equal(clean.loc[empty,name],clean.loc[empty,'price'])
        for name in ['static_cal','gate_cal']:np.testing.assert_array_equal(clean.loc[empty,name],clean.loc[empty,'price_cal'])
    # Price-label check against raw bars, independently from the prepared label column.
    for sym,g in inputs.groupby('symbol'):
        file=ROOT/'work/stock-data/raw/CHARTS'/('APPLE5.csv' if sym=='AAPL' else 'AMAZON5.csv');bars=pd.read_csv(file,header=None,names=['date','time','open','high','low','close','activity']);bars.index=pd.to_datetime(bars.date+' '+bars.time,format='%Y.%m.%d %H:%M',utc=True)
        for r in g.itertuples():
            v=bars.reindex(pd.date_range(r.start_utc,periods=48,freq='5min'));assert v[['open','close']].notna().all().all();assert int(v.close.iloc[-1]>v.open.iloc[0])==r.label
    inference_sources={str(p):sha(p) for p in [run/'inputs.pkl',run/'AAPL_system.joblib',run/'AMZN_system.joblib',OLD/'prepared/articles.npz',B/'core.py',OLD/'run.py',B.parent/'stock_four_hour_v2/text_vectorizer.py']}
    dump(out,{'status':'PASS','fits_checked':len(fits),'raw_4h_labels_checked':len(inputs),'frozen_predictions_checked':int(pred.phase.eq('frozen').sum()),'maximum_reload_error':maximum,'future_label_mutation_unchanged':True,'exact_no_news_and_calibrated_fallback':True,'C_static_and_router_choices_recomputed':True,'code_snapshots_and_data_hashes_verified':True,'inference_sources':inference_sources,'methodology_note':'Programmatic checks do not certify external timestamp accuracy or forecasting efficacy.'})
    print('PASS',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.run,a.output)
