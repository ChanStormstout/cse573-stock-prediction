"""Temporal, outcome-blind retrieval, prompt, fallback and numerical contracts."""
from common import *

def main():
    seal=check_prepared();d=pd.read_pickle(PRIVATE/'rows.pkl');docs=json.loads((PRIVATE/'documents.json').read_text())
    audits=json.loads((PRIVATE/'retrieval.json').read_text());groups=[r['group'] for r in audits]
    arrays=np.load(PRIVATE/'retrieval_arrays.npz');sim=arrays['similarity'];prompts=jsonl(PRIVATE/'prompts.jsonl')
    lookup={(r['key'],r['variant']):r for r in prompts};count=0
    baseline=pd.read_csv(ROOT/'outputs/stock_foundation_4h/v1/all_predictions.csv').set_index('key')
    assert d.key.is_unique and len(d)==1607
    assert np.array_equal(baseline.loc[d.key,'label'].to_numpy(),d.label.to_numpy())
    assert ((d.end_utc-d.start_utc)==pd.Timedelta('4h')).all()
    assert ((d.start_utc-d.cutoff_utc)==pd.Timedelta('5min')).all()
    for _,stock in d.groupby('symbol'):assert stock.start_utc.is_monotonic_increasing
    for row in jsonl(PACKS):
        assert all(pd.Timestamp(b['end'])<=pd.Timestamp(row['cutoff']) for b in row['price_rows'])
    for i,r in enumerate(audits):
        query=d.iloc[i];inds=r['case_indices'];count+=len(inds)
        assert len(inds)<=3 and len(set(r['case_groups']))==len(inds)
        assert len(set(str(d.iloc[j].day) for j in inds))==len(inds)
        for j in inds:
            past=d.iloc[j]
            assert past.symbol==query.symbol and past.month<min(query.month,'2018-09') and past.end_utc<query.cutoff_utc
            assert docs[j]['category']==docs[i]['category'] and not docs[j]['category'].endswith('unknown')
            assert groups[j]!=groups[i]
            assert pd.Timestamp(docs[j]['input']['news']['available_at'])<=past.cutoff_utc
        if (query.key,'P2') in lookup:
            a=json.loads(lookup[(query.key,'P2')]['messages'][1]['content'])
            b=json.loads(lookup[(query.key,'P3')]['messages'][1]['content'])
            assert a['CURRENT']==b['CURRENT']
            for hh,j in zip(b['HISTORICAL_CASES'],inds):
                assert hh['outcome']['direction']==('UP' if d.iloc[j].label else 'DOWN')
                assert abs(hh['outcome']['return_pct']-arrays['returns'][j])<=.00000051
                hh.pop('outcome')
            assert a==b
    # Representative perturbation checks across all three phases, both stocks.
    altered=d.copy();altered['label']=1-altered.label;tested=0
    for key in json.loads((PRIVATE/'case_selection.json').read_text()):
        i=int(d.index[d.key==key][0]);original=select_cases(i,d,docs,sim[i],groups)
        assert select_cases(i,altered,docs,sim[i],groups)==original
        badsim=sim[i].copy();future=(d.end_utc>=d.iloc[i].cutoff_utc)|(d.month>=min(d.iloc[i].month,'2018-09'))
        badsim[future]=1e6
        assert select_cases(i,d,docs,badsim,groups)==original
        tested+=1
    results=dict(prepared_fingerprints=True,temporal_cases_checked=count,identical_P2_P3_except_outcomes=True,
                 label_flip_retrieval_checks=tested,future_similarity_perturbation_checks=tested,
                 original_labels_keys_horizons=True,price_bars_before_cutoff=True)
    # Explicitly exercise cache rejection without modifying production files.
    from unittest.mock import patch
    with patch('common.fingerprint',return_value={'changed':'input'}):
        try:check_prepared()
        except ValueError:pass
        else:raise AssertionError('cache mismatch not rejected')
    results['cache_mismatch_rejected']=True
    if (PRIVATE/'completed.json').exists():
        pred=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip');ev=pred[pred.phase!='warmup']
        generated=jsonl(PRIVATE/'generated.jsonl');assert len(generated)==len(prompts)
        assert len({(r['key'],r['variant']) for r in generated})==len(prompts)
        err=0.
        for r in generated:
            p=float(1/(1+np.exp(r['logp_down']-r['logp_up'])))
            err=max(err,abs(r['p']-p));assert err<1e-12
            assert r['prompt_sha256']==digest(json.dumps(lookup[(r['key'],r['variant'])]['messages'],ensure_ascii=False))
        for m in ['P0','P1','P2','P3']:
            g=ev[ev.has_excerpt==0];assert np.array_equal(g[m].to_numpy(),g.R1.to_numpy())
        g=ev[ev.n_cases==0]
        assert np.array_equal(g.P2.to_numpy(),g.P0.to_numpy()) and np.array_equal(g.P3.to_numpy(),g.P0.to_numpy())
        assert np.array_equal(g.P1.to_numpy(),g.R1.to_numpy())
        met=pd.read_csv(OUT/'metrics.csv');maxerr=0.
        for r in met.itertuples():
            g=ev[(ev.symbol==r.symbol)&(ev.phase==r.phase)];y=g.label.to_numpy();p=g[r.method].to_numpy();q=p>=.5
            # Independent BA/Brier formulas; MCC separately via confusion counts.
            tp=int(((y==1)&q).sum());tn=int(((y==0)&~q).sum());fp=int(((y==0)&q).sum());fn=int(((y==1)&~q).sum())
            denom=float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
            mm=(tp*tn-fp*fn)/denom if denom else 0
            actual=[.5*(tp/(tp+fn)+tn/(tn+fp)),mm,np.square(y-p).mean()]
            maxerr=max(maxerr,max(abs(a-b) for a,b in zip(actual,[r.BA,r.MCC,r.Brier])))
        assert maxerr<1e-12
        results.update(inference_calls=len(generated),exact_fallback=True,probability_max_error=err,
                       independently_recomputed_metric_rows=len(met),metric_max_error=maxerr)
    dump(OUT/'verification.json',results);print(json.dumps(results,indent=2))

if __name__=='__main__':main()
