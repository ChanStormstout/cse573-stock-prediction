"""Build additional audit materials without changing trained results."""
from core import *
import importlib.metadata as md

def materials(run):
    out=run/'materials'
    if out.exists():raise FileExistsError(out)
    out.mkdir();d=load(run);news=pd.read_pickle(W/'audit/news_index.pkl');news['key']=news.archive+'::'+news.member;news=news.set_index('key');events=pd.read_json(run/'M06/events.jsonl',lines=True)
    # Reviewer coverage/recall material must include articles with NO extracted frame.
    rows=[];rng=np.random.default_rng(577)
    for stock,s in d[d.split.eq('train')].groupby('symbol'):
        ids=sorted({k for x in s.news_record_keys for k in x.split('|') if k});extracted=set(events.loc[events.symbol.eq(stock),'record_key'])
        for stratum in ['has_frame','no_frame']:
            candidates=[k for k in ids if (k in extracted)==(stratum=='has_frame')];chosen=rng.choice(candidates,min(50,len(candidates)),replace=False)
            for k in chosen:rows.append(dict(symbol=stock,record_key=k,stratum=stratum,title=news.loc[k,'title'],body_source=str(O/'stock_event_facts/results/bodies.json'),human_event_present='',human_missed_events='',human_target_correct='',reviewer='',note='',purpose='recall audit only; not independent gold until reviewed'))
    pd.DataFrame(rows).to_csv(out/'article_recall_review.csv',index=False)
    # Known real failures remain inspectable alongside new fields/abstentions.
    before=json.loads((O/'stock_event_facts_v3/results/assistant_review.json').read_text());reg=[]
    for x in before:
        if x.get('review_id') not in [2,16,22]:continue
        matches=events[events.record_key.eq(x['record_key'])];bad=x.get('actor');assert not matches.actor.fillna('').eq(bad).any();reg.append(dict(record_key=x['record_key'],old_actor=bad,new_frames=matches[['kind','actor','action','extraction_status','unknown_fields']].to_dict('records'),result='old erroneous issuer span not emitted; not a full-event quality verdict'))
    dump(out/'real_error_regressions.json',reg)
    # Exactly matched reruns of existing controls.
    prev=pd.read_csv(O/'stock_adaptive/results/predictions.csv');cur=pd.read_csv(run/'M03/predictions.csv');oldref=prev[prev.penalty.eq('l2')&prev.policy.eq('expanding')];newref=cur[cur.method.eq('retuned')];z=newref.merge(oldref,on=['symbol','start_utc'],suffixes=('_new','_old'));assert len(z)==1233;np.testing.assert_allclose(z.p_new,z.p_old,atol=1e-10,rtol=0)
    baseline=pd.read_csv(run/'M02/predictions.csv');frozen=cur[cur.method.eq('frozen')];zz=baseline[baseline.method.eq('k500_l2')].merge(frozen,on=['symbol','start_utc'],suffixes=('_new','_old'));np.testing.assert_allclose(zz.p_new,zz.p_old,atol=1e-10,rtol=0)
    dump(out/'matched_reference_parity.json',{'E13_L2_expanding_max_abs_diff':float(np.max(abs(z.p_new-z.p_old))),'E12_L2_frozen_max_abs_diff':float(np.max(abs(zz.p_new-zz.p_old))),'n_each':1233})
    foldmanifest={}
    for stock,s in d.groupby('symbol'):
        for month,a,b in folds(s):foldmanifest[stock+'_'+month]=dict(train_keys=keys(a),validation_keys=keys(b),train_label_end=str(a.end_utc.max()),validation_cutoff=str(b.cutoff_utc.min()),features=PRICE+['stem_body'])
    dump(out/'forward_fold_manifest.json',foldmanifest)
    # Currency-like fee accounting in units of initial wealth, alongside proportional fee sums.
    tr=pd.read_csv(run/'M09/transitions.csv');cost=[]
    for (stock,method,bps,split),g in tr.groupby(['symbol','method','cost_bps','split']):
        w=np.r_[1,np.exp(np.cumsum(g.log_return.to_numpy()))];cost.append(dict(symbol=stock,method=method,cost_bps=bps,split=split,paid_cost_initial_wealth=float(np.dot(w[:-1],g.fee_fraction)),net_return=w[-1]-1))
    pd.DataFrame(cost).to_csv(out/'rl_wealth_costs.csv',index=False)
    seed=pd.read_csv(run/'M09/summary.csv');seed=seed[seed.method.str.startswith('PPO_')];seed.groupby(['symbol','cost_bps','period'])[['net_return','max_drawdown','turnover','cash_fraction']].agg(['mean','std','min','max']).to_csv(out/'rl_seed_variability.csv')
    versions={x:md.version(x) for x in ['numpy','pandas','scipy','scikit-learn','joblib','torch','transformers','tokenizers','nltk','stable-baselines3','gymnasium','matplotlib','tabulate']};dump(out/'versions.json',versions);(B/'requirements.txt').write_text(''.join(f'{k}=={v}\n' for k,v in versions.items()))
    # Source/input sealing added after incremental implementation; not retroactive preregistration.
    paths=list(B.glob('*.py'))+[B/'configs/main.json',W/'audit/news_index.pkl',W/'audit/xnys_schedule.csv',O/'stock_event_facts/results/bodies.json',O/'stock_event_facts_v2/extract.py',O/'stock_event_facts/extract.py']+list((W/'raw/CHARTS').glob('*5.csv'))+list((W/'raw/news').glob('*.zip'))
    dump(run/'sealed_inputs.json',{str(p):sha(p) for p in paths});dump(out/'provenance_note.json',{'initial_sources':'runs/v1/sources.json registered M01/M02/M03 core before execution','incremental_modules':'M04-M09 implemented sequentially; final hashes sealed after execution. This is implementation provenance, not independent preregistration. Config/gates were registered before experiment outcomes.','failed_versions':['M04 pandas key index ambiguity before any results; fixed in M04_v2','report renderer missing tabulate; report_failed_v1 retained','M07 unique event diagnostic initially counted across all periods; M07_v2 restricts count to train. Gate unchanged, no event model trained.'],'unchanged_training_results':True})
    dump(out/'ui_verification.json',{'browser':'local in-app browser','AAPL_base_fusion':'59.75% both at first development window','AMZN_base_fusion_text':['62.84%','50.95%','46.98%'],'stock_selector':'tested','news_timestamps':'visible','screenshot_layout':'checked','external_requests_in_demo':'none'})
    print('materials complete',flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();materials(a.run.resolve())
