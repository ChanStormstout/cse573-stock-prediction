"""Train-period advancement followed by separate exposed-period reporting."""
from core import *
from scipy.special import expit,logit
from scipy.optimize import minimize_scalar
import html
PAIRS=[('P_cross_lr','P_own_lr'),('P_cross_hgb','P_own_hgb'),('P_cross_tabpfn','P_own_tabpfn'),('P_own_hgb','P_own_lr'),('P_own_tabpfn','P_own_lr'),('P_cross_hgb','P_cross_lr'),('P_cross_tabpfn','P_cross_lr'),('T_A1','T_original'),('T_original_interaction','T_original'),('T_A1_interaction','T_A1'),('T_mix','F2_new'),('H1','H0'),('H2','H1')]
def monthscores(d,p):return pd.DataFrame([dict(symbol=s,month=m,**metric(g.label,g[p])) for (s,m),g in d.groupby(['symbol','month'])])
def gates(d):
 result=[]
 for method,base in PAIRS:
  a=monthscores(d,method).set_index(['symbol','month']);b=monthscores(d,base).set_index(['symbol','month']);aa=a.BA.groupby('symbol').mean();bb=b.BA.groupby('symbol').mean();delta=aa-bb;monthdelta=(a.BA-b.BA).groupby('month').mean();gain=float(aa.min()-bb.min());passed=bool(gain>=.01 and delta.min()>=-.01 and (monthdelta>0).sum()>=2)
  result.append(dict(method=method,base=base,weaker_gain=gain,macro_gain=float(delta.mean()),AAPL_gain=float(delta.AAPL),AMZN_gain=float(delta.AMZN),positive_months=int((monthdelta>0).sum()),passed=passed))
 return result

def combinations(d):
 """Architecture eligibility for final blend is fixed by outer months only.
 Earlier blend scores are post-selection diagnostic and never labelled outer validation.
 """
 gate=gates(d[d.month.between('2018-06','2018-08')]);dump(OUT/'advancement.json',gate);passed=sorted([g for g in gate if g['passed']],key=lambda g:(-g['weaker_gain'],-g['macro_gain'],g['method']));chosen=[]
 for g in passed:
  if g['method'] not in chosen:chosen.append(g['method'])
  if len(chosen)==2:break
 records=[]
 for method in chosen:
  name='Blend_'+method;d[name]=np.nan
  for month in MONTHS:
   boundary='2018-09' if month=='final' else month;prior=d[d.month.ge('2018-03')&d.month.lt(boundary)];weight=0.
   if len(prior):
    candidates=[]
    for w in (0.,.25,.5,.75,1.):
     a=prior.copy();a['candidate']=(1-w)*a.F1_new+w*a[method];s=monthscores(a,'candidate').groupby('symbol').BA.mean();candidates.append((s.min(),s.mean(),-w,w))
    weight=max(candidates)[-1]
   mask=d.month.ge('2018-09') if month=='final' else d.month.eq(month);d.loc[mask,name]=(1-weight)*d.loc[mask,'F1_new']+weight*d.loc[mask,method];records.append(dict(method=name,month=month,weight=weight,architecture_selected_using='June-August; March-August blend figures post-selection diagnostic',weight_selection_months=sorted(prior.month.unique())))
 dump(OUT/'combination_selection.json',records);return ['Blend_'+m for m in chosen]
def temperature(d,methods):
 rows=[]
 for method in methods:
  name=method+'_temperature';d[name]=np.nan
  for month in MONTHS:
   boundary='2018-09' if month=='final' else month
   for symbol in ('AAPL','AMZN'):
    past=d[d.month.ge('2018-03')&d.month.lt(boundary)&d.symbol.eq(symbol)];a=1.
    if len(past)>=30:
     logits=logit(np.clip(past[method].to_numpy(),1e-6,1-1e-6));y=past.label.to_numpy();res=minimize_scalar(lambda v:np.mean(np.logaddexp(0,np.exp(v)*logits)-y*np.exp(v)*logits),bounds=(-4.6,4.6),method='bounded');assert res.success;a=float(np.exp(res.x))
    mask=d.symbol.eq(symbol)&(d.month.ge('2018-09') if month=='final' else d.month.eq(month));p=d.loc[mask,method].to_numpy();q=expit(a*logit(np.clip(p,1e-6,1-1e-6)));assert np.array_equal(p>=.5,q>=.5);d.loc[mask,name]=q;rows.append(dict(method=method,month=month,symbol=symbol,positive_slope=a,past_n=len(past)))
 dump(OUT/'calibration.json',rows)

def paired_intervals(d,methods,matched=False):
 rows=[]
 # The same joint-date block draw is used for both stocks and every method.
 for phase,g in d[d.phase.ne('warmup')].groupby('phase'):
  days=sorted(g.day.unique());n=len(days);loc={v:i for i,v in enumerate(days)}
  for block in (1,5):
   rng=np.random.default_rng(573);weights=[]
   for _ in range(1000):
    starts=rng.integers(0,n,size=int(np.ceil(n/block)));draw=np.concatenate([(start+np.arange(block))%n for start in starts])[:n];weights.append(np.bincount(draw,minlength=n))
   weights=np.array(weights)
   for s,gg in g.groupby('symbol'):
    ids=np.array([loc[v] for v in gg.day]);w=weights[:,ids];y=gg.label.to_numpy();baseline=gg.F0.to_numpy()>=.5
    def ba(q):
     return .5*((w[:,y==1]@q[y==1])/w[:,y==1].sum(1)+(w[:,y==0]@(~q[y==0]))/w[:,y==0].sum(1))
    for method,base in (PAIRS if matched else [(m,'F0') for m in methods]):
     b=ba(gg[base].to_numpy()>=.5);delta=ba(gg[method].to_numpy()>=.5)-b;lo,hi=np.nanquantile(delta,[.025,.975]);rows.append(dict(phase=phase,symbol=s,method=method,baseline=base,block_days=block,BA_difference_low=lo,BA_difference_high=hi,resamples=1000,selection_adjusted=False))
 return pd.DataFrame(rows)
def seed_reports(d):
 rows=[]
 for method in ('T_original_interaction','T_A1','T_A1_interaction'):
  folder=PRIVATE/method;choices=json.loads((folder/'done.json').read_text())['choices'];allp={seed:np.full(len(d),np.nan) for seed in (573,574,575)}
  for choice in choices:
   for symbol in ('AAPL','AMZN'):
    tr,ev=split(d,symbol,choice['month']);z=np.load(folder/f"seeds_{tr[-1]}_{choice['C']}.npz");assert np.array_equal(z['indices'],ev)
    for seed,p in zip(z['seeds'],z['probabilities']):
     p=p.astype(float).copy();none=d.iloc[ev].has_original_news.to_numpy()==0;p[none]=d.iloc[ev].P_own_lr.to_numpy()[none];allp[int(seed)][ev]=p
  for seed,p in allp.items():
   for (symbol,phase),g in d[d.phase.ne('warmup')].groupby(['symbol','phase']):rows.append(dict(method=method,seed=seed,symbol=symbol,phase=phase,**metric(g.label,p[g.index])))
 frame=pd.DataFrame(rows);frame.to_csv(OUT/'seed_metrics.csv',index=False);frame.groupby(['method','symbol','phase'])[['BA','MCC','Brier']].agg(['mean','std']).to_csv(OUT/'seed_summary.csv')

def main():
 d=data();methods=[];evidence={}
 for p in sorted(PRIVATE.glob('*/done.json')):
  method=p.parent.name;pred=pd.read_csv(p.parent/'predictions.csv',float_precision='round_trip');assert np.array_equal(pred.key,d.key);d[method]=pred.p;methods.append(method);evidence[method]=json.loads(p.read_text())
 required=['P_own_lr','P_cross_lr','P_own_hgb','P_cross_hgb','P_own_tabpfn','P_cross_tabpfn','F1_new','F2_new','T_mix','T_original','T_A1','T_original_interaction','T_A1_interaction','H0','H1','H2'];assert set(required)<=set(methods),set(required)-set(methods)
 blends=combinations(d);methods+=blends;seed_reports(d)
 refs=['F0','R1','F1','F2','F6'];temperature(d,methods);metrics=[];monthly=[];strata=[];transitions=[]
 for method in refs+methods+[m+'_temperature' for m in methods]:
  for (s,phase),g in d[d.phase.ne('warmup')].groupby(['symbol','phase']):
   metrics.append(dict(method=method,symbol=s,phase=phase,**metric(g.label,g[method])));a=g[method].to_numpy()>=.5;b=g.F0.to_numpy()>=.5;y=g.label.to_numpy();transitions.append(dict(method=method,symbol=s,phase=phase,corrected=int(((a==y)&(b!=y)).sum()),broken=int(((a!=y)&(b==y)).sum()),both_wrong=int(((a!=y)&(b!=y)).sum())))
  for (s,m),g in d[d.phase.ne('warmup')].groupby(['symbol','month']):monthly.append(dict(method=method,symbol=s,month=m,**metric(g.label,g[method])))
  for (s,phase,news),g in d[d.phase.ne('warmup')].groupby(['symbol','phase','has_original_news']):strata.append(dict(method=method,symbol=s,phase=phase,original_news=int(news),**metric(g.label,g[method])))
 pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(monthly).to_csv(OUT/'monthly_metrics.csv',index=False);pd.DataFrame(strata).to_csv(OUT/'strata_metrics.csv',index=False);pd.DataFrame(transitions).to_csv(OUT/'transitions.csv',index=False);paired_intervals(d,methods).to_csv(OUT/'paired_intervals.csv',index=False);paired_intervals(d,methods,matched=True).to_csv(OUT/'matched_intervals.csv',index=False)
 d[['key','symbol','day','month','phase','label','has_original_news']+refs+methods+[m+'_temperature' for m in methods]].to_csv(OUT/'predictions.csv',index=False)
 # Save detailed training records (no article text, raw price features or weights).
 dump(OUT/'training_evidence.json',evidence);dump(OUT/'protocol.json',dict(protocol_sha256=sha(OUT.parent/'PRE_REGISTRATION.md'),status='completed finite existing-data experiment',sample_count=len(d),all_periods_exposed=True,external_market='AUTH_REQUIRED',methods=methods,independent_event_quality_review=False))
 table=pd.DataFrame(metrics);plain=table[table.method.isin(refs+methods)];pivot=plain.pivot(index='method',columns=['symbol','phase'],values='BA');outer=[]
 for method in refs+methods:
  s=monthscores(d[d.month.between('2018-06','2018-08')],method).groupby('symbol').BA.mean();outer.append(dict(method=method,AAPL=s.AAPL,AMZN=s.AMZN,weakest=s.min()))
 outer=pd.DataFrame(outer);outer.to_csv(OUT/'outer_monthly_summary.csv',index=False)
 # Cases fixed before new results. Describe correctness separately from interpretation.
 bykey=d.set_index('key');cases=json.loads((OUT/'case_selection.json').read_text());case_rows=[]
 for case in cases:
  r=bykey.loc[case['key']];case_rows.append(dict(**case,label=int(r.label),probabilities={m:float(r[m]) for m in ['F0','F1_new','T_original','T_A1','H0','H1','H2','P_own_lr','P_cross_lr']},correct={m:bool((r[m]>=.5)==r.label) for m in ['F0','F1_new','T_original','T_A1','H0','H1','H2','P_own_lr','P_cross_lr']}))
 dump(OUT/'cases.json',case_rows)
 (OUT/'CASE_NOTES.md').write_text('# Fixed cases\n\nKeys were selected by SHA256 within stock/phase/original-news strata before new predictions. These are prediction diagnostics, not independent event labels. See cases.json for exact probabilities. No-news T branches must equal new price fallback; H1/H2 may change only when qualified past news exists. A changed or corrected prediction does not establish a causal market mechanism.\n\n'+ '\n'.join(f"- {c['key']}: F0 {'correct' if c['correct']['F0'] else 'wrong'}; A1 {'correct' if c['correct']['T_A1'] else 'wrong'}; history+reaction {'correct' if c['correct']['H2'] else 'wrong'}." for c in case_rows)+'\n')
 lines=['# Four-hour finite new-information experiment','', 'All existing periods are exposed exploratory backtests. The 60% objective is not a guaranteed outcome. No new independent holdout or human extraction review is claimed.','', '## BA, separately by stock and period','', '| Method | AAPL outer monthly | AMZN outer monthly | AAPL development | AMZN development | AAPL later | AMZN later |','|---|---:|---:|---:|---:|---:|---:|']
 for m in refs+methods:
  o=outer.set_index('method').loc[m];values=[o.AAPL,o.AMZN]+[float(plain[(plain.method==m)&(plain.symbol==s)&(plain.phase==p)].BA.iloc[0]) for p in ('development','later') for s in ('AAPL','AMZN')];lines.append('| '+m+' | '+' | '.join(f'{100*v:.2f}%' for v in values)+' |')
 lines += ['', '## Meaning and limitations','','P: own versus cross-stock past prices, with LR, shallow boosting and fixed synthetic TabPFN. T: same protected target-company text and CLS pooling, original versus January–February adapted A1; LR versus rank-two interaction. H: original four-hour news, separate preceding-session news, and already-realized reaction. F1_new/F2_new select C on actual issued probabilities, including new price fallback; they are new controls, not corrections to historical scores.','', 'Outer means weight June, July and August equally. Pooled train_forward_oof includes March–August and is separately reported. C is global across stocks; no hindsight per-stock architecture selection. Brier is a separate diagnostic, and positive temperature never changes direction.','', '## Advancement','',json.dumps(json.loads((OUT/'advancement.json').read_text()),indent=2),'', 'Blends: '+(', '.join(blends) or 'none; no contrast passed the registered gate.')+'. Blend architecture was selected on June–August; any earlier blend figure is post-selection descriptive, not new outer validation. Final blend weights use past stock OOF only.','', '## Training and provenance','','Actual LR/boosting/bilinear fitting is recorded in training_evidence.json. A1 encoders are frozen existing checkpoints, not retrained; original and A1 share token hashes. TabPFN conditions its fixed prior on past samples and does not update its foundation weights. Private model files and conditioning bundles are reloaded to verify predictions. Preliminary runs before upstream fingerprint hardening are preserved privately and excluded from reported results.','', 'The Alpaca 2018 sample requires authentication (HTTP401). No external SPY/QQQ features are included and no coverage assumption is made.','', 'Raw news, price features, model weights and environment files remain private. Joint-date paired 1/5-day block intervals are descriptive, not selection-adjusted significance tests. Exact coverage and fixed prediction cases are supplied; they do not prove causal explanations.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(outer.to_string(index=False),flush=True)
 # Small offline report/replay, no raw text.
 records=d[d.phase.isin(['development','later'])][['key','symbol','phase','label']+refs+methods].to_dict('records');payload=json.dumps(records);head='<html><meta charset="utf-8"><title>Four-hour exploration</title><style>body{font:16px system-ui;max-width:1200px;margin:35px auto}td,th{padding:7px;border-bottom:1px solid #ddd}table{border-collapse:collapse}select{padding:8px}</style><h1>Four-hour finite exploration</h1><p>Exposed historical backtest. Saved-prediction replay; not live forecasting.</p>'
 script='''<h2>Window replay</h2><select id="row"></select><pre id="detail"></pre><script>const rows=PAYLOAD;const sel=document.getElementById('row');rows.forEach((r,i)=>sel.add(new Option(r.key,i)));function show(){document.getElementById('detail').textContent=JSON.stringify(rows[sel.value||0],null,2)}sel.onchange=show;show()</script>'''.replace('PAYLOAD',payload)
 (OUT/'demo.html').write_text(head+plain[plain.phase.isin(['development','later'])].to_html(index=False,float_format=lambda x:f'{x:.4f}')+script+'</html>')
if __name__=='__main__':main()
