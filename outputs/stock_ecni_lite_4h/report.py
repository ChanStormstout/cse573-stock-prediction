from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent; OUT=HERE/'v1'

def pct(x): return 'n/a' if pd.isna(x) else f'{100*x:.2f}%'

def main():
    v=json.loads((OUT/'VERIFICATION.json').read_text())
    if v['status']!='PASS': raise RuntimeError('verification is not PASS')
    m=pd.read_csv(OUT/'metrics.csv'); g=pd.read_csv(OUT/'advancement.csv'); t=pd.read_csv(OUT/'transitions.csv')
    ex=json.loads((OUT/'EXECUTION.json').read_text()); audit=json.loads((OUT/'INPUT_AUDIT.json').read_text())
    labels={'BASE':'Price + full-body sparse baseline','D':'Baseline + dissemination metadata','U':'Baseline + provisional event updates','A':'Baseline + past reaction analogies','AL':'Baseline + analogies + saved LLM response','DUA':'Baseline + dissemination + updates + analogies','DUAL':'Baseline + all shared blocks','DUAL_PARTIAL':'All blocks + shrunk stock contrast'}
    lines=['# ECNI-lite four-hour combination','',
      '**Status:** verified exposed exploratory chronological historical backtest. All periods have already influenced project development.','',
      '## What was tested','',
      'The canonical price + full-body sparse probability was held fixed. Small gated residual corrections tested dissemination metadata, provisional company-event updates, past-only analogous reactions, and an existing saved LLM outcome-response score. When no eligible evidence exists, the output equals the baseline probability exactly.','',
      f"Training-period selection chose **{labels.get(ex['selected'],ex['selected'])}**. This choice used only the preregistered June-August advancement screen.",'',
      '## Main results','']
    for phase,title in [('train_forward_oof','Training forward OOF'),('development','Development (exposed)'),('later','Later (exposed)')]:
        lines += ['',f'### {title}','', '| Method | AAPL BA | AAPL Brier | AMZN BA | AMZN Brier |','|---|---:|---:|---:|---:|']
        q=m[m.phase==phase]
        for method in ['BASE']+list(labels)[1:]:
            a=q[(q.method==method)&(q.symbol=='AAPL')];z=q[(q.method==method)&(q.symbol=='AMZN')]
            if a.empty or z.empty: continue
            lines.append(f"| {labels[method]} | {pct(a.BA.iloc[0])} | {a.Brier.iloc[0]:.4f} | {pct(z.BA.iloc[0])} | {z.Brier.iloc[0]:.4f} |")
    lines += ['','## Preregistered advancement screen','', '| Method | AAPL mean dBA | AMZN mean dBA | Positive macro months | Pass |','|---|---:|---:|---:|:---:|']
    for r in g.itertuples(): lines.append(f"| {labels[r.method]} | {100*r.AAPL_delta_BA:+.2f} pp | {100*r.AMZN_delta_BA:+.2f} pp | {r.positive_macro_months}/3 | {'yes' if r.passes else 'no'} |")
    lines += ['','## Evidence coverage','', '| Block | AAPL windows | AMZN windows |','|---|---:|---:|']
    for b,c in audit['coverage'].items(): lines.append(f"| {b} | {c['AAPL']} | {c['AMZN']} |")
    lines += ['','## Interpretation','',
      '- Passing the advancement screen would mean a feature block improved both stocks during June-August under the frozen rule. It would still require a new untouched period for confirmation.',
      '- Failing the screen means the combined representation did not earn promotion; exposed development or later scores cannot reverse that decision.',
      '- Event facts are provisional model outputs rather than independently reviewed gold. Historical reactions are associations and do not establish causality.',
      '- The saved LLM control reuses earlier outputs. No new LLM call was made.', '',
      '## Verification','',f"Fit-free reload/replay: **{v['status']}**. Maximum probability error `{v['maximum_probability_error']:.3g}`; fallback errors `{v['fallback_errors']}`; verifier fit calls `{v['fit_calls']}`."]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Cases are selected by behavior category, before adding explanatory prose.
    p=pd.read_csv(OUT/'predictions.csv')
    selected=ex['selected']; base=p.BASE>=.5; new=p[selected]>=.5; y=p.label.astype(bool)
    cases=[]
    for name,mask in [('corrected',(new==y)&(base!=y)),('introduced_error',(new!=y)&(base==y)),('unchanged_correct',(new==y)&(base==y)),('unchanged_error',(new!=y)&(base!=y))]:
        q=p[mask].head(5)
        for r in q.itertuples(): cases.append({'category':name,'key':r.key,'symbol':r.symbol,'month':r.month,'base_probability':r.BASE,'final_probability':getattr(r,selected),'label':r.label,'articles':r.articles,'clusters':r.clusters,'analog_cases':r.n_cases})
    pd.DataFrame(cases).to_csv(OUT/'case_panel.csv',index=False)
    (OUT/'CASE_NOTES.md').write_text('# Fixed behavior case panel\n\nCases are selected mechanically from corrected, introduced-error, unchanged-correct and unchanged-error groups. See `case_panel.csv`. No causal explanation is assigned without reading the underlying evidence.\n')
    print(OUT/'REPORT.md')

if __name__=='__main__': main()
