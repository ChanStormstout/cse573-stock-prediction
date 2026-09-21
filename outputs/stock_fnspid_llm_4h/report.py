#!/usr/bin/env python3
from pathlib import Path
import json,pandas as pd
HERE=Path(__file__).resolve().parent;OUT=HERE/'v1'
def pct(x):return f'{100*float(x):.2f}%'
def main():
 v=json.loads((OUT/'VERIFICATION.json').read_text());
 if v['status']!='PASS':raise RuntimeError('verification failed')
 m=pd.read_csv(OUT/'metrics.csv');g=json.loads((OUT/'promotion_gate.json').read_text());a=json.loads((OUT/'ARTICLE_JOB_AUDIT.json').read_text());aa=json.loads((OUT/'ANALOGY_JOB_AUDIT.json').read_text());pred=pd.read_csv(OUT/'predictions.csv')
 names={'PRICE_R1':'Recent-price reference','AUG_UNFILTERED_FINBERT':'Unfiltered augmented FinBERT','ORIG_FINBERT':'Original-course FinBERT','L1_FILTERED_FINBERT':'L1 LLM-filtered FinBERT','L2_FACT_CHANGE':'L2 LLM fact-change residual','L3_CASE_VOTE':'L3 historical-case vote','L4_LLM_ANALOGY':'L4 historical cases + LLM','L5_GUARDED_SYSTEM':'L5 guarded system'}
 lines=['# FNSPID-augmented LLM information-control experiment','','## Plain-language result','',f"L1-L4 were executed in the registered order. The OOF-only L5 choice is **{names[g['L5_choice']]}**. Development and later remain exposed exploratory backtests.",'','## Actual work','',f"- L1/L2: {a['jobs']:,} frozen-Qwen binary article scores across {a['articles']:,} added FNSPID articles.",f"- L4: {aa['jobs']:,} time-safe case-conditioned Qwen scores; {aa['fallback_rows']} rows use exact fallback.",'- L3: non-LLM similarity-weighted votes using strictly earlier same-stock cases.','- L1 and L2 refit their downstream models chronologically; LLM weights were frozen.','- Independent post-run verification is PASS.','','## Results','']
 for phase in ('train_forward_oof','development','later'):
  lines += [f'### {phase}','','| Stock | Method | BA | Accuracy | MCC | Brier | AUC |','|---|---|---:|---:|---:|---:|---:|']
  for r in m[m.phase.eq(phase)].itertuples(index=False):lines.append(f'| {r.symbol} | {names[r.method]} | {pct(r.BA)} | {pct(r.accuracy)} | {r.MCC:.3f} | {r.Brier:.4f} | {r.AUC:.4f} |')
  lines.append('')
 lines += ['## L5 training-period gate','','| Component | Macro ΔBA | AAPL ΔBA | AMZN ΔBA | Positive months | Max ΔBrier | Pass |','|---|---:|---:|---:|---:|---:|---:|']
 for x in g['components']:lines.append(f"| {names[x['method']]} | {100*x['macro_delta_BA']:+.2f} pp | {100*x['stock_delta_BA']['AAPL']:+.2f} pp | {100*x['stock_delta_BA']['AMZN']:+.2f} pp | {x['positive_macro_months']}/6 | {max(x['stock_delta_Brier'].values()):+.4f} | {x['passes']} |")
 lines += ['','## Coverage and evidence boundary','',f"L1 retained added FNSPID evidence in {int((pred.l1_fnspid>0).sum())}/{len(pred)} canonical windows. L2 strictly equals the price model in every row without accepted FNSPID evidence.",'','The LLM scores are frozen-model judgments, not independently reviewed gold labels. FNSPID entity links also remain without completed independent semantic review, and source timestamps are date-only. Scores therefore show how this fixed pipeline behaved, not that the LLM classifications are objectively correct or that the result will generalize to a fresh market period.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('wrote report')
if __name__=='__main__':main()
