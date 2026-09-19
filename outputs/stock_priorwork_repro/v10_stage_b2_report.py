"""Create the frozen Stage B2 report and classical-lane summary after PASS."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/stock_priorwork_repro/v10'
def table(path,cols=None):
 d=pd.read_csv(O/path);return d if cols is None else d[cols]
def main():
 a=json.loads((O/'STAGE_B2_FINAL_AUDIT.json').read_text());
 if a['status']!='PASS':raise SystemExit('STAGE_B2_AUDIT_NOT_PASS')
 m4=table('JOINT_METRICS_4H.csv',['method','stock','phase','n','BA','Brier']);m1=table('JOINT_METRICS_1D.csv',['horizon_window','method','stock','phase','n','BA','Brier']);c4=table('INCREMENTAL_COMPARISON_4H.csv');c1=table('INCREMENTAL_COMPARISON_1D.csv');cons=json.loads((O/'INCREMENTAL_CONSISTENCY_SUMMARY.json').read_text());f1=json.loads((O/'F1_COMPARISON.json').read_text());window=json.loads((O/'DAILY_WINDOW_COMPARISON.json').read_text())
 report='# V10 Stage B2 frozen NEWS+PRICE report\n\nStatus: **PASS**. All 18 frozen branches and 108 monthly models were executed and independently replayed. No B2 result selected or tuned a model.\n\n## Four-hour joint results\n\n'+m4.to_markdown(index=False)+'\n\n## Daily joint results\n\n'+m1.to_markdown(index=False)+'\n\n## Four-hour system-level increments\n\n'+c4[['method','stock','phase','DELTA_BA_VS_R1','DELTA_BA_VS_NEWS_ONLY']].to_markdown(index=False)+'\n\n## Daily system-level increments\n\n'+c1[['horizon_window','method','stock','phase','DELTA_BA_VS_DPRICE','DELTA_BA_VS_NEWS_ONLY']].to_markdown(index=False)+'\n\nThese are system-level comparisons when classifier families differ; they are not pure causal attribution to news. Development/later are historical evaluations under their registered labels, not untouched tests.\n'
 (O/'STAGE_B2_REPORT.md').write_text(report)
 dom=[k for k,v in f1.items() if v['STRICTLY_DOMINATES_F1']];daily=[k for k,v in cons['daily'].items() if v['ALL_FOUR_POSITIVE_DELTA_VS_DPRICE']];knn=cons['daily'].get('1d:DNEWS_24H|TFIDF_KNN',{}).get('ALL_FOUR_POSITIVE_DELTA_VS_DPRICE',False)
 summary='# V10 classical lane final summary\n\nRandom-split prior-work scores are not directly comparable because they allow temporally mixed train/evaluation samples. V9 replaced that protocol with chronological issued NEWS-only models and full March--August grids. Stage B1 independently verified a seven-feature DPRICE baseline and froze three NEWS methods per setting using only March--August issued evidence. Stage B2 then executed every frozen NEWS+PRICE branch without a new search.\n\n## Findings\n\n- Four-hour methods descriptively strictly dominating historical F1 across all four stock/phase cells: '+(', '.join(dom) if dom else 'none')+'.\n- Daily methods with positive system-level BA deltas versus DPRICE in all four cells: '+(', '.join(daily) if daily else 'none')+'.\n- Frozen 24-hour KNN all-four positive pattern: '+str(knn)+'.\n- Matched PAPER_2G 24-hour consistently stronger than overnight: '+str(window['24H_CONSISTENTLY_STRONGER_THAN_OVERNIGHT_FOR_PAPER_2G'])+'.\n\nThese are descriptive exposed historical evaluations. Joint-versus-price comparisons may also change classifier family, so they do not isolate a causal news contribution. The evidence covers only AAPL and AMZN, is not prospective financial validation, and does not establish state of the art. No B2 branch was promoted after seeing results.\n\nCLASSICAL_LANE_FROZEN\n'
 (O/'CLASSICAL_LANE_FINAL_SUMMARY.md').write_text(summary);print('V10_CLASSICAL_LANE_REPORT_COMPLETE')
if __name__=='__main__':main()
