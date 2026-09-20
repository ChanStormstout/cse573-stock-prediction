"""Report only independently replayed comparisons, never select outer winners."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent/'v1'
def table(df,keys):
 rows=['| Method | Stock | BA mean ± seed SD | Brier mean | Changed / repaired / introduced, mean |','|---|---|---:|---:|---:|']
 for k,g in df.groupby(keys,sort=False):
  method,stock=k
  rows.append(f"| {method} | {stock} | {g.BA.mean()*100:.2f}% ± {g.BA.std()*100:.2f} | {g.Brier.mean():.4f} | {g.changed.mean():.1f} / {g.repaired.mean():.1f} / {g.introduced.mean():.1f} |")
 return '\n'.join(rows)
def main():
 assert json.loads((ROOT/'CLASSICAL_VERIFICATION.json').read_text())['status']=='PASS'
 text='# FULL and target-paragraph semantic extension\n\n## Interpretation\n\nAll results are exploratory random backtests on previously exposed outer splits. Random historical classification is not unseen-future forecasting. The owner redirected this lane after seeing prior baseline scores; no new design claims an unobserved outer test. Hyperparameters and finite fusion weights are chosen inside training partitions. Both stocks and all three seeds are retained.\n\n## How to read the tables\n\nBA averages the recognition rate for rising and falling windows; higher is better. Brier measures probability error; lower is better. FULL means past prices plus binary words from the accepted full news bodies. PAPER is the sparse word-only reference; PRICE uses recent prices only. FinBERT and FinModernBERT are financial-language encoders that turn text into numerical features. Neither encoder is finetuned in this experiment.\n\n## Traditional finite comparison\n\n'
 text+=table(pd.read_csv(ROOT/'CLASSICAL_METRICS.csv'),['method','symbol'])
 text+='\n\nTFIDF methods here use text only; FULL combines binary full-body word features and past price features. Therefore text-only versus FULL is a practical comparator, not an isolated representation ablation. SVM probabilities use inner training-only sigmoid calibration. Seed SD describes split variation, not an independent confidence interval. Changed/repaired/introduced counts are relative to FULL.\n'
 if (ROOT/'TITLE_FUSION_AUDIT.json').exists():
  assert json.loads((ROOT/'TITLE_FUSION_AUDIT.json').read_text())['status']=='PASS_FIT_FREE_ARITHMETIC_REPLAY'
  text+='\n## FULL plus existing title-encoder controls\n\n'+table(pd.read_csv(ROOT/'TITLE_FUSION_METRICS.csv'),['encoder','symbol'])+'\n\nThese finite mixtures use existing inner-selected title heads and choose only a mixing weight on inner validation. They help distinguish combination benefits from longer-text benefits, but are not a perfectly isolated input ablation: body-head C and weight are jointly selected and its evidence gate differs. No learned stacker is fitted.\n'
 if (ROOT/'SEMANTIC_VERIFICATION.json').exists():
  assert json.loads((ROOT/'SEMANTIC_VERIFICATION.json').read_text())['status']=='PASS'
  text+='\n## FULL plus frozen target-paragraph semantics\n\n'+table(pd.read_csv(ROOT/'SEMANTIC_METRICS.csv'),['encoder','symbol'])+'\n\nEncoder weights are frozen. Newly fitted objects are per-fold PCA/scalers, small logistic classifiers, and an inner-selected finite mixing weight. Weight zero is allowed; absent target evidence returns FULL exactly. This is not LLM finetuning, attention or a learned stacker. Paragraph matching uses explicit company names and remains unvalidated for semantic target role. Both encoders receive the same full selected paragraph content in common chunks.\n'
 else:text+='\n## Target-paragraph semantic experiment\n\nPENDING: do not infer a result from input preparation or encoding progress.\n'
 text+='\n## Historical-case LLM\n\n1,972 outputs preserved with fingerprints and processes suspended. Existing fold/time membership verification passed, but prompt coverage is incomplete. No partial-output selected-window predictive score is reported; full inference has not resumed.\n'
 (ROOT/'REPORT.md').write_text(text)
if __name__=='__main__':main()
