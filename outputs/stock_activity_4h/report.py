"""Render a verified Activity v1 report; never run before independent PASS."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
def main(root):
 verify=json.loads((root/"verification.json").read_text())
 if verify.get("status")!="PASS":raise SystemExit("Activity report requires a PASS independent verifier.")
 metrics=pd.read_csv(root/"metrics.csv");monthly=pd.read_csv(root/"monthly_metrics.csv");adv=json.loads((root/"advancement.json").read_text());attr=json.loads((root/"attribution_control.json").read_text());coverage=pd.read_csv(root.parent/"activity_feature_coverage_by_month.csv")
 lines=["# Activity Incremental Experiment v1","","Activity remains `SEMANTICS_UNRESOLVED_OPAQUE_ACTIVITY`.","","## March-August forward OOF aggregate","",metrics[metrics.phase.eq("train_forward_oof")].to_markdown(index=False),"","## June-August registered gate months","",monthly[monthly.month.isin(["2018-06","2018-07","2018-08"])].to_markdown(index=False),"","## Primary advancement: A1 vs A0","",pd.DataFrame([adv]).to_markdown(index=False),"","## Attribution control: A1_matchedC vs A0","",pd.DataFrame([attr]).to_markdown(index=False),"","## Activity coverage","",coverage.to_markdown(index=False),"","## EXPOSED EXPLORATORY HISTORICAL BACKTEST","",metrics[metrics.phase.isin(["development","later"])].to_markdown(index=False),""]
 if not adv["passes"]: lines += ["The frozen Activity v1 construction did not satisfy the preregistered stable two-stock incremental gate. This does not establish that the opaque raw field is generally useless."]
 elif attr["AAPL_mean_delta_BA"]>0 and attr["AMZN_mean_delta_BA"]>0: lines += ["The activity-enhanced pipeline passed the preregistered gate, and the matched-C diagnostic is directionally consistent with an activity-feature contribution."]
 else: lines += ["The activity-enhanced pipeline passed, but regularization-selection differences prevent clean attribution of the improvement to the activity fields alone."]
 (root/"REPORT.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);main(p.parse_args().root)
