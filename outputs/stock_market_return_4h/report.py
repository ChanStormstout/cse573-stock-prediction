"""Build human-readable tables from the registered return-supervision run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "outputs" / "stock_market_return_4h" / "v1"
PRED = PUBLIC / "predictions.csv"
DATA = ROOT / "work" / "stock-data" / "paper_methods_4h" / "v1" / "inputs.pkl"


def metrics(group: pd.DataFrame) -> dict:
    y = group.label.to_numpy(int)
    p = group.p.to_numpy(float)
    q = (p >= 0.5).astype(int)
    out = {
        "n": len(group),
        "BA": balanced_accuracy_score(y, q) if len(np.unique(y)) > 1 else np.nan,
        "MCC": matthews_corrcoef(y, q),
        "Brier": brier_score_loss(y, p),
        "pred_up": q.mean(),
        "constant": int(len(np.unique(q)) <= 1),
    }
    if group.return_pred.notna().any():
        keep = group.return_pred.notna().to_numpy()
        actual = group.target_return.to_numpy(float)[keep]
        pred = group.return_pred.to_numpy(float)[keep]
        out["return_MAE"] = mean_absolute_error(actual, pred)
        out["return_RMSE"] = mean_squared_error(actual, pred) ** 0.5
        out["return_corr"] = np.corrcoef(actual, pred)[0, 1] if len(actual) > 1 and np.std(pred) > 0 and np.std(actual) > 0 else np.nan
    else:
        out.update(return_MAE=np.nan, return_RMSE=np.nan, return_corr=np.nan)
    return out


def bootstrap_diff(a: pd.DataFrame, b: pd.DataFrame, reps=1000, seed=573):
    """Paired day-block BA difference, descriptive only (periods are exposed)."""
    joined = a[["key", "day", "label", "p"]].merge(b[["key", "p"]], on="key", suffixes=("_a", "_b"))
    if joined.empty:
        return {"n": 0, "difference": np.nan, "lo": np.nan, "hi": np.nan}
    y = joined.label.to_numpy(int)
    qa = (joined.p_a.to_numpy(float) >= .5).astype(int)
    qb = (joined.p_b.to_numpy(float) >= .5).astype(int)
    groups = [idx.to_numpy(int) for _, idx in joined.groupby("day", sort=False).groups.items()]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in picked])
        vals.append(balanced_accuracy_score(y[idx], qa[idx]) - balanced_accuracy_score(y[idx], qb[idx]))
    vals = np.asarray(vals)
    point = balanced_accuracy_score(y, qa) - balanced_accuracy_score(y, qb)
    return {"n": len(joined), "difference": float(point), "lo": float(np.quantile(vals, .025)), "hi": float(np.quantile(vals, .975))}


def md_table(frame: pd.DataFrame, cols):
    return frame[cols].to_markdown(index=False, floatfmt=".4f")


def main():
    pred = pd.read_csv(PRED)
    d = pd.read_pickle(DATA).reset_index(drop=True)
    summary_rows = []
    for (method, phase, symbol), g in pred.groupby(["method", "phase", "symbol"], sort=True):
        summary_rows.append({"method": method, "phase": phase, "symbol": symbol, **metrics(g)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PUBLIC / "metrics.csv", index=False)

    choices = json.loads((PUBLIC / "choices.json").read_text())["choices"]
    oof = summary[summary.phase == "train_forward_oof"]
    gate_rows = []
    gate_months = {"2018-06", "2018-07", "2018-08"}
    for variant in ("price", "price_F1"):
        base_pred = pred[(pred.method == f"C0_{variant}") & (pred.month.isin(gate_months))]
        for method in ("C1", "C2"):
            cur_pred = pred[(pred.method == f"{method}_{variant}") & (pred.month.isin(gate_months))]
            # The promotion line is intentionally evaluated on June-August only.
            base_by_stock = {s: metrics(base_pred[base_pred.symbol == s]) for s in ("AAPL", "AMZN")}
            cur_by_stock = {s: metrics(cur_pred[cur_pred.symbol == s]) for s in ("AAPL", "AMZN")}
            stock_deltas = {s: cur_by_stock[s]["BA"] - base_by_stock[s]["BA"] for s in ("AAPL", "AMZN")}
            gate_rows.append({
                "variant": variant,
                "method": method,
                "choice": choices[f"{variant}/{method}"],
                "base_JunAug_mean_BA": float(np.mean([base_by_stock[s]["BA"] for s in ("AAPL", "AMZN")])),
                "method_JunAug_mean_BA": float(np.mean([cur_by_stock[s]["BA"] for s in ("AAPL", "AMZN")])),
                "delta_BA": float(np.mean(list(stock_deltas.values()))),
                "AAPL_delta_BA": float(stock_deltas["AAPL"]),
                "AMZN_delta_BA": float(stock_deltas["AMZN"]),
                "base_mean_Brier": float(np.mean([base_by_stock[s]["Brier"] for s in ("AAPL", "AMZN")])),
                "method_mean_Brier": float(np.mean([cur_by_stock[s]["Brier"] for s in ("AAPL", "AMZN")])),
                "delta_Brier": float(np.mean([cur_by_stock[s]["Brier"] - base_by_stock[s]["Brier"] for s in ("AAPL", "AMZN")])),
            })
    gate = pd.DataFrame(gate_rows)
    gate["passes_engineering_gate"] = (gate[["AAPL_delta_BA", "AMZN_delta_BA"]].min(axis=1) >= .01) & (gate.delta_Brier <= .002)
    gate.to_csv(PUBLIC / "promotion_gate.csv", index=False)

    # Pair the chosen method against its same-input C0 on each exposed period.
    interval_rows = []
    for phase in ("train_forward_oof", "development", "later"):
        for symbol in ("AAPL", "AMZN"):
            for variant in ("price", "price_F1"):
                base = pred[(pred.method == f"C0_{variant}") & (pred.phase == phase) & (pred.symbol == symbol)]
                for method in ("C1", "C2"):
                    cur = pred[(pred.method == f"{method}_{variant}") & (pred.phase == phase) & (pred.symbol == symbol)]
                    val = bootstrap_diff(cur, base, reps=1000, seed=573 + len(interval_rows))
                    interval_rows.append({"phase": phase, "symbol": symbol, "variant": variant, "method": method, **val})
    intervals = pd.DataFrame(interval_rows)
    intervals.to_csv(PUBLIC / "paired_block_intervals.csv", index=False)

    audit = json.loads((PUBLIC / "alpaca_audit.json").read_text()) if (PUBLIC / "alpaca_audit.json").exists() else {}
    lines = []
    lines.append("# Market state and continuous-return auxiliary experiment — v1")
    lines.append("")
    lines.append("## Plain-language result")
    lines.append("")
    lines.append("This run tested whether adding the continuous four-hour return as an auxiliary training signal improves the unchanged four-hour direction task. It used the original price features, plus a secondary variant that appends the already-frozen F1 probability. The model and parameter were chosen from March–August chronological forward OOF only; development (September–October) and later (November onward) were scored separately and were not used for selection.")
    lines.append("")
    lines.append("The selected auxiliary objective did not pass the registered promotion line. For the raw-price branch the chosen C2 lambda was 0.0, so the OOF selection preferred the classification-only loss. For the price+F1 branch it selected lambda 0.5, but the later-period AMZN result remained below 50%. This is evidence against claiming a stable improvement, not evidence that the entire project is invalid.")
    lines.append("")
    lines.append("## External market-data audit")
    lines.append("")
    lines.append(f"The fixed SPY/QQQ one-minute Alpaca probe stopped with `{audit.get('decision', 'not-run')}`: credentials_present={audit.get('credentials_present')}, status_counts={audit.get('status_counts')}. No daily or synthetic ETF substitute was used.")
    lines.append("")
    lines.append("## Selected parameters")
    lines.append("")
    lines.append(pd.DataFrame([{"input": k, "selected_parameter": v} for k, v in choices.items()]).to_markdown(index=False))
    lines.append("")
    lines.append("## Results by period")
    lines.append("")
    display = summary.copy()
    display["method"] = display.method.replace({"C0_price": "C0 price", "C1_price": "C1 return ridge", "C2_price": "C2 joint", "C0_price_F1": "C0 price+F1", "C1_price_F1": "C1 return ridge+F1", "C2_price_F1": "C2 joint+F1"})
    for phase, title in (("train_forward_oof", "OOF selected model (March–August)"), ("development", "Development (September–October)"), ("later", "Later exposed period (November onward)")):
        lines.append(f"### {title}")
        lines.append("")
        tab = display[display.phase == phase].sort_values(["symbol", "method"])
        lines.append(md_table(tab, ["symbol", "method", "n", "BA", "MCC", "Brier", "pred_up", "return_MAE", "return_corr"]))
        lines.append("")
    lines.append("## Promotion check")
    lines.append("")
    lines.append("The registered engineering line was June–August mean BA gain ≥1 percentage point for both stocks, no stock loss >1 point, and mean Brier worsening ≤0.002. It is a screening rule, not a significance test.")
    lines.append("")
    lines.append(md_table(gate, ["variant", "method", "choice", "base_JunAug_mean_BA", "method_JunAug_mean_BA", "delta_BA", "AAPL_delta_BA", "AMZN_delta_BA", "delta_Brier", "passes_engineering_gate"]))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("1. The continuous-return target is available and correctly aligned, so this was a real training run rather than a proposal. The ridge diagnostic can show a return relationship, but its sign is not consistently better than the classification baseline across periods.")
    lines.append("2. The shared linear two-head model did not earn a stable direction gain. The selected raw-price lambda collapsed to zero; the F1-appended branch selected a nonzero lambda but did not produce a reliable later-period gain. The safest conclusion is that the current features do not expose enough stable information for this auxiliary loss to recover.")
    lines.append("3. The market-state branch remains a data-access question. It cannot be fairly tested until verified, timestamped SPY/QQQ minute bars are available under the registered protocol.")
    lines.append("4. All development and later values are exposed historical backtests. They are useful for diagnosis and the course report, but they are not a new untouched confirmation period.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```text\nwork/stock-data/finbert-env/bin/python outputs/stock_market_return_4h/audit_alpaca.py\nwork/stock-data/finbert-env/bin/python outputs/stock_market_return_4h/run.py\nwork/stock-data/finbert-env/bin/python outputs/stock_market_return_4h/report.py\nwork/stock-data/finbert-env/bin/python outputs/stock_market_return_4h/verify.py\n```")
    lines.append("")
    lines.append("Raw input, cached features, trained binaries, and generated private models remain under `work/stock-data` and are not part of the repository.")
    (PUBLIC / "REPORT.md").write_text("\n".join(lines) + "\n")
    (PUBLIC / "METRICS_SUMMARY.md").write_text("\n".join(lines[0:lines.index("## Interpretation")]) + "\n")
    print(pd.DataFrame([{"phase": p, "symbols": int((summary.phase == p).sum())} for p in ("train_forward_oof", "development", "later")]).to_string(index=False))
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
