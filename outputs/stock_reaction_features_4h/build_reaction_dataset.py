"""Build the full-corpus target-association and same-session reaction audit.

Only aggregate audit files are public.  Article text, evidence sentences and
the pair-level reaction table stay under ``work/stock-data``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stock_recency_dense_4h"))
from common import RAW, WORK, dump, schedule, sha

OUT_ROOT = Path(__file__).resolve().parent.parent / "stock_reaction_features_4h"
DEFAULT_PUBLIC = OUT_ROOT / "v1"
DEFAULT_PRIVATE = WORK / "reaction_features_4h" / "v1"
NEWS_RAW = WORK / "raw" / "news"
TICKER = {"AAPL": re.compile(r"(?<![A-Z0-9])AAPL(?![A-Z0-9])", re.I), "AMZN": re.compile(r"(?<![A-Z0-9])AMZN(?![A-Z0-9])", re.I)}
LEGAL = {"AAPL": re.compile(r"\bApple\s*(?:,?\s*Inc\.?|Corporation)\b", re.I), "AMZN": re.compile(r"\bAmazon(?:\.com)?\s*(?:,?\s*Inc\.?|Corporation)\b", re.I)}
COMMON = {"AAPL": re.compile(r"\bApple\b", re.I), "AMZN": re.compile(r"\bAmazon\b", re.I)}
EXCLUDE = {"AAPL": ("apple pie", "apple valley", "apple sauce", "apple picking", "apple orchard", "apple tree"), "AMZN": ("amazon rainforest", "amazon river", "amazonian", "amazon jungle")}


def text_sentences(text: str):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text or "") if x.strip()]


def contains_target(pattern, text):
    return bool(pattern.search(text or ""))


def evidence_for(symbol: str, title: str, body: str):
    title = title or ""; body = body or ""; all_text = title + "\n" + body
    low = all_text.casefold()
    if contains_target(TICKER[symbol], all_text) or contains_target(LEGAL[symbol], all_text):
        sent = next((i for i, s in enumerate(text_sentences(all_text)) if contains_target(TICKER[symbol], s) or contains_target(LEGAL[symbol], s)), None)
        return {"tier": 0, "evidence_type": "ticker_or_legal", "sentence_id": sent}
    if contains_target(COMMON[symbol], title) and not any(x in low for x in EXCLUDE[symbol]):
        sent = 0 if title else None
        return {"tier": 1, "evidence_type": "title_company_name", "sentence_id": sent}
    sentences = text_sentences(body)
    for i, sent in enumerate(sentences):
        if contains_target(COMMON[symbol], sent) and not any(x in sent.casefold() for x in EXCLUDE[symbol]):
            lo = max(0, i - 1); hi = min(len(sentences), i + 2)
            return {"tier": 2, "evidence_type": "body_company_sentence", "sentence_id": i, "context_sentence_ids": list(range(lo, hi))}
    return None


def valid_time(x):
    try:
        t = pd.to_datetime(x, utc=True, errors="coerce")
        return t if pd.notna(t) else None
    except Exception:
        return None


def make_group(symbol, row):
    for field, label in (("normalized_hash", "normalized_full_text"), ("exact_hash", "exact_text"), ("title_norm", "title_fallback")):
        value = str(row.get(field, "") or "").strip()
        if value and value.lower() != "nan":
            return f"{symbol}|{label}|{value}", label
    return f"{symbol}|article|{row['article_key']}", "article_key"


def load_article(zips, row):
    archive = row.archive
    z = zips.get(archive)
    if z is None:
        z = zipfile.ZipFile(NEWS_RAW / archive); zips[archive] = z
    try:
        obj = json.loads(z.read(row.member))
        title = str(obj.get("title") or row.title or "")
        body = str(obj.get("text") or "")
        return title, body, obj
    except Exception:
        return str(row.title or ""), "", {}


def build_pairs(index: pd.DataFrame, private: Path):
    zips = {}; candidates = []; counts = Counter(); schema_keys = set(); sample_objs = []
    for n, row in enumerate(index.itertuples(index=False), 1):
        article_key = f"{row.archive}::{row.member}"
        try:
            title, body, obj = load_article(zips, row)
        except Exception:
            counts["json_read_error"] += 1; continue
        if n <= 5:
            schema_keys.update(obj.keys()); sample_objs.append({"top_level_keys": sorted(obj.keys()), "text_type": type(obj.get("text")).__name__, "entity_keys": sorted((obj.get("entities") or {}).keys()) if isinstance(obj.get("entities"), dict) else []})
        counts["raw_rows_seen"] += 1
        language = str(row.language or "").casefold()
        if language not in {"english", "en"}:
            counts["non_english"] += 1; continue
        counts["english"] += 1
        if len(body.strip()) < 100:
            counts["short_text"] += 1; continue
        counts["sufficient_text"] += 1
        pub = valid_time(row.published_utc); crawled = valid_time(row.crawled_utc)
        if pub is None and crawled is None:
            counts["invalid_time"] += 1; continue
        available = max(x for x in (pub, crawled) if x is not None)
        for symbol in ("AAPL", "AMZN"):
            ev = evidence_for(symbol, title, body)
            if ev is None: continue
            counts[f"associated_{symbol}"] += 1; counts[f"tier{ev['tier']}_{symbol}"] += 1
            g, gtype = make_group(symbol, {**row._asdict(), "article_key": article_key})
            candidates.append({"article_key": article_key, "archive": row.archive, "member": row.member, "symbol": symbol, "title": title, "body": body, "url": str(row.url or ""), "site": str(row.site or ""), "published_utc": pub, "crawled_utc": crawled, "available_utc": available, "language": language, "tier": int(ev["tier"]), "evidence_type": ev["evidence_type"], "evidence_sentence_id": ev.get("sentence_id"), "context_sentence_ids": ev.get("context_sentence_ids", []), "group_key": g, "dedup_basis": gtype, "normalized_hash": str(row.normalized_hash or ""), "exact_hash": str(row.exact_hash or ""), "title_norm": str(row.title_norm or "")})
    for z in zips.values(): z.close()
    cand = pd.DataFrame(candidates)
    if cand.empty: raise RuntimeError("no target pairs")
    cand["available_utc"] = pd.to_datetime(cand.available_utc, utc=True); cand = cand.sort_values(["symbol", "group_key", "available_utc", "article_key"])
    canonical = cand.groupby(["symbol", "group_key"], as_index=False, sort=False).first()
    counts["candidate_pairs"] = int(len(cand)); counts["canonical_groups_all_time"] = int(len(canonical)); counts["duplicate_suppressed_pairs"] = int(len(cand)-len(canonical))
    private.mkdir(parents=True, exist_ok=True); cand.to_pickle(private / "all_candidate_pairs.pkl"); canonical.to_pickle(private / "canonical_pairs_all_time.pkl")
    return cand, canonical, counts, schema_keys, sample_objs


def bars_and_reactions(canonical, private):
    sessions = schedule(); sessions["day"] = sessions.open.dt.strftime("%Y-%m-%d")
    out=[]
    bar_maps={}
    for symbol, name in (("AAPL","APPLE"),("AMZN","AMAZON")):
        b=pd.read_csv(RAW/f"{name}5.csv",header=None,names=["date","time","open","high","low","close","activity"])
        b.index=pd.to_datetime(b.date+" "+b.time,format="%Y.%m.%d %H:%M",utc=True); bar_maps[symbol]=b.sort_index()
    horizons=(30,60,120,240)
    for row in canonical.itertuples(index=False):
        avail=pd.Timestamp(row.available_utc).tz_convert("UTC"); eligible=sessions[(sessions.open<=avail)&(sessions.close>avail)]
        rec={"article_key":row.article_key,"symbol":row.symbol,"group_key":row.group_key,"available_utc":avail,"title":row.title,"tier":row.tier,"evidence_type":row.evidence_type,"session_day":None}
        b=bar_maps[row.symbol]
        if len(eligible):
            s=eligible.iloc[0]; rec["session_day"]=s.day; idx=b.index; pos=int(idx.searchsorted(avail,side="left"))
            for h in horizons:
                n=h//5; sl=idx[pos:pos+n]
                valid=len(sl)==n and len(sl)>0 and sl[-1] < s.close and np.all(np.diff(sl.view("i8"))==5*60*10**9) and not b.loc[sl,["open","close"]].isna().any().any()
                rec[f"reaction_{h}m"] = float(np.log(b.loc[sl[-1],"close"]/b.loc[sl[0],"open"])) if valid else np.nan
                rec[f"reaction_{h}m_valid"] = int(valid)
                rec[f"reaction_{h}m_end_utc"] = str(sl[-1]+pd.Timedelta("5min")) if valid else None
        else:
            for h in horizons: rec[f"reaction_{h}m"]=np.nan;rec[f"reaction_{h}m_valid"]=0;rec[f"reaction_{h}m_end_utc"]=None
        out.append(rec)
    rx=pd.DataFrame(out); rx["available_utc"]=pd.to_datetime(rx.available_utc,utc=True); rx.to_pickle(private/"article_reactions_all_time.pkl")
    return rx


def audit(public, private, index, cand, canonical, rx, counts, schema_keys, sample_objs):
    start=pd.Timestamp("2018-01-01",tz="UTC"); end=pd.Timestamp("2018-09-01",tz="UTC")
    window=rx[(rx.available_utc>=start)&(rx.available_utc<end)].copy(); counts["jan_aug_canonical_pairs"] = int(len(window));
    waterfall=[]
    for stage,n in [("raw_index_rows",len(index)),("english_rows",counts["english"]),("sufficient_english_text",counts["sufficient_text"]),("candidate_target_pairs",counts["candidate_pairs"]),("canonical_groups_all_time",counts["canonical_groups_all_time"]),("jan_aug_canonical_groups",len(window))]: waterfall.append({"stage":stage,"count":int(n)})
    pd.DataFrame(waterfall).to_csv(public/"reaction_dataset_waterfall.csv",index=False)
    months=[]
    for (symbol,month),g in window.assign(month=window.available_utc.dt.strftime("%Y-%m")).groupby(["symbol","month"]):
        months.append({"symbol":symbol,"month":month,"unique_groups":int(g.group_key.nunique()),"dates":int(g.session_day.nunique()),**{f"valid_{h}m":int(g[f"reaction_{h}m_valid"].sum()) for h in (30,60,120,240)}})
    pd.DataFrame(months).to_csv(public/"reaction_counts_by_month.csv",index=False)
    dist=[]
    for symbol,g in window.groupby("symbol"):
        for h in (30,60,120,240):
            x=g.loc[g[f"reaction_{h}m_valid"]==1,f"reaction_{h}m"].dropna()
            dist.append({"symbol":symbol,"horizon_minutes":h,"n":int(len(x)),"mean":float(x.mean()) if len(x) else None,"median":float(x.median()) if len(x) else None,"std":float(x.std(ddof=1)) if len(x)>1 else None,"p05":float(x.quantile(.05)) if len(x) else None,"p95":float(x.quantile(.95)) if len(x) else None})
    pd.DataFrame(dist).to_csv(public/"reaction_distribution.csv",index=False)
    gate=[]
    for symbol,g in window.groupby("symbol"):
        monthly=g.assign(month=g.available_utc.dt.strftime("%Y-%m")).groupby("month").size()
        n60=int(g.reaction_60m_valid.sum());n240=int(g.reaction_240m_valid.sum());dates=int(g.session_day.nunique());months50=int((monthly>=50).sum())
        gate.append({"symbol":symbol,"unique_target_article_groups":int(g.group_key.nunique()),"valid_60m":n60,"valid_240m":n240,"dates":dates,"months_with_at_least_50_groups":months50,"passes":bool(n60>=800 and n240>=800 and dates>=100 and months50>=6)})
    gate_pass=bool(gate and all(x["passes"] for x in gate))
    # Fixed deterministic association audit.  Review cards contain text only in
    # private storage; the public file reports counts and tier distributions.
    cards=[]; selection=[]
    for symbol in ("AAPL","AMZN"):
        for tier in (0,1,2):
            pool=cand[(cand.symbol==symbol)&(cand.tier==tier)].sort_values("article_key").drop_duplicates("group_key"); pool=pool.assign(_hash=pool.article_key.map(lambda x:int(hashlib.sha256(f"{symbol}|{tier}|{x}".encode()).hexdigest(),16))) .sort_values("_hash").head(50)
            selection.extend(pool.index.tolist())
            for _,r in pool.iterrows(): cards.append({"symbol":symbol,"tier":tier,"article_key":r.article_key,"title":r.title,"body_context":text_sentences(r.body)[:3],"evidence_sentence_id":r.evidence_sentence_id})
    pd.DataFrame([{"symbol":s,"tier":t,"selected":sum(1 for x in cards if x["symbol"]==s and x["tier"]==t)} for s in ("AAPL","AMZN") for t in (0,1,2)]).to_csv(private/"association_selection.csv",index=False)
    (private/"association_review_cards.jsonl").write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in cards)+"\n")
    audit_rows=[]
    for symbol in ("AAPL","AMZN"):
        for tier in (0,1,2):
            p=[x for x in cards if x["symbol"]==symbol and x["tier"]==tier];audit_rows.append({"symbol":symbol,"tier":tier,"sampled":len(p),"distinct_sites":int(cand[(cand.symbol==symbol)&(cand.tier==tier)].site.nunique()),"note":"Deterministic text review cards are private; no outcomes supplied."})
    pd.DataFrame(audit_rows).to_csv(public/"TARGET_ASSOCIATION_AUDIT.csv",index=False)
    schema={"index_shape":list(index.shape),"index_columns":index.columns.tolist(),"raw_json_top_level_keys":sorted(schema_keys),"sample_schema":sample_objs,"text_field":"text","entities_field":"entities","availability_rule":"max(valid published_utc,crawled_utc)","pair_unit":"(article_key,target_symbol)","public_text_policy":"no article text or evidence sentences published"};(public/"DATA_SCHEMA_AUDIT.md").write_text("# Full-corpus schema audit\n\n```json\n"+json.dumps(schema,indent=2,default=str)+"\n```\n")
    dump(public/"reaction_gate.json",{"status":"PASS" if gate_pass else "FAIL_STOP_BEFORE_PREDICTIVE_TRAINING","gate":gate,"period":"2018-01-01 through 2018-08-31","horizons_minutes":[30,60,120,240],"counts":dict(counts)})
    dump(public/"build_evidence.json",{"status":"COMPLETE","raw_rows":len(index),"candidate_pairs":len(cand),"canonical_all_time":len(canonical),"jan_aug_rows":len(window),"gate_pass":gate_pass,"private_pair_path":str(private/"canonical_pairs_all_time.pkl")})
    return gate_pass, gate


def main(public_path, private_path):
    public=Path(public_path);private=Path(private_path)
    if public.exists() and any(public.iterdir()): raise FileExistsError("reaction v1 public directory must be new")
    public.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=False)
    idx=pd.read_pickle(WORK/"audit"/"news_index.pkl");idx["article_key"]=idx.archive+"::"+idx.member
    cand,canonical,counts,schema_keys,sample_objs=build_pairs(idx,private);rx=bars_and_reactions(canonical,private);ok,gate=audit(public,private,idx,cand,canonical,rx,counts,schema_keys,sample_objs)
    print(json.dumps({"status":"COMPLETE","gate_pass":ok,"gate":gate,"candidate_pairs":len(cand),"canonical_all_time":len(canonical)},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--public",default=str(DEFAULT_PUBLIC));p.add_argument("--private",default=str(DEFAULT_PRIVATE));a=p.parse_args();main(a.public,a.private)
