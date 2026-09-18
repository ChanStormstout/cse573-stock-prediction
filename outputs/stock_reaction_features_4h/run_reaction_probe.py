"""Frozen-FinBERT article-reaction probe after the full-corpus feasibility gate."""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stock_recency_dense_4h"))
from common import RAW, WORK, dump, schedule, sha

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "stock_reaction_features_4h" / "v1"
PRIVATE = WORK / "reaction_features_4h" / "v1"
MODEL_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
CS = (0.01, 0.1, 1.0)


def write_json(path, value):
    def clean(x):
        if hasattr(x, "item"): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [clean(v) for v in x]
        return x
    Path(path).write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def metric(y,p):
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);q=p>=.5
    return {"n":int(len(y)),"BA":float(balanced_accuracy_score(y,q)) if len(np.unique(y))==2 else None,"MCC":float(matthews_corrcoef(y,q)),"Brier":float(brier_score_loss(y,p)),"pred_up":float(q.mean())}


def sentences(text):
    import re
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text or "") if x.strip()]


def context(row):
    ss=sentences(str(row.body)); target="Apple" if row.symbol=="AAPL" else "Amazon"; hits=[i for i,s in enumerate(ss) if target.casefold() in s.casefold() or row.symbol in s]
    if hits:
        i=hits[0]; text=" ".join(ss[max(0,i-1):min(len(ss),i+2)])
    else: text=" ".join(ss[:2])
    return f"TARGET={row.symbol}\nTITLE={row.title}\nEVIDENCE={text}"[:6000]


def embed_texts(texts, private, source_keys=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    cache=private/"finbert_reaction_embeddings.npz"; manifest=private/"finbert_reaction_embedding_manifest.json"
    keys=list(texts.keys())
    fingerprint=hashlib.sha256("\n".join(f"{k}\t{texts[k]}" for k in keys).encode()).hexdigest()
    if cache.exists() and manifest.exists():
        m=json.loads(manifest.read_text())
        if m.get("keys_sha256")==fingerprint and m.get("revision")==MODEL_REVISION:
            z=np.load(cache); return keys,z["embeddings"].astype(float),m
    device="mps" if hasattr(torch.backends,"mps") and torch.backends.mps.is_available() else "cpu"
    try:
        tok=AutoTokenizer.from_pretrained("ProsusAI/finbert",revision=MODEL_REVISION,local_files_only=True)
        model=AutoModel.from_pretrained("ProsusAI/finbert",revision=MODEL_REVISION,local_files_only=True).to(device).eval()
    except OSError:
        # The repository already contains the frozen article vectors used by
        # the canonical F2 experiment.  The model binary itself is intentionally
        # private/absent on this host, so use those vectors as a coverage-limited
        # AR2 fallback rather than silently downloading or inventing features.
        z=np.load(ROOT/"stock_integrated_4h"/"prepared"/"articles.npz"); lookup={str(k):v.astype(float) for k,v in zip(z["keys"],z["embeddings"])}
        source_keys = source_keys or {}
        keep=[k for k in keys if source_keys.get(k,k) in lookup]; emb=np.vstack([lookup[source_keys.get(k,k)] for k in keep]).astype(np.float32)
        m={"revision":MODEL_REVISION,"keys_sha256":fingerprint,"rows":len(keep),"dim":int(emb.shape[1]),"device":"precomputed_private_article_vectors","status":"MODEL_BINARY_UNAVAILABLE_COVERAGE_LIMITED"}
        np.savez_compressed(cache,keys=np.asarray(keep),embeddings=emb);manifest.write_text(json.dumps(m,indent=2)+"\n");return keep,emb,m
    out=[]; batch=16
    for i in range(0,len(keys),batch):
        enc=tok([texts[k] for k in keys[i:i+batch]],padding=True,truncation=True,max_length=256,return_tensors="pt")
        enc={k:v.to(device) for k,v in enc.items()}
        with torch.no_grad(): h=model(**enc).last_hidden_state[:,0,:].detach().float().cpu().numpy()
        out.append(h)
    emb=np.vstack(out).astype(np.float32); np.savez_compressed(cache,keys=np.asarray(keys),embeddings=emb)
    m={"revision":MODEL_REVISION,"keys_sha256":fingerprint,"rows":len(keys),"dim":int(emb.shape[1]),"device":device,"max_length":256};manifest.write_text(json.dumps(m,indent=2)+"\n");return keys,emb,m


def price_context(rx):
    sessions=schedule();sessions["day"]=sessions.open.dt.strftime("%Y-%m-%d")
    maps={}
    for s,name in (("AAPL","APPLE"),("AMZN","AMAZON")):
        b=pd.read_csv(RAW/f"{name}5.csv",header=None,names=["date","time","open","high","low","close","activity"]);b.index=pd.to_datetime(b.date+" "+b.time,format="%Y.%m.%d %H:%M",utc=True);maps[s]=b.sort_index()
    rows=[]
    for r in rx.itertuples(index=False):
        t=pd.Timestamp(r.available_utc).tz_convert("UTC");b=maps[r.symbol];idx=b.index;pos=int(idx.searchsorted(t,side="left"));vals=[]
        for mins in (5,15,30,60):
            sl=idx[max(0,pos-mins//5):pos]
            valid=len(sl)==mins//5 and len(sl)>0 and np.all(np.diff(sl.view("i8"))==5*60*10**9)
            vals += [float(np.log(b.loc[sl[-1],"close"]/b.loc[sl[0],"open"])) if valid else 0.0,float(np.sqrt(np.square(np.log(b.loc[sl,"close"].to_numpy()/b.loc[sl,"open"].to_numpy())).sum())) if valid else 0.0]
        sess=sessions[(sessions.open<=t)&(sessions.close>t)];mins_open=float((t-sess.iloc[0].open).total_seconds()/60) if len(sess) else 0.0
        rows.append([r.article_key,r.symbol,r.group_key,*vals,mins_open])
    cols=["article_key","symbol","group_key"]+[f"pre_{v}_{n}" for n in (5,15,30,60) for v in ("return","rv")]+["minutes_from_open"]
    return pd.DataFrame(rows,columns=cols)


def fit_model(train,ev,method,C,emb_map):
    price_cols=[c for c in train.columns if c.startswith("pre_") or c=="minutes_from_open"]
    x0=train[price_cols].to_numpy(float);z0=ev[price_cols].to_numpy(float);x0=np.nan_to_num(x0);z0=np.nan_to_num(z0)
    if method=="AR0": x=x0;z=z0
    elif method=="AR1":
        vec=CountVectorizer(binary=True,min_df=3,max_features=500);a=vec.fit_transform(train.context);b=vec.transform(ev.context);x=np.hstack([x0,a.toarray()]);z=np.hstack([z0,b.toarray()])
    else:
        tr_emb=np.vstack([emb_map[k] for k in train.group_key]);ev_emb=np.vstack([emb_map[k] for k in ev.group_key]);pca=PCA(n_components=16,svd_solver="randomized",random_state=573).fit(tr_emb);x=np.hstack([x0,pca.transform(tr_emb)]);z=np.hstack([z0,pca.transform(ev_emb)])
    weights=1.0/train.groupby(["symbol","session_day"])["group_key"].transform("count").to_numpy(float)
    model=LogisticRegression(C=C,solver="liblinear",max_iter=3000,tol=1e-8,random_state=573).fit(x,train.label.to_numpy(int),sample_weight=weights)
    return model.predict_proba(z)[:,1]


def downstream_window_experiment(prom, reaction_predictions):
    """Use strict article OOF predictions as a small four-hour residual input."""
    selected = sorted({int(x["horizon_minutes"]) for x in prom if x["candidate"] == "AR1" and x["passes"]})
    if not selected:
        return {"status":"NOT_RUN_NO_PROMOTED_HORIZON"}
    official = pd.read_pickle(WORK/"paper_methods_4h"/"v1"/"inputs.pkl").copy(); official["key"] = official.symbol+"|"+pd.to_datetime(official.start_utc,utc=True).astype(str); official["start_utc"]=pd.to_datetime(official.start_utc,utc=True); official["cutoff_utc"]=pd.to_datetime(official.cutoff_utc,utc=True); official["phase"]=np.select([official.start_utc.dt.strftime("%Y-%m")<"2018-03",official.start_utc.dt.strftime("%Y-%m")<"2018-09"],["warmup","train_forward_oof"],default="exposed")
    base=pd.read_csv(ROOT/"stock_goal60_4h"/"v1"/"predictions.csv")[['key','F1']].rename(columns={"F1":"F1_base"}); official=official.merge(base,on="key",how="left",validate="one_to_one")
    rp=reaction_predictions.copy();rp["available_utc"]=pd.to_datetime(rp.available_utc,utc=True)
    for h in selected:
        q=rp[(rp.method=="AR1")&(rp.horizon_minutes==h)][["article_key","symbol","available_utc","AR1"]].copy();means=[];counts=[];stds=[]
        for r in official.itertuples(index=False):
            z=q[(q.symbol==r.symbol)&(q.available_utc<=r.cutoff_utc)&(q.available_utc>r.cutoff_utc-pd.Timedelta("4h"))]
            means.append(float(z.AR1.mean()) if len(z) else .5);counts.append(int(len(z)));stds.append(float(z.AR1.std(ddof=0)) if len(z) else 0.0)
        official[f"rx{h}_mean"]=means;official[f"rx{h}_n"]=counts;official[f"rx{h}_std"]=stds
    metrics=[];preds=[]
    def logitv(x): return np.log(np.clip(x,1e-6,1-1e-6)/(1-np.clip(x,1e-6,1-1e-6)))
    for symbol in ("AAPL","AMZN"):
        tr=official[(official.symbol==symbol)&(official.phase=="train_forward_oof")&(official.start_utc.dt.strftime("%Y-%m").isin(["2018-03","2018-04","2018-05"]))].copy(); ev=official[(official.symbol==symbol)&(official.start_utc.dt.strftime("%Y-%m").isin(["2018-06","2018-07","2018-08"]))].copy()
        if len(tr)<30 or ev.empty: continue
        base_tr=tr.F1_base.to_numpy(float);base_ev=ev.F1_base.to_numpy(float)
        columns={"W0":[],"W2":[]}
        # W0 is the saved canonical F1 control. W1 is one promoted reaction
        # horizon; W2 uses both horizons if two independently passed.
        for h in selected:
            X=np.c_[logitv(base_tr),tr[[f"rx{h}_mean",f"rx{h}_n"]].assign(**{f"rx{h}_n":np.log1p(tr[f"rx{h}_n"])}).to_numpy(float)]
            Z=np.c_[logitv(base_ev),ev[[f"rx{h}_mean",f"rx{h}_n"]].assign(**{f"rx{h}_n":np.log1p(ev[f"rx{h}_n"])}).to_numpy(float)]
            mod=LogisticRegression(C=.1,solver="liblinear",max_iter=3000,random_state=573).fit(X,tr.label.to_numpy(int));p=mod.predict_proba(Z)[:,1];name=f"W1_{h}m";ev[name]=p
        if len(selected)>=2:
            fs=[];zs=[]
            for h in selected: fs += [tr[f"rx{h}_mean"].to_numpy(float),np.log1p(tr[f"rx{h}_n"].to_numpy(float))];zs += [ev[f"rx{h}_mean"].to_numpy(float),np.log1p(ev[f"rx{h}_n"].to_numpy(float))]
            X=np.column_stack([logitv(base_tr),*fs]);Z=np.column_stack([logitv(base_ev),*zs]);mod=LogisticRegression(C=.1,solver="liblinear",max_iter=3000,random_state=573).fit(X,tr.label.to_numpy(int));ev["W2_multi"]=mod.predict_proba(Z)[:,1]
        h=selected[0];X=np.c_[logitv(base_tr),tr[f"rx{h}_mean"],np.log1p(tr[f"rx{h}_n"]),tr[f"rx{h}_std"]];Z=np.c_[logitv(base_ev),ev[f"rx{h}_mean"],np.log1p(ev[f"rx{h}_n"]),ev[f"rx{h}_std"]];mod=LogisticRegression(C=.1,solver="liblinear",max_iter=3000,random_state=573).fit(X,tr.label.to_numpy(int));adj=mod.predict_proba(Z)[:,1];ev["W3_gated"]=np.where(ev[f"rx{h}_n"].to_numpy(int)>=2,adj,base_ev)
        for name,p in [("W0",base_ev)]+[(f"W1_{h}m",ev[f"W1_{h}m"].to_numpy(float)) for h in selected]+([("W2_multi",ev.W2_multi.to_numpy(float))] if "W2_multi" in ev else [])+[("W3_gated",ev.W3_gated.to_numpy(float))]:
            metrics.append({"symbol":symbol,"method":name,**metric(ev.label,p)})
        q=ev[["key","symbol","day","label","F1_base"]].copy();q["W0"]=base_ev;preds.append(q.assign(**{n:ev[n].to_numpy(float) for n in [f"W1_{h}m" for h in selected]+(["W2_multi"] if "W2_multi" in ev else [])+["W3_gated"]}))
    out=pd.DataFrame(metrics);out.to_csv(PUBLIC/"reaction_downstream_metrics.csv",index=False);pd.concat(preds,ignore_index=True).to_csv(PUBLIC/"reaction_downstream_predictions.csv",index=False)
    return {"status":"COMPLETE","promoted_horizons":selected,"rows":int(len(out)),"metrics":metrics}


def main():
    t0=time.monotonic();gate=json.loads((PUBLIC/"reaction_gate.json").read_text())
    if gate.get("status")!="PASS": raise SystemExit("reaction feasibility gate failed; no predictive training")
    rx=pd.read_pickle(PRIVATE/"article_reactions_all_time.pkl");pairs=pd.read_pickle(PRIVATE/"canonical_pairs_all_time.pkl")
    rx["available_utc"]=pd.to_datetime(rx.available_utc,utc=True);rx["reaction_end_utc_60"]=pd.to_datetime(rx["reaction_60m_end_utc"],utc=True,errors="coerce");rx["reaction_end_utc_240"]=pd.to_datetime(rx["reaction_240m_end_utc"],utc=True,errors="coerce")
    rx=rx[(rx.available_utc>=pd.Timestamp("2018-01-01",tz="UTC"))&(rx.available_utc<pd.Timestamp("2018-09-01",tz="UTC"))].copy();pairs=pairs[["symbol","group_key","body","title"]]
    rx=rx.merge(pairs[["symbol","group_key","body"]],on=["symbol","group_key"],how="left",validate="one_to_one");rx["month"]=rx.available_utc.dt.strftime("%Y-%m");rx["context"]=rx.apply(context,axis=1)
    price_path=PRIVATE/"price_context.pkl"
    px=pd.read_pickle(price_path) if price_path.exists() else price_context(rx)
    if not price_path.exists(): px.to_pickle(price_path)
    rx=rx.merge(px,on=["article_key","symbol","group_key"],validate="one_to_one")
    # Run all four reaction horizons independently.  Only articles with a
    # complete same-session label for that horizon enter its folds.
    valid_any = rx[[f"reaction_{h}m_valid" for h in (30,60,120,240)]].sum(axis=1) > 0
    rx = rx[valid_any].copy(); rx_all = rx.copy()
    first = rx.drop_duplicates("group_key").set_index("group_key")
    all_keys = sorted(first.index.tolist()); texts = {k: context(first.loc[k]) for k in all_keys}
    source_keys = rx.drop_duplicates("group_key").set_index("group_key")["article_key"].to_dict()
    keys,emb,m=embed_texts(texts,PRIVATE,source_keys);embedding_manifest=m;emb_map={k:e for k,e in zip(keys,emb)}
    pred=[];selection=[];fits=[]
    embedding_coverage = float(rx.group_key.isin(set(emb_map)).mean()) if len(rx) else 0.0
    for h in (30,60,120,240):
        label=f"reaction_{h}m";valid=f"{label}_valid";endcol=f"reaction_{h}m_end_utc";rx["label"]= (rx[label]>0).astype(int);rx["reaction_end_utc"]=pd.to_datetime(rx[endcol],utc=True,errors="coerce");rx_all["label"]=(rx_all[label]>0).astype(int);rx_all["reaction_end_utc"]=pd.to_datetime(rx_all[endcol],utc=True,errors="coerce")
        for method in ("AR0","AR1","AR2"):
            method_rx = rx_all if method in ("AR0","AR1") else rx_all[rx_all.group_key.isin(set(emb_map))].copy()
            inner=[]
            for C in CS:
                for month in ("2018-03","2018-04","2018-05"):
                    for symbol in ("AAPL","AMZN"):
                        ev=method_rx[(method_rx.symbol==symbol)&(method_rx.month==month)&(method_rx[valid]==1)].copy();train=method_rx[(method_rx.symbol==symbol)&(method_rx.month<month)&(method_rx[valid]==1)&(method_rx.reaction_end_utc < ev.available_utc.min())].copy()
                        if len(train)<30 or ev.empty: continue
                        p=fit_model(train,ev,method,C,emb_map);inner.append({"horizon_minutes":h,"method":method,"C":C,"month":month,"symbol":symbol,**metric(ev.label,p)})
            tab=pd.DataFrame(inner);scores=[]
            for C in CS:
                g=tab[tab.C==C];scores.append((C,float(g.BA.mean()) if len(g) else -1))
            chosen=min(scores,key=lambda x:(-x[1],x[0]))[0] if scores else .1;selection.append({"horizon_minutes":h,"method":method,"C":chosen,"inner_scores":scores})
            for month,phase in [("2018-03","inner"),("2018-04","inner"),("2018-05","inner"),("2018-06","outer"),("2018-07","outer"),("2018-08","outer")]:
                for symbol in ("AAPL","AMZN"):
                    ev=method_rx[(method_rx.symbol==symbol)&(method_rx.month==month)&(method_rx[valid]==1)].copy();train=method_rx[(method_rx.symbol==symbol)&(method_rx.month<month)&(method_rx[valid]==1)&(method_rx.reaction_end_utc < ev.available_utc.min())].copy()
                    if len(train)<30 or ev.empty: continue
                    p=fit_model(train,ev,method,chosen,emb_map);q=ev[["article_key","symbol","group_key","available_utc","month","label",label,"session_day"]].copy();q["phase"]=phase;q["horizon_minutes"]=h;q["method"]=method;q[method]=p;pred.append(q)
    pred=pd.concat(pred,ignore_index=True);pred.to_csv(PUBLIC/"reaction_predictions.csv",index=False)
    rows=[]
    for (phase,h,s,m),g in pred.groupby(["phase","horizon_minutes","symbol","method"]): rows.append({"phase":phase,"horizon_minutes":h,"symbol":s,"method":m,**metric(g.label,g[m])})
    metrics=pd.DataFrame(rows);metrics.to_csv(PUBLIC/"reaction_metrics.csv",index=False);pd.DataFrame(selection).to_json(PUBLIC/"reaction_selection.json",orient="records",indent=2)
    outer=metrics[metrics.phase=="outer"];prom=[]
    for h in (30,60,120,240):
        b=outer[(outer.horizon_minutes==h)&(outer.method=="AR0")].set_index("symbol");
        for method in ("AR1","AR2"):
            n=outer[(outer.horizon_minutes==h)&(outer.method==method)].set_index("symbol");common=b.index.intersection(n.index);d=(n.loc[common,"BA"]-b.loc[common,"BA"]);prom.append({"horizon_minutes":h,"candidate":method,"AAPL_delta_BA":float(d.get("AAPL",np.nan)),"AMZN_delta_BA":float(d.get("AMZN",np.nan)),"macro_delta_BA":float(d.mean()) if len(d) else None,"passes":bool(len(d)==2 and d.min()>=.01 and (n.loc[common,"Brier"]-b.loc[common,"Brier"]).max()<=.002)})
    downstream=downstream_window_experiment(prom,pred);gate_out={"status":"COMPLETE","promotion":prom,"any_horizon_passes":bool(any(x["passes"] for x in prom)),"downstream_W0_W3":"COMPLETE" if downstream.get("status")=="COMPLETE" else "NOT_RUN_NO_HORIZON_PASSED","downstream":downstream,"runtime_seconds":time.monotonic()-t0,"embedding":embedding_manifest,"embedding_coverage_for_ar2":embedding_coverage,"ar2_scope":"coverage_limited_precomputed_frozen_vectors" if embedding_manifest.get("status") else "full_frozen_context","protocol_sha256":sha(PUBLIC.parent/"PRE_REGISTRATION.md")};write_json(PUBLIC/"reaction_probe_gate.json",gate_out);write_json(PUBLIC/"reaction_training_evidence.json",{"status":"COMPLETE","selection":selection,"prediction_rows":len(pred),"fits":len(selection)*6*2,"runtime_seconds":time.monotonic()-t0,"model_revision":MODEL_REVISION,"embedding_coverage_for_ar2":embedding_coverage});print(json.dumps(gate_out,indent=2))


if __name__=="__main__": main()
