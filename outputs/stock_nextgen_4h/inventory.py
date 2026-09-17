"""Audit locally available inputs and record external minute-data feasibility."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import B, DIRECT, INTEGRATED, ROOT, W, dump, sha


def chart_inventory(path: Path, symbol: str, interval: int) -> dict:
    frame = pd.read_csv(
        path,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "activity"],
    )
    stamp = pd.to_datetime(frame.date + " " + frame.time, format="%Y.%m.%d %H:%M", utc=True)
    activity = frame.activity
    return {
        "symbol": symbol,
        "interval_minutes": interval,
        "path": str(path),
        "sha256": sha(path),
        "rows": int(len(frame)),
        "first_timestamp": stamp.min().isoformat(),
        "last_timestamp": stamp.max().isoformat(),
        "duplicate_timestamps": int(stamp.duplicated().sum()),
        "ohlc_missing": int(frame[["open", "high", "low", "close"]].isna().sum().sum()),
        "seventh_field_name_in_project": "activity",
        "activity_min": float(activity.min()),
        "activity_max": float(activity.max()),
        "activity_integer_fraction": float((activity % 1 == 0).mean()),
        "activity_zero_fraction": float((activity == 0).mean()),
        "can_call_volume": False,
        "reason": "Course files and project guidance do not define the seventh column.",
    }


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    charts = []
    for symbol, prefix in (("AAPL", "APPLE"), ("AMZN", "AMAZON")):
        for interval in (5, 15, 30, 60, 240, 1440):
            charts.append(chart_inventory(W / f"raw/CHARTS/{prefix}{interval}.csv", symbol, interval))

    index_path = W / "audit/news_index.pkl"
    news = pd.read_pickle(index_path)
    archive_paths = sorted((W / "raw/news").glob("*.zip"))
    article_key = news.archive.astype(str) + "::" + news.member.astype(str)
    available = pd.to_datetime(news.available_utc, utc=True)
    published = pd.to_datetime(news.published_utc, utc=True)
    crawled = pd.to_datetime(news.crawled_utc, utc=True)
    target_counts = {
        "records": int(len(news)),
        "unique_record_keys": int(article_key.nunique()),
        "exact_hash_duplicates": int(len(news) - news.exact_hash.nunique()),
        "normalized_hash_duplicates": int(len(news) - news.normalized_hash.nunique()),
        "AAPL_flag": int(news.aapl.sum()),
        "AMZN_flag": int(news.amzn.sum()),
        "both_flags": int((news.aapl & news.amzn).sum()),
        "unknown_published": int(published.isna().sum()),
        "available_before_published": int((available < published).fillna(False).sum()),
        "crawled_before_published": int((crawled < published).fillna(False).sum()),
        "available_time_definition": "max(published_utc, crawled_utc) when both are present; conservative receipt time",
        "median_collection_delay_hours": float(news.lag_hours.dropna().median()),
        "sources": int(news.site.nunique()),
        "archives": len(archive_paths),
        "archive_first": archive_paths[0].name,
        "archive_last": archive_paths[-1].name,
        "body_storage": "raw JSON bodies inside local ZIP archives",
    }

    data = pd.read_pickle(INTEGRATED / "prepared/data.pkl")
    data["start_utc"] = pd.to_datetime(data.start_utc, utc=True)
    split_counts = {
        f"{symbol}|{split}": int(n)
        for (symbol, split), n in data.groupby(["symbol", "split"]).size().items()
    }
    local_names = {path.name.lower() for path in (W / "raw/CHARTS").glob("*.csv")}
    factors = {
        name: {
            "local_minute_file": any(name.lower() in filename for filename in local_names),
            "status": "missing",
        }
        for name in ("SPY", "QQQ", "XLK", "XLY")
    }
    calendar_path = W / "audit/xnys_schedule.csv"
    calendar = pd.read_csv(calendar_path)
    # Search only the raw data area. Environment/package filenames such as
    # sklearn's *_learning.py are not course event datasets.
    raw_root = W / "raw"
    macro_candidates = [
        path
        for path in raw_root.rglob("*")
        if path.is_file()
        and any(term in path.name.lower() for term in ("fomc", "cpi", "payroll", "jobs", "consensus", "surprise"))
    ]
    earnings_candidates = [
        path for path in raw_root.rglob("*") if path.is_file() and any(term in path.name.lower() for term in ("earning", "consensus", "estimate", "actual"))
    ]

    external = {
        "Alpha Vantage": {
            "official_url": "https://www.alphavantage.co/documentation/",
            "finding": "Historical intraday month access is documented as premium.",
            "qualifies_free_reproducible_2018_minute": False,
        },
        "Polygon": {
            "official_url": "https://polygon.io/docs/flat-files/stocks/minute-aggregates/2009/03",
            "finding": "Long minute history exists, but the relevant access is tied to paid plans.",
            "qualifies_free_reproducible_2018_minute": False,
        },
        "Twelve Data": {
            "official_url": "https://support.twelvedata.com/en/articles/5214728-getting-historical-data",
            "finding": "Intraday history is documented, but no adequate free 2018 entitlement and redistribution contract was established for this run.",
            "qualifies_free_reproducible_2018_minute": False,
        },
    }
    inventory = {
        "status": "COMPLETE",
        "task": "AAPL/AMZN four-hour direction",
        "rows": len(data),
        "split_counts": split_counts,
        "charts": charts,
        "news": target_counts,
        "calendar": {
            "path": str(calendar_path),
            "sha256": sha(calendar_path),
            "rows": len(calendar),
            "status": "available",
            "role": "exchange session boundaries only",
        },
        "market_factors": factors,
        "macro_event_calendar": {
            "status": "missing",
            "files": [str(path) for path in macro_candidates],
        },
        "earnings_consensus_actuals": {
            "status": "missing",
            "files": [str(path) for path in earnings_candidates],
        },
        "external_feasibility": external,
        "data_roles": {
            "supervision": ["four-hour UP/DOWN label recomputed from 48 target five-minute OHLC bars"],
            "ordinary_inputs": ["cutoff-safe AAPL/AMZN OHLC", "news title/body/source/published/crawled/available timestamps", "XNYS session boundaries"],
            "external_metadata": ["XNYS calendar", "official provider documentation used only for feasibility"],
            "provisional_LLM_annotations_used": [],
            "missing": ["SPY/QQQ/XLK/XLY cutoff-aligned minute bars", "macro release calendar and consensus", "earnings consensus and actuals", "independently reviewed event labels"],
        },
        "decision": "Run M0 (R1) only; stop M1-M3 and downstream PCA/HMM/GNN.",
        "source_hashes": {
            str(index_path): sha(index_path),
            str(INTEGRATED / "prepared/data.pkl"): sha(INTEGRATED / "prepared/data.pkl"),
            str(DIRECT / "PROTOCOL.md"): sha(DIRECT / "PROTOCOL.md"),
            str(B / "PROTOCOL.md"): sha(B / "PROTOCOL.md"),
            str(Path(__file__)): sha(Path(__file__)),
        },
    }
    dump(out / "inventory.json", inventory)

    five = [row for row in charts if row["interval_minutes"] == 5]
    lines = [
        "# 数据盘点与外部数据可行性",
        "",
        "本文件在本轮模型评分前生成。结论是：AAPL／AMZN 五分钟 OHLC、交易日历和新闻正文可用；市场 ETF、宏观公布表、盈利预期／实际值在本地缺失。",
        "",
        "## 已有输入",
        "",
        "| 项目 | 状态 | 可用于什么 | 限制 |",
        "|---|---|---|---|",
        f"| AAPL 5分钟 OHLC | 可用，{five[0]['rows']:,} 行 | R0/R1/R2、标签复核 | 第七列语义未知 |",
        f"| AMZN 5分钟 OHLC | 可用，{five[1]['rows']:,} 行 | R0/R1/R2、标签复核 | 第七列语义未知 |",
        f"| XNYS 日历 | 可用，{len(calendar):,} 日 | 开收盘和隔夜边界 | 不是宏观事件表 |",
        f"| 新闻索引和正文 | 可用，{len(news):,} 条 | P0--P3 | 语料偏向 AAPL，时间字段存在已知异常 |",
        "| SPY/QQQ/XLK/XLY 分钟线 | 缺失 | 无 | M1--M3 停止 |",
        "| FOMC/CPI/就业公布及 consensus | 缺失 | 无 | 不能构造 surprise |",
        "| 盈利 consensus/actuals | 缺失 | 无 | 不能构造 earnings surprise |",
        "",
        "## 第七列是否为成交量",
        "",
        "课程文件没有定义第七列。它是整数且为正，只能记录为 `activity`；这些现象不能证明它是成交量。因此本轮不构造 volume、VWAP、成交量冲击或订单流特征。",
        "",
        "## 新闻时间、来源与重复",
        "",
        f"新闻来自 {news.site.nunique():,} 个站点；精确哈希重复 {target_counts['exact_hash_duplicates']:,} 条，规范化正文哈希重复 {target_counts['normalized_hash_duplicates']:,} 条。AAPL 标记 {target_counts['AAPL_flag']:,} 条，AMZN 标记 {target_counts['AMZN_flag']:,} 条，后者全部同时带有 AAPL 标记，说明语料覆盖不对称。",
        "",
        f"原始字段中有 {target_counts['crawled_before_published']:,} 条记录的抓取时间早于标注发布时间。流水线的 `available_utc` 明确定义为可用发布时间和抓取时间中的较晚者，所以派生后的 `available_utc < published_utc` 为 {target_counts['available_before_published']:,} 条；这只是保守截止规则，不证明源时间元数据无异常。",
        "",
        "## 数据角色",
        "",
        "| 角色 | 本轮内容 |",
        "|---|---|",
        "| 监督标签 | 由目标区间48根五分钟线重新计算的四小时UP/DOWN |",
        "| 普通输入 | 截止前AAPL/AMZN OHLC、新闻标题/正文/来源/时间、交易日边界 |",
        "| 外部元数据 | XNYS日历；外部供应商文档只用于可行性判断 |",
        "| 暂定LLM标注 | 本轮预测没有使用；旧事件标注不进入P0--P3 |",
        "| 完全缺失 | 市场/行业分钟线、宏观公布与consensus、盈利consensus/actuals、独立复核事件标签 |",
        "",
        "## 外部分钟数据门槛",
        "",
        "官方资料显示 Alpha Vantage 的历史月度 intraday 属于 premium；Polygon 的长期分钟聚合需要相应付费访问；Twelve Data 虽有 intraday 接口，但本轮没有确认足够的免费 2018 历史权限和可复现／再分发条件。没有下载或混入来源不明的数据。",
        "",
        "因此只执行 M0=R1。M1--M3、market PCA、HMM 和 graph 分支按预注册停止，不把日频 ETF 数据混入四小时任务。",
        "",
        "## 任务数据",
        "",
        f"固定保留 {len(data):,} 个四小时窗口：" + "，".join(f"{k}={v}" for k, v in split_counts.items()) + "。",
        "",
        "所有 September 2018 之后的标签均已在历史探索中暴露，本轮仍称探索性历史回放。",
    ]
    (out / "DATA_INVENTORY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "COMPLETE", "decision": inventory["decision"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.out)
