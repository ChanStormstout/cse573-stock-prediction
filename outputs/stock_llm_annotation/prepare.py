"""Immutable, content-stratified annotation dataset. Never reads stock outcomes."""
import argparse
import collections
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'outputs/stock_llm_4h'))
import importlib.util
spec = importlib.util.spec_from_file_location('pilot_prepare', ROOT / 'outputs/stock_llm_4h/prepare.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

HERE = Path(__file__).resolve().parent
ALIAS = {s: re.compile(p, re.I) for s, p in pilot.ALIAS.items()}
ACTION = re.compile(r'price.target|target.price|upgrad|downgrad|reiterat|reaffirm|\brating\b|coverage|\braised\b|\blowered\b', re.I)
HISTORICAL = re.compile(r'other (?:broker|analyst)|among \d+ analysts|analysts covering', re.I)

def sha(x):
    return hashlib.sha256(x).hexdigest()

def dump(path, x):
    path.write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def write_rows(path, rs):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rs))

def period(t):
    return 'train' if t < '2018-03' else 'valid' if t < '2018-04' else 'check'

def tokens(title):
    return set(re.findall(r'[a-z]+|\d+(?:\.\d+)?', title.lower()))

def similar(a, b):
    # Different amounts can encode a new event; do not merge those near titles.
    numbers = lambda ts: {t for t in ts if t[0].isdigit()}
    return numbers(a) == numbers(b) and len(a & b)/max(1, len(a | b)) >= .72

def select(symbol, title, body):
    """Current lead + actionable target context, preserving exact raw offsets."""
    spans = pilot.split_sentences(body)
    valid = [i for i, (_, _, text) in enumerate(spans) if 25 <= len(text) <= 1500 and not pilot.BAD.search(text)]
    ranked = sorted((i for i in valid if ALIAS[symbol].search(spans[i][2])),
                    key=lambda i: (-int(bool(ACTION.search(spans[i][2]))), int(bool(HISTORICAL.search(spans[i][2]))), i))
    priority = list(valid[:2])
    for i in ranked[:4]:
        priority.extend(j for j in (i, i-1, i+1) if j in valid)
    ids, size = set(), len(title)
    for i in priority:
        if i not in ids and size+len(spans[i][2]) <= 4300:
            ids.add(i)
            size += len(spans[i][2])
    sentences = {'S0': title}
    mapping = {'S0': {'source': 'title', 'start': 0, 'end': len(title)}}
    for i in sorted(ids):
        a, b, text = spans[i]
        k = f'S{len(sentences)}'
        sentences[k] = text
        mapping[k] = {'source': 'body', 'start': a, 'end': b}
    return sentences, mapping

def balanced_select(candidates, n):
    # Two-thirds actionable candidates where available; flag != confirmed positive.
    chosen, used = [], set()
    quota = {'title_action': int(n*.4), 'body_action': int(n*.3), 'control': n-int(n*.4)-int(n*.3)}
    pools = {}
    for stratum in quota:
        pools[stratum] = {}
        for s in ALIAS:
            pools[stratum][s] = sorted((r for r in candidates if r['stratum']==stratum and r['symbol']==s), key=lambda r: sha((r['record_key']+s).encode()))
    for stratum, count in quota.items():
        queues = pools[stratum]
        for _ in range(count):
            options = []
            for s in ALIAS:
                while queues[s] and queues[s][0]['group'] in used:
                    queues[s].pop(0)
                if queues[s]:
                    options.append((sum(r['symbol']==s for r in chosen), s))
            if not options:
                break
            s = min(options)[1]
            r = queues[s].pop(0)
            chosen.append(r); used.add(r['group'])
    # Fill genuine availability shortfalls, never duplicate/fabricate positives.
    rest = sorted(candidates, key=lambda r: sha((r['record_key']+r['symbol']).encode()))
    for r in rest:
        if len(chosen) >= n:
            break
        if r['group'] not in used:
            chosen.append(r); used.add(r['group'])
    return chosen

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    index = ROOT / 'work/stock-data/audit/news_index.pkl'
    raw = pd.read_pickle(index)
    raw = raw[raw.lag_hours.ge(0) & raw.language.eq('english') & raw.available_utc.ge('2018-01-01') & raw.available_utc.lt('2018-05-01')].sort_values(['available_utc','archive','member'])
    old = [json.loads(x) for x in (ROOT/'outputs/stock_llm_4h/data/pilot_v3/inputs.jsonl').read_text().splitlines()]
    exposed_keys = {r['record_key'] for r in old}
    exposed_hashes = {r['body_sha256'] for r in old}
    exposed_titles = [tokens(r['title']) for r in old]
    # Use existing registered event groups as one additional conservative key.
    known = {}
    for line in (ROOT/'outputs/stock_adaptive_4h/coverage/v3/article_candidates.jsonl').read_text().splitlines():
        r = json.loads(line)
        known.setdefault(r['record_key'], set()).add(str(r['event_group']))
    anchors, exact, urls, event_keys, candidates, sources = [], {}, {}, {}, [], []
    skip = collections.Counter()
    archives = {}
    try:
        for j, row in enumerate(raw.to_dict('records')):
            archive, member = row['archive'], row['member']
            key = archive+'::'+member
            if archive not in archives:
                archives[archive] = zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/archive)
            body = json.loads(archives[archive].read(member)).get('text', '')
            title = row['title'] or ''
            syms = [s for s in ALIAS if ALIAS[s].search(title+'\n'+body)]
            if not syms or len(body)<50:
                skip['no_target_or_short'] += 1; continue
            tt = tokens(title)
            if key in exposed_keys or sha(body.encode()) in exposed_hashes or any(similar(tt, t) for t in exposed_titles):
                skip['old_pilot_or_similar_title'] += 1; continue
            t = row['available_utc'].isoformat()
            split = period(t)
            norm = str(row['normalized_hash'])
            group = exact.get(norm) or urls.get(row['url'])
            if group is None:
                group = next((event_keys[k] for k in sorted(known.get(key, [])) if k in event_keys), None)
            if group is None:
                group = next((a['group'] for a in reversed(anchors) if (row['available_utc']-a['time']).days <= 30 and similar(tt,a['tokens'])), None)
            if group is not None:
                skip['duplicate_group_'+split] += 1
                exact[norm] = group
                if row['url']: urls[row['url']] = group
                for k in known.get(key, []): event_keys[k] = group
                continue
            group = sha(key.encode())[:20]
            exact[norm] = group
            if row['url']: urls[row['url']] = group
            for k in known.get(key, []): event_keys[k] = group
            anchors.append({'group':group,'tokens':tt,'time':row['available_utc']})
            sources.append({'record_key':key,'body':body,'title':title})
            for s in syms:
                sentences, spans = select(s,title,body)
                nearby = any(ALIAS[s].search(v) and ACTION.search(v) for k,v in sentences.items() if k!='S0')
                stratum = 'title_action' if ALIAS[s].search(title) and ACTION.search(title) else 'body_action' if nearby else 'control'
                candidates.append({'symbol':s,'record_key':key,'group':group,'split':split,'stratum':stratum,'available_utc':t,'published_utc':row['published_utc'].isoformat(),'title':title,'sentences':sentences,'spans':spans,'body_sha256':sha(body.encode())})
            if j%2000==0: print('SCANNED',j,'candidates',len(candidates),flush=True)
    finally:
        for z in archives.values(): z.close()
    selected = []
    for split,n in [('train',300),('valid',60),('check',100)]:
        selected.extend(balanced_select([r for r in candidates if r['split']==split],n))
    selected.sort(key=lambda r:(r['available_utc'],r['record_key'],r['symbol']))
    for i,r in enumerate(selected):
        r['id'] = f'N{i:04d}'
        r['input_sha256'] = sha(json.dumps(r['sentences'],sort_keys=True).encode())
    keys = {r['record_key'] for r in selected}
    write_rows(args.out/'inputs.jsonl',selected)
    write_rows(args.out/'sources.jsonl',[r for r in sources if r['record_key'] in keys])
    write_rows(args.out/'candidate_ledger.jsonl',candidates)
    summary = {'status':'PREPARED_UNLABELLED','eligible_raw_records':len(raw),'candidate_pairs':len(candidates),'selected_pairs':len(selected),'counts':dict(collections.Counter(r['split']+'|'+r['symbol']+'|'+r['stratum'] for r in selected)),'candidate_counts':dict(collections.Counter(r['split']+'|'+r['symbol']+'|'+r['stratum'] for r in candidates)),'excluded':dict(skip),'source_sha256':sha(index.read_bytes()),'inputs_sha256':sha((args.out/'inputs.jsonl').read_bytes()),'sources_sha256':sha((args.out/'sources.jsonl').read_bytes()),'protocol_sha256':sha((HERE/'PROTOCOL.md').read_bytes()),'prepare_sha256':sha(Path(__file__).read_bytes()),'extractor_earliest_freeze':'2018-05-01T00:00:00+00:00','price_labels_read':False,'independent_review_passed':False,'warning':'Heuristic candidate strata are not event labels. Grouping is conservative and not proof of semantic independence; historical exposure outside registered pilot may remain.'}
    dump(args.out/'manifest.json',summary)
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
