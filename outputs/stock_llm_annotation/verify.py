"""Verify source spans, disjoint grouping, budgets and sealed packet inputs."""
import argparse
import collections
from pathlib import Path
from labels import load_panel, rows, sha, dump, PROMPT

def verify(data):
    data=Path(data);rs,m=load_panel(data)
    assert sha((data/'sources.jsonl').read_bytes())==m['sources_sha256']
    source={r['record_key']:r for r in rows(data/'sources.jsonl')}
    assert len({r['id'] for r in rs})==len(rs)
    groups=collections.defaultdict(set);spans=0
    for r in rs:
        groups[r['group']].add(r['split'])
        raw=source[r['record_key']]
        assert sha(raw['body'].encode())==r['body_sha256']
        assert sha(__import__('json').dumps(r['sentences'],sort_keys=True).encode())==r['input_sha256']
        assert r['published_utc']<=r['available_utc']
        assert r['available_utc']<m['extractor_earliest_freeze']
        assert (r['split']=='train' and r['available_utc']<'2018-03') or (r['split']=='valid' and '2018-03'<=r['available_utc']<'2018-04') or (r['split']=='check' and '2018-04'<=r['available_utc']<'2018-05')
        for k,p in r['spans'].items():
            assert raw[p['source']][p['start']:p['end']]==r['sentences'][k]
            spans+=1
    assert all(len(v)==1 for v in groups.values())
    assert len(groups)==len(rs), 'Selected panel repeats event group'
    counts=dict(collections.Counter(r['split'] for r in rs))
    if 'parent_inputs_sha256' not in m:
        assert counts=={'train':300,'valid':60,'check':100}
    else:
        assert counts['train']==300 and len(rs)==m['selected_pairs']
        assert {r['id'] for r in rs if r['id'].startswith('N')}=={f'N{i:04d}' for i in range(460)}
        assert all(r['split']==('valid' if r['id'].startswith('D') else 'check') for r in rs if not r['id'].startswith('N'))
    report={'verified_pairs':len(rs),'exact_spans':spans,'split_counts':counts,'unique_registered_groups':len(groups),'cross_split_group_overlap':0,'no_price_label_fields':all(not any(k in r for k in ['y','return','future_return','label']) for r in rs),'labels_completed':False,'heuristic_grouping_not_semantic_certification':True}
    dump(data/'verification.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);a=p.parse_args();print(verify(a.data))
