"""Split a visible multi-packet reply into exact-ID batches, preserving source hash."""
import argparse
import json
from pathlib import Path
from labels import parse_reply, import_reply, write_rows, dump, sha

def main():
    p=argparse.ArgumentParser()
    for key in ['data','packets','response','batches','pass-name','url','out']:p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.out)
    text=Path(a.response).read_text();rs=parse_reply(text);ids=[r.get('id') for r in rs]
    manifest=json.loads((Path(a.packets)/'manifest.json').read_text())
    entries=[r for r in manifest['batches'] if r['batch'] in a.batches.split(',')]
    if {r['batch'] for r in entries}!=set(a.batches.split(',')):raise ValueError('Unknown batch')
    expected={i for r in entries for i in r['ids']}
    if len(ids)!=len(set(ids)) or set(ids)!=expected:
        raise ValueError({'missing':sorted(expected-set(ids)),'foreign':sorted(set(ids)-expected),'duplicate_ids':len(ids)-len(set(ids))})
    out.mkdir(parents=True,exist_ok=False)
    (out/'original_response.txt').write_text(text)
    for r in entries:
        raw=out/(r['batch']+'.txt');write_rows(raw,[x for x in rs if x['id'] in r['ids']])
        import_reply(a.data,a.packets,r['batch'],a.pass_name,raw,a.url,'GPT 6 High (browser UI)',out/r['batch'])
    dump(out/'aggregate_receipt.json',{'original_response_sha256':sha(text.encode()),'conversation_url':a.url,'n':len(rs),'batches':[r['batch'] for r in entries],'derived_chunk_files':'Identity-preserving JSON serialization, no fact edits'})

if __name__=='__main__':main()
