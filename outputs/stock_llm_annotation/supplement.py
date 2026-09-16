"""Immutable March-only content supplement; never consumes April check labels."""
import argparse, json, re, shutil, zipfile
from pathlib import Path
from labels import rows, sha, dump, write_rows, PROMPT, HERE, ROOT, load_panel

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--split',choices=['valid','check'],default='valid');p.add_argument('--ledger',type=Path);a=p.parse_args()
 original,m=load_panel(a.data);used={r['group'] for r in original};extra=[]
 for r in rows(a.ledger or a.data/'candidate_ledger.jsonl'):
  if r['split']!=a.split or r['group'] in used:continue
  t=r['title'];alias=r['symbol'] in t or ('Apple' in t if r['symbol']=='AAPL' else 'Amazon' in t)
  if alias and re.search(r'price.target|target.price|upgrad|downgrad|reiterat|reaffirm|rating|\bPT\b',t,re.I) and not re.search(r'Holder|Holding|Stake|Position|Sentiment|Earning.*Coverage',t,re.I):
   r=dict(r);r['id']=('D' if a.split=='valid' else 'Q')+f'{len(extra):04d}';r['input_sha256']=sha(json.dumps(r['sentences'],sort_keys=True).encode());extra.append(r);used.add(r['group'])
 a.out.mkdir(parents=True,exist_ok=False);write_rows(a.out/'inputs.jsonl',original+extra)
 sources=rows(a.data/'sources.jsonl');existing={r['record_key'] for r in sources}
 for r in extra:
  if r['record_key'] in existing:continue
  archive,member=r['record_key'].split('::',1)
  with zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/archive) as z:body=json.loads(z.read(member)).get('text','')
  sources.append({'record_key':r['record_key'],'body':body,'title':r['title']});existing.add(r['record_key'])
 write_rows(a.out/'sources.jsonl',sources)
 m.update(status='PREPARED_WITH_'+a.split.upper()+'_SUPPLEMENT',parent_inputs_sha256=m['inputs_sha256'],inputs_sha256=sha((a.out/'inputs.jsonl').read_bytes()),sources_sha256=sha((a.out/'sources.jsonl').read_bytes()),selected_pairs=len(original)+len(extra),supplement_n=len(extra),amendment_sha256=sha((HERE/'DEVELOPMENT_AMENDMENT.md').read_bytes()),supplement_script_sha256=sha(Path(__file__).read_bytes()))
 dump(a.out/'manifest.json',m);shutil.copytree(a.data/'comparisons',a.out/'comparisons');shutil.copy2(a.data/'decisions.jsonl',a.out/'decisions.jsonl')
 directory=a.out/'packets';directory.mkdir();entries=[]
 for j in range(0,len(extra),20):
  rs=extra[j:j+20];name=('supplement_' if a.split=='valid' else 'check_supplement_')+f'{j//20:02d}';file=directory/(name+'.txt');file.write_text(PROMPT+'\nITEMS\n'+''.join(json.dumps({k:r[k] for k in ['id','symbol','sentences']},ensure_ascii=False)+'\n' for r in rs));entries.append({'batch':name,'ids':[r['id'] for r in rs],'file':file.name,'sha256':sha(file.read_bytes()),'split':a.split})
 dump(directory/'manifest.json',{'inputs_sha256':m['inputs_sha256'],'prompt_sha256':sha(PROMPT.encode()),'batches':entries});print(json.dumps({'total':len(original)+len(extra),'supplement':len(extra)}))
if __name__=='__main__':main()
