"""Read-only raw-news coverage investigation; no new records enter forecasting."""
from core import *
from evidence import COMP,role,extract,quality_gate
import zipfile,re,argparse,shutil
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def main(out):
    out.mkdir(parents=True,exist_ok=False)
    (out/'code_snapshot').mkdir()
    for source in ['audit.py','evidence.py','core.py']:shutil.copy2(B/source,out/'code_snapshot'/source)
    rawpath=ROOT/'work/stock-data/audit/news_index.pkl';raw=pd.read_pickle(rawpath);raw['key']=raw.archive+'::'+raw.member
    d=pd.read_pickle(OLD/'prepared/data.pkl');d['key']=d.symbol+'|'+d.start_utc.astype(str)
    hashes={str(rawpath):sha(rawpath),str(B/'audit.py'):sha(B/'audit.py'),str(B/'evidence.py'):sha(B/'evidence.py'),str(OLD/'prepared/data.pkl'):sha(OLD/'prepared/data.pkl')}
    valid=raw[raw.language.eq('english')&raw.chars.ge(100)&raw.lag_hours.ge(0)&~raw.physician].copy().sort_values(['available_utc','archive','member'])
    times=valid.available_utc.astype('int64').to_numpy();indices=set()
    for cut in d.cutoff_utc.unique():
        lo=np.searchsorted(times,(cut-pd.Timedelta('4h')).value,side='right');hi=np.searchsorted(times,cut.value,side='right');indices.update(range(lo,hi))
    eligible=valid.iloc[sorted(indices)].copy();needed=set(eligible.key);bodies={}
    # These cached raw strings are verified against the archive when no existing content hash exists.
    for archive,g in eligible.groupby('archive'):
        path=ROOT/'work/stock-data/raw/news'/archive;hashes[str(path)]=sha(path)
        with zipfile.ZipFile(path) as z:
            for r in g.itertuples():bodies[r.key]=json.loads(z.read(r.member)).get('text','')
        print('AUDIT read',archive,len(g),flush=True)
    rows=[];events=[];group_history={'AAPL':[],'AMZN':[]};first_exact={}
    for r in eligible.itertuples():
        body=bodies[r.key];title=r.title if pd.notna(r.title) else ''
        for sym in ['AAPL','AMZN']:
            category,spans,marketing=role(sym,title,body)
            if category=='no_target':continue
            toks=set(re.findall('[a-z0-9]+',title.lower()))-set(ENGLISH_STOP_WORDS)
            prev=group_history[sym];gid=None
            for item in reversed(prev[-200:]):
                if r.available_utc-item['time']>pd.Timedelta('7d'):break
                union=toks|item['tokens']
                if union and len(toks&item['tokens'])/len(union)>=.75:gid=item['group'];break
            exact=(sym,r.normalized_hash)
            gid=first_exact.get(exact,gid) or hashlib.sha256((sym+'|'+r.key).encode()).hexdigest()[:16]
            first_exact[exact]=gid;prev.append({'time':r.available_utc,'tokens':toks,'group':gid})
            extracted=extract(sym,r.key,title,body,str(r.published_utc),str(r.available_utc))
            for ev in extracted:ev['event_group']=gid;events.append(ev)
            rows.append({'symbol':sym,'record_key':r.key,'available_utc':str(r.available_utc),'published_utc':str(r.published_utc),'month':r.available_utc.strftime('%Y-%m'),'title':title,'title_hit':bool(re.search(COMP[sym],title,re.I)),'role_rule':category,'marketing':marketing,'event_group':gid,'body_sha256':hashlib.sha256(body.encode()).hexdigest(),'evidence_spans':spans,'event_count':len(extracted),'quality_status':'PROVISIONAL_RULE_ONLY'})
    pairs=pd.DataFrame(rows);pairs.to_json(out/'article_candidates.jsonl',orient='records',lines=True,force_ascii=False)
    (out/'events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n' for e in events))
    look=eligible.set_index('key');window_rows=[];usage=[]
    for sym,g in d.groupby('symbol'):
        p=pairs[pairs.symbol==sym].copy();p['available_utc']=pd.to_datetime(p.available_utc,utc=True,format='mixed')
        # Causal first occurrence under the expanded candidate policy; preserve baseline IDs separately.
        p['normalized_hash']=p.record_key.map(look.normalized_hash);p['title_norm']=p.record_key.map(look.title_norm)
        p=p.sort_values(['available_utc','record_key']).drop_duplicates('normalized_hash').drop_duplicates('title_norm')
        for row in g.itertuples():
            window=p[(p.available_utc>row.cutoff_utc-pd.Timedelta('4h'))&(p.available_utc<=row.cutoff_utc)]
            old={k for k in row.news_record_keys.split('|') if k};novel=window[~window.record_key.isin(old)]
            direct=novel[(novel.role_rule=='direct_rule_candidate')&~novel.marketing&~novel.title_hit]
            rec={'key':row.key,'symbol':sym,'split':row.split,'month':row.start_utc.strftime('%Y-%m'),'old_count':len(old),'all_body_candidates':len(window),'new_non_title_direct_rule':len(direct),'new_comparison_rule':int((~novel.title_hit&novel.role_rule.eq('comparison_candidate')).sum()),'new_mention_or_promotion':int((~novel.title_hit&novel.role_rule.eq('mention_or_promotion')).sum()),'potential_union_count':len(old|set(direct.record_key))}
            window_rows.append(rec)
            usage.extend({'window_key':row.key,'symbol':sym,'split':row.split,'record_key':k,'type':'new_non_title_direct_rule'} for k in direct.record_key)
    windows=pd.DataFrame(window_rows);windows.to_csv(out/'window_coverage.csv',index=False);pd.DataFrame(usage).to_csv(out/'candidate_usage.csv',index=False)
    totals=[]
    for (sym,split),g in windows.groupby(['symbol','split']):
        used=[x['record_key'] for x in usage if x['symbol']==sym and x['split']==split]
        totals.append({'symbol':sym,'split':split,'windows':len(g),'original_news_windows':int((g.old_count>0).sum()),'potential_news_windows_rule_only':int((g.potential_union_count>0).sum()),'new_candidate_articles':len(set(used)),'newly_covered_windows_rule_only':int(((g.old_count==0)&(g.potential_union_count>0)).sum())})
    pd.DataFrame(totals).to_csv(out/'coverage_summary.csv',index=False)
    # Review panels use only Jan--Aug text and no outcome labels. Event group disjoint.
    pairmap={(x['symbol'],x['record_key']):x for x in rows};review=[]
    event_df=pd.DataFrame(events)
    train_events=event_df[pd.to_datetime(event_df.available_utc,utc=True,format='mixed')<pd.Timestamp('2018-09-01',tz='UTC')]
    chosen=train_events.sort_values('event_id').drop_duplicates(['symbol','kind','event_group'])
    for r in chosen.astype(object).where(pd.notna(chosen),None).to_dict('records'):
        split='check' if int(hashlib.sha256(r['event_group'].encode()).hexdigest()[:8],16)%3==0 else 'development'
        if sum(x['symbol']==r['symbol'] and x['kind']==r['kind'] and x['review_split']==split for x in review)>=30:continue
        info=pairmap[(r['symbol'],r['record_key'])]
        inp=json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False);digest=hashlib.sha256(inp.encode()).hexdigest()
        review.append({'sample_id':r['event_id'],'input_sha256':digest,'symbol':r['symbol'],'kind':r['kind'],'event_group':r['event_group'],'review_split':split,'title':info['title'],'evidence':r['evidence'],'proposed_fields':inp,'reviewer':'','critical_fields_correct':'','missed_target':'','uncertain':'','notes':''})
    panel=pd.DataFrame(review);panel.to_csv(out/'review_all.csv',index=False)
    check=panel[panel.review_split=='check'];check.to_csv(out/'independent_check.csv',index=False)
    dump(out/'review_fingerprints.json',dict(zip(check.sample_id,check.input_sha256)))
    dump(out/'quality_status.json',quality_gate(out/'independent_check.csv',out/'review_fingerprints.json'))
    # A second review panel evaluates newly found company coverage, not numeric extraction.
    new_ids={(u['symbol'],u['record_key']) for u in usage if u['split']=='train'}
    new=pairs[[((s,k) in new_ids) for s,k in zip(pairs.symbol,pairs.record_key)]].copy()
    new['order']=new.record_key.map(lambda k:hashlib.sha256(k.encode()).hexdigest());new=new.sort_values('order').drop_duplicates(['symbol','event_group']).groupby('symbol').head(40)
    new['independent_reviewer']='';new['direct_company_fact_correct']='';new['assistant_review']='';new.to_json(out/'coverage_review.jsonl',orient='records',lines=True,force_ascii=False)
    (out/'candidate_excerpts.md').write_text('# 新增覆盖候选：规则提名，未独立验收\n\n'+'\n\n'.join(f"## {r.symbol} / {r.record_key}\n\n{r.title}\n\n"+'\n\n'.join(s['text'] for s in r.evidence_spans)+f"\n\nEvent group: {r.event_group}" for r in new.itertuples()))
    dump(out/'funnel.json',{'raw_records':len(raw),'valid_language_body_nonnegative_time_nonphysician':len(valid),'records_available_in_any_original_window':len(eligible),'company_article_pairs':len(pairs),'rule_events':len(events),'status':'AUDIT_ONLY_NO_PREDICTOR_INPUT_CHANGED','grouping':'past-only normalized-content identity or last200 same-company titles within7days token-Jaccard>=.75; heuristic not human verified'})
    check_hashes(hashes);dump(out/'sources.json',hashes)
    print(pd.DataFrame(totals).to_string(index=False),flush=True)
    print('QUALITY',json.loads((out/'quality_status.json').read_text()),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();main(args.output)
