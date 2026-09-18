"""Input-only coverage/concentration diagnostics; no evaluation direction labels."""
from common import *

def main():
    check_prepared();d=pd.read_pickle(PRIVATE/'rows.pkl');a=json.loads((PRIVATE/'retrieval.json').read_text())
    docs=json.loads((PRIVATE/'documents.json').read_text());rows=[];categories=[]
    for i,r in d.iterrows():
        if r.month<'2018-03':continue
        phase='train_oof' if r.month<'2018-09' else ('development' if r.month<'2018-11' else 'later')
        categories.append(dict(symbol=r.symbol,phase=phase,category=docs[i]['category'] if docs[i] else 'no_excerpt',
                               has_analog=int(len(a[i]['case_indices'])>0)))
        for rank,(j,score) in enumerate(zip(a[i]['case_indices'],a[i]['scores']),1):
            case=d.iloc[j];inp=docs[j]['input']
            rows.append(dict(symbol=r.symbol,phase=phase,query_key=r.key,case_index=j,rank=rank,
                             score=score,case_age_days=float((r.cutoff_utc-case.end_utc).total_seconds()/86400),
                             case_news_age_hours=inp['news']['age_hours'],case_price_age_hours=inp['price']['history_age_hours'],
                             query_news_age_hours=docs[i]['input']['news']['age_hours']))
    rr=pd.DataFrame(rows);rr.to_csv(OUT/'retrieval_diagnostics.csv',index=False)
    summary=[]
    for (s,p),g in rr.groupby(['symbol','phase']):
        counts=g.case_index.value_counts()
        summary.append(dict(symbol=s,phase=p,retrieval_slots=len(g),unique_historical_cases=int(len(counts)),
            most_reused_case_count=int(counts.max()),largest_case_fraction=float(counts.max()/len(g)),
            median_case_age_days=float(g.case_age_days.median()),
            median_case_news_age_hours=float(g.case_news_age_hours.median())))
    pd.DataFrame(summary).to_csv(OUT/'retrieval_summary.csv',index=False)
    pd.DataFrame(categories).groupby(['symbol','phase','category']).agg(windows=('has_analog','size'),
        with_analog=('has_analog','sum')).to_csv(OUT/'category_coverage.csv')
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':main()
