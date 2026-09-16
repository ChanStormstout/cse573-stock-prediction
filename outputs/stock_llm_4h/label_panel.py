"""Assistant-provisional labels authored after reading the displayed numbered inputs.
No stock outcomes used. This is not independent human annotation.
"""
from common import *

def event(kind,action,old,new,*ids):return {'kind':kind,'action':action,'old':old,'new':new,'unit':'USD' if kind=='target_price' else 'rating','evidence_ids':list(ids)}
LABELS={
'L03':[event('target_price','maintain',None,'190.00','S0','S2')],
'L04':[event('rating','lower','buy','neutral','S7')],
'L05':[event('target_price','raise','200.00','220.00','S1')],
'L08':[event('target_price','raise','193.00','204.00','S1')],
'L09':[event('target_price','unknown',None,'195.00','S1')],
'L10':[event('target_price','unknown',None,'174.00','S2')],
'L11':[event('rating','lower','buy','neutral','S2')],
'L14':[event('rating','lower','overweight','neutral','S1','S2'),event('target_price','maintain',None,'190','S1','S2')],
'L15':[event('rating','lower','Buy','Neutral','S6')],
'L24':[event('rating','lower','Overweight','Sector Weight','S2')],
'L25':[event('rating','maintain',None,'outperform','S2')],
'L26':[event('rating','maintain',None,'positive','S0','S3'),event('target_price','raise','190.00','200.00','S0','S3','S4')],
'L27':[event('rating','maintain',None,'equal weight','S2')],
'L28':[event('rating','maintain',None,'buy','S1'),event('target_price','raise','204','210','S1','S4')],
'L29':[event('rating','maintain',None,'buy','S2'),event('target_price','lower','200.00','195.00','S2','S3')],
'L40':[event('rating','lower','strong-buy','buy','S2')],
'L41':[event('target_price','unknown',None,'205','S1')],
'L42':[event('rating','maintain',None,'market perform','S2')],
}
NOTES={'L03':'Title explicitly reiterates while lead says set; no old number inferred. Title/lead contrast flagged for human review.', 'L05':'Present buy rating is not a reported rating action; only current target increase. Historical ratings excluded.', 'L09':'Set target is reported but change relative to old value is unknown.', 'L10':'Current target assignment; no inferred rating maintenance from currently neutral.', 'L14':'Two current facts; kept old target null because retained does not explicitly quote old number separately.', 'L24':'Target withdrawal is outside limited action vocabulary; retained in exclusion note, not claimed as full-schema recall. Fair-value estimate is not a newly assigned target.', 'L25':'Static current price target omitted; only explicit restatement is an action.', 'L26':'Baird current events only; other broker historical/current lists outside PRIMARY disclosure scope.', 'L28':'Current target increase occurs later in quoted commentary, not lead. No historical analyst lists.', 'L31':'Body discusses end-of-day changes despite morning crawl metadata; input freshness cannot be externally certified. No analyst action.', 'L32':'Personal valuation/recommendation, no explicit brokerage analyst action. Truncated headline conflicts with body value; no inferred price target change.', 'L39':'Non-news / no usable evidence.', 'L40':'Current Vetr downgrade only; static target and earlier other-broker updates excluded.', 'L41':'Explicit set target, no assertion of initiation or old value.', 'L42':'Explicit rating reiterated; old list excluded.'}

def main():
 p=B/'data/pilot_v3';rs=rows(p/'inputs.jsonl');out=[]
 for r in rs:
  answer={'events':LABELS.get(r['id'],[])};ok,why=validate(answer,r);assert ok,(r['id'],why)
  group='AAPL_Longbow_2018_01_17' if r['id'] in {'L04','L11','L15'} else r['group']
  # Two extra reports of same downgrade stay in diagnostic set, never extra SFT examples.
  split='train_duplicate_diagnostic' if r['id'] in {'L11','L15'} else r['split']
  out.append({'id':r['id'],'split':split,'symbol':r['symbol'],'event_group':group,'available_utc':r['available_utc'],'answer':answer,'annotation_status':'assistant_provisional','independent_reviewer':None,'notes':NOTES.get(r['id'],'No supported current analyst action in the supplied input; historical/other-company facts do not qualify.' if not answer['events'] else 'Current primary target-company action supported by cited input.'),'input_sha256':digest(json.dumps(r['sentences'],sort_keys=True).encode())})
 dest=p/'labels.jsonl'
 if dest.exists():raise RuntimeError('Labels already sealed')
 write_rows(dest,out)
 dump(p/'label_seal.json',{'sha256':file_sha(dest),'inputs_sha256':file_sha(p/'inputs.jsonl'),'status':'ASSISTANT_PROVISIONAL_NO_INDEPENDENT_REVIEW','scope':'Current PRIMARY brokerage analyst rating/target actions only; excludes target withdrawal, personal investment theses, static ratings/targets and other-broker history. Not full event schema acceptance.','frozen_before_fact_inference':True,'n':len(out),'positive_articles':sum(bool(r['answer']['events']) for r in out),'train_unique':sum(r['split']=='train' for r in out),'warning':'Small, manually reviewed input panel; not 300–600 high-quality examples. All AMZN examples here are no-event. Late check has supplemental positives; not a representative random sample.'})
 print((p/'label_seal.json').read_text())
if __name__=='__main__':main()
