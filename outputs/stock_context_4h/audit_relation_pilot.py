"""Post-run audits only; never alters frozen Qwen relations."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'audit_v2';PRIVATE=ROOT/'work/stock-data/context_increment_4h/audit_v2'
def main():
 pairs=pd.read_json(PRIVATE/'news_pairs_private.jsonl',lines=True); outs=[json.loads(x) for x in (PRIVATE/'qwen_relations.jsonl').open() if x.strip()]; by={x['pair_id']:x for x in outs}; rows=[]
 for r in pairs.itertuples():
  x=by[r.pair_id];o=x.get('parsed_output') or {};rows.append({'target':r.target,'stratum':r.stratum,'split':r.split,'relation':o.get('pair_relation','INVALID'),'valid':not x['validation_errors'],'current_citations_nonempty':bool(o.get('current_evidence_ids')),'past_citations_nonempty':bool(o.get('past_evidence_ids')),'both_citations_nonempty':bool(o.get('current_evidence_ids')) and bool(o.get('past_evidence_ids')),'changes_nonempty':bool(o.get('changes')),'broad_novelty':o.get('pair_relation') in {'new_or_changed_fact','explicit_correction_or_denial'}})
 d=pd.DataFrame(rows)
 def rec(frame):return frame.groupby(['target','stratum','split','relation']).size().reset_index(name='count').to_dict('records')
 (OUT/'relation_reporting_audit.json').write_text(json.dumps({'all_generated_counts':rec(d),'mechanically_valid_only_counts':rec(d[d.valid]),'mechanically_invalid_count':int((~d.valid).sum())},indent=2)+'\n')
 coverage={k:int(d[k].sum()) for k in ['current_citations_nonempty','past_citations_nonempty','both_citations_nonempty','changes_nonempty','broad_novelty']};coverage['all_generated']=len(d);coverage['mechanically_valid']=int(d.valid.sum());(OUT/'relation_citation_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
 locked=d[d.split=='locked_check'].copy();template=[{'pair_id':r.pair_id,'fine_relation':'','evidence_sufficient':'','reviewer_uncertain':'','brief_reason':''} for r in pairs[pairs.split=='locked_check'].itertuples()];(PRIVATE/'locked_review_template.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in template));(OUT/'locked_review_template_manifest.json').write_text(json.dumps({'count':len(template),'sha256':__import__('hashlib').sha256((PRIVATE/'locked_review_template.jsonl').read_bytes()).hexdigest(),'human_fields_empty':True},indent=2)+'\n')
if __name__=='__main__':main()
