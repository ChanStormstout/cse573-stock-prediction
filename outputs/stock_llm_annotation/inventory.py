"""Publish aggregate inventory only; source text and labels stay local."""
import argparse,collections,json
from pathlib import Path
from labels import rows,dump,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--sealed',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 inputs=rows(a.data/'inputs.jsonl');comparisons=[r for d in (a.data/'comparisons').iterdir() if (d/'comparison.jsonl').exists() for r in rows(d/'comparison.jsonl')];assert len(comparisons)==len(inputs)
 sealed=json.loads((a.sealed/'label_seal.json').read_text());decisions=rows(a.data/'decisions.jsonl')
 report={'candidate_pairs':len(inputs),'candidate_split_counts':dict(collections.Counter(r['split'] for r in inputs)),'teacher_output_rows':2*len(comparisons),'teacher':'GPT 6 High via ChatGPT UI, two blind conversations per cohort, not independent humans','field_agreement':sum(r['field_agreement'] for r in comparisons),'evidence_agreement':sum(r['evidence_agreement'] for r in comparisons),'assistant_reviewed_pairs':len(decisions),'accepted_pairs':sealed['n'],'accepted_counts':sealed['counts'],'excluded_by_reason':dict(collections.Counter(r['reason'] for r in sealed['excluded'])),'independent_human_review_passed':False,'input_sha256':sha((a.data/'inputs.jsonl').read_bytes()),'labels_sha256':sealed['sha256'],'raw_news_in_git':False,'model_provisional_labels_in_git':False,'adapters_in_git':False,'earliest_extractor_freeze':sealed['extractor_frozen_at']}
 dump(a.out,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
