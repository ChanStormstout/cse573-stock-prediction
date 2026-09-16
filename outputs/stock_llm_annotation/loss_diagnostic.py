"""Explain the relative weight of the event/empty branching decision in SFT."""
import argparse,json
from pathlib import Path
from labels import rows,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 labels={r['id']:r for r in rows(a.data/'labels.jsonl')};weights=json.loads((a.run/'loss_weights.json').read_text());audit=json.loads((a.run/'token_audit.json').read_text())['examples'];groups={True:[],False:[]}
 for r in audit:
  l=labels[r['id']]
  if l['split']!='train':continue
  event=bool(l['answer']['events']);w=weights['positive' if event else 'negative'];groups[event].append(w/r['supervised_tokens'])
 result={'interpretation':'Per-article mean token CE gives each one-token branching decision inverse-completion-length weight. This is objective accounting, not proof of the only cause of failed generation.','positive_articles':len(groups[True]),'negative_articles':len(groups[False]),'positive_gate_weight_per_epoch':sum(groups[True]),'negative_gate_weight_per_epoch':sum(groups[False]),'negative_to_positive_gate_weight_ratio':sum(groups[False])/sum(groups[True]),'does_not_use_check_predictions':True}
 dump(a.out,result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
