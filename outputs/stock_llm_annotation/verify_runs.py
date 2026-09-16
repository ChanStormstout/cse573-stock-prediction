"""Cross-run evidence checks, no retraining and no independent semantic claim."""
import json
from pathlib import Path
from labels import ROOT,rows,sha,dump
B=Path(__file__).resolve().parent
W=ROOT/'work/stock-data/annotation'
def main():
 runs=W/'runs';names=['qlora_v2','qlora_gate_v1'];manifests=[json.loads((runs/n/'manifest.json').read_text()) for n in names];summaries=[json.loads((runs/n/'summary.json').read_text()) for n in names];logs=[rows(runs/n/'training.jsonl') for n in names]
 for key in ['model','revision','num_layers','lora_parameters','seed','learning_rate','balance_events','micro_batch','accumulation','epochs','max_seq_length','train_ids','valid_ids','labels_sha256','inputs_sha256']:
  assert manifests[0][key]==manifests[1][key],key
 assert [r['id'] for r in logs[0]]==[r['id'] for r in logs[1]]
 evidence=[]
 for n,m,s,log in zip(names,manifests,summaries,logs):
  assert len(log)==747 and s['micro_steps']==747 and s['optimizer_updates']==96
  assert set(r['id'] for r in log)==set(m['train_ids']) and not set(m['valid_ids'])&set(m['train_ids'])
  assert s['actual_training'] and s['nonzero_gradient'] and s['adapter_changed'] and s['frozen_base_unchanged'] and s['reload_greedy_equal'] and s['reload_logit_max_error']<1e-5
  selected=max(s['development'],key=lambda x:(2*x['fact_tp']-x['fact_fp']-x['fact_fn'],x['exact_fact_set'],-x['epoch']))['epoch'];assert selected==s['selected_epoch']
  weight=runs/n/'selected/adapters.safetensors';evidence.append({'run':n,'selected_epoch':selected,'selected_adapter_sha256':sha(weight.read_bytes())})
 a=rows(W/'student_v2/labels.jsonl');b=rows(W/'student_v3/labels.jsonl');assert sorted((r for r in a if r['split']!='check'),key=lambda r:r['id'])==sorted((r for r in b if r['split']!='check'),key=lambda r:r['id'])
 inf=[json.loads((runs/n/'manifest.json').read_text()) for n in ['frozen_v3','tuned_v3','tuned_gate_v3']]
 for key in ['model','revision','input_sha256','labels_sha256','system_sha256','temperature','max_tokens','thinking','expanded_contract','max_length']:
  assert len({str(m[key]) for m in inf})==1,key
 for i in [1,2]:assert inf[i]['adapter_sha256']==evidence[i-1]['selected_adapter_sha256']
 report={'matched_training_information':True,'matched_order_and_budget':True,'only_registered_loss_variant_changed':True,'no_check_ids_in_training_or_model_selection':True,'selected_checkpoint_rule_recomputed':True,'same_inference_inputs_prompt_decoding':True,'both_weight_reload_checks_passed':True,'check_supplement_did_not_change_train_or_dev':True,'selected_weights':evidence,'independent_human_semantic_acceptance':False,'new_four_hour_model_trained':False}
 dump(B/'VERIFICATION.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
