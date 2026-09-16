"""Actual local QLoRA with explicit non-thinking prefix and completion-only loss.
Tiny pilot, not full corpus adaptation. No check labels used during model selection.
"""
import os,time,argparse,math,shutil
from common import *
os.environ['HF_HOME']=str(ROOT/'work/stock-data/llm-cache')
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_map
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.trainer import grad_checkpoint

def token_example(tok,r,label,max_length=2048):
 prefix=tok.apply_chat_template(messages(r),tokenize=False,add_generation_prompt=True,enable_thinking=False)
 p=tok.encode(prefix,add_special_tokens=False);answer=json.dumps(label['answer'],separators=(',',':'))
 a=tok.encode(answer,add_special_tokens=False)+[tok.eos_token_id]
 assert tok.decode(a[:-1])==answer
 tokens=p+a
 if len(tokens)>max_length:raise ValueError(f"Refuse answer truncation: {r['id']} {len(tokens)}")
 return tokens,len(p),prefix

def array_hash(items):
 h=hashlib.sha256()
 for name,a in sorted(items):
  mx.eval(a);h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(np.asarray(a.astype(mx.float32) if a.dtype==mx.bfloat16 else a).tobytes())
 return h.hexdigest()

def loss(model,tokens,offset,branch_position=-1,branch_weight=0.):
 x=mx.array(tokens)[None,:];logits=model(x[:,:-1]);target=x[:,1:]
 mask=mx.arange(1,len(tokens))>=offset
 ce=nn.losses.cross_entropy(logits,target)
 ordinary=(ce*mask[None,:]).sum()/mask.sum()
 return ordinary+branch_weight*ce[0,branch_position-1] if branch_position>=offset else ordinary

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--data',type=Path,default=B/'data/pilot_v3');a.add_argument('--smoke',action='store_true');a.add_argument('--balance-events',action='store_true');a.add_argument('--event-gate-loss',action='store_true');a.add_argument('--seed',type=int,default=573);a.add_argument('--expanded-contract',action='store_true');a.add_argument('--max-length',type=int,default=2048);args=a.parse_args();out=new_run(args.out)
 if args.event_gate_loss and not args.expanded_contract:raise ValueError('Gate-loss experiment requires expanded contract')
 if args.expanded_contract:
  import sys
  sys.path.insert(0,str(ROOT/'outputs/stock_llm_annotation'))
  import model_contract
  globals()['messages']=model_contract.messages
 if args.max_length not in (2048,3072):raise ValueError('Unregistered token budget')
 check_panel(args.data)
 rs={r['id']:r for r in rows(args.data/'inputs.jsonl')};labels=rows(args.data/'labels.jsonl');train_labels=[r for r in labels if r['split']=='train'];valid_labels=[r for r in labels if r['split']=='valid']
 if args.smoke:
  pos=[r for r in train_labels if r['answer']['events']][:4];neg=[r for r in train_labels if not r['answer']['events']][:4];train_labels=pos+neg
 config={'model':MODEL,'revision':REVISION,'fine_tune_type':'lora','num_layers':8,'lora_parameters':{'rank':8,'scale':16.,'dropout':.05,'keys':['self_attn.q_proj','self_attn.v_proj']},'seed':args.seed,'learning_rate':1e-4,'balance_events':args.balance_events,'micro_batch':1,'accumulation':8,'epochs':3 if not args.smoke else None,'smoke_micro_steps':32 if args.smoke else None,'max_seq_length':args.max_length,'expanded_contract':args.expanded_contract,'thinking':False,'assistant_only_loss':True,'train_ids':[r['id'] for r in train_labels],'valid_ids':[r['id'] for r in valid_labels],'labels_sha256':file_sha(args.data/'labels.jsonl'),'inputs_sha256':file_sha(args.data/'inputs.jsonl'),'source_sha256':file_sha(__file__),'common_sha256':file_sha(B/'common.py'),'quality_status':'MODEL_PROVISIONAL_NOT_INDEPENDENT' if args.expanded_contract else 'ASSISTANT_PROVISIONAL_NOT_INDEPENDENT'}

 if args.expanded_contract:
  config['contract_sha256']=file_sha(ROOT/'outputs/stock_llm_annotation/model_contract.py')
  config['label_seal_sha256']=file_sha(args.data/'label_seal.json')
  config['extractor_earliest_freeze']=json.loads((args.data/'label_seal.json').read_text())['extractor_frozen_at']
 config['event_gate_loss']=args.event_gate_loss
 dump(out/'manifest.json',config);(out/'code_snapshot').mkdir();
 if args.expanded_contract:shutil.copy2(ROOT/'outputs/stock_llm_annotation/model_contract.py',out/'code_snapshot/model_contract.py')
 shutil.copy2(__file__,out/'code_snapshot/train_adapter.py');shutil.copy2(B/'common.py',out/'code_snapshot/common.py');mx.random.seed(args.seed);rng=np.random.default_rng(args.seed)
 model,tok=load(MODEL,revision=REVISION);model.freeze();linear_to_lora_layers(model,8,config['lora_parameters']);mx.eval(model.parameters())
 trainable=dict(tree_flatten(model.trainable_parameters()));assert len(trainable)==32 and all('lora_' in k for k in trainable),list(trainable)
 frozen=[(k,v) for k,v in tree_flatten(model.parameters()) if k not in trainable];base_before=array_hash(frozen);adapter_before=array_hash(trainable.items());print('Trainable parameters',sum(x.size for x in trainable.values()),flush=True)
 grad_checkpoint(model.layers[0]);examples={r['id']:token_example(tok,rs[r['id']],r,args.max_length) for r in train_labels+valid_labels};dump(out/'token_audit.json',{'examples':[{'id':k,'tokens':len(v[0]),'prompt_tokens':v[1],'supervised_tokens':len(v[0])-v[1],'first_supervised':tok.decode(v[0][v[1]:v[1]+3])} for k,v in examples.items()],'truncation':False,'prompt_excluded':True})
 optimizer=optim.Adam(learning_rate=1e-4);value_grad=nn.value_and_grad(model,loss);logs=[];evals=[];steps=0;updates=0;start=time.time();nonzero=False;seen=[]
 positive_n=sum(bool(r['answer']['events']) for r in train_labels)
 pos_weight=len(train_labels)*(2/3)/positive_n if args.balance_events else 1.
 neg_weight=len(train_labels)*(1/3)/(len(train_labels)-positive_n) if args.balance_events else 1.
 gate_index=model_contract.branch_index(tok) if args.event_gate_loss else -1
 dump(out/'loss_weights.json',{'positive':pos_weight,'negative':neg_weight,'positive_articles':positive_n,'negative_articles':len(train_labels)-positive_n,'no_duplication':True})
 total_epochs=4 if args.smoke else 3
 for epoch in range(1,total_epochs+1):
  model.train();accum=None;acc_n=0
  for idx in rng.permutation(len(train_labels)):
   label=train_labels[int(idx)];tokens,offset,_=examples[label['id']];tick=time.time()
   weight=pos_weight if label['answer']['events'] else neg_weight
   gate_class_weight=len(train_labels)/(2*(positive_n if label['answer']['events'] else len(train_labels)-positive_n))
   branch_weight=gate_class_weight/weight if args.event_gate_loss else 0.
   l,g=value_grad(model,tokens,offset,offset+gate_index if args.event_gate_loss else -1,branch_weight);mx.eval(l,g)
   value=float(l.item());assert math.isfinite(value)
   if not nonzero:nonzero=any(bool(mx.any(v!=0).item()) for _,v in tree_flatten(g))
   g=tree_map(lambda x:x*weight,g)
   accum=g if accum is None else tree_map(lambda x,y:x+y,accum,g);acc_n+=1;steps+=1;seen.append(label['id'])
   last=acc_n==8 or len(seen)%len(train_labels)==0
   if last:
    optimizer.update(model,tree_map(lambda x:x/acc_n,accum));mx.eval(model.parameters(),optimizer.state);updates+=1;accum=None;acc_n=0
   logs.append({'micro_step':steps,'epoch':epoch,'id':label['id'],'loss':value,'loss_weight':weight,'branch_weight_before_class_weight':branch_weight,'branch_token_index':gate_index,'seconds':time.time()-tick,'optimizer_updates':updates,'peak_gb':mx.get_peak_memory()/1e9});write_rows(out/'training.jsonl',logs)
   if steps%4==0:print('STEP',steps,'loss',round(value,4),'peakGB',round(mx.get_peak_memory()/1e9,2),flush=True)
  checkpoint=out/f'epoch_{epoch}';checkpoint.mkdir();dump(checkpoint/'adapter_config.json',config);mx.save_safetensors(str(checkpoint/'adapters.safetensors'),dict(tree_flatten(model.trainable_parameters())))
  if not args.smoke:
   model.eval();preds=[]
   for r in valid_labels:
    inp=rs[r['id']];prompt=examples[r['id']][2];answer=generate(model,tok,prompt=prompt,max_tokens=512,sampler=make_sampler(temp=0),verbose=False);obj=parse(answer);ok,errors=validate(obj,inp);preds.append({'id':r['id'],'raw':answer,'parsed':obj,'valid':ok,'errors':errors})
   metrics=evaluate(preds,valid_labels)['valid'];evals.append({'epoch':epoch,**metrics});write_rows(checkpoint/'development_predictions.jsonl',preds);dump(checkpoint/'development_metrics.json',metrics);print('DEV',epoch,metrics,flush=True)
  mx.clear_cache()
 # Selection only on development facts, then exact sets then earlier checkpoint. Never check labels.
 selected=total_epochs if args.smoke else max(evals,key=lambda x:(2*x['fact_tp']-x['fact_fp']-x['fact_fn'],x['exact_fact_set'],-x['epoch']))['epoch']
 selected_dir=out/'selected';shutil.copytree(out/f'epoch_{selected}',selected_dir)
 base_after=array_hash([(k,v) for k,v in tree_flatten(model.parameters()) if k not in trainable]);adapter_after=array_hash(tree_flatten(model.trainable_parameters()))
 assert base_after==base_before and adapter_after!=adapter_before and nonzero
 # Validate the selected saved adapter by comparing logits and greedy generation after reloading in a fresh model.
 model.load_weights(str(selected_dir/'adapters.safetensors'),strict=False);model.eval();sample=train_labels[0];tokens,offset,prompt=examples[sample['id']]
 before=np.asarray(model(mx.array(tokens[:offset])[None,:])[:,-1,:].astype(mx.float32));answer_before=generate(model,tok,prompt=prompt,max_tokens=128,sampler=make_sampler(temp=0),verbose=False)
 del model;mx.clear_cache();reloaded,_=load(MODEL,revision=REVISION,adapter_path=str(selected_dir));reloaded.eval();after=np.asarray(reloaded(mx.array(tokens[:offset])[None,:])[:,-1,:].astype(mx.float32));answer_after=generate(reloaded,tok,prompt=prompt,max_tokens=128,sampler=make_sampler(temp=0),verbose=False)
 error=float(np.max(np.abs(before-after)));assert error<1e-5 and answer_before==answer_after
 result={'actual_training':True,'kind':'SMOKE_ONLY' if args.smoke else ('EXPANDED_MODEL_PROVISIONAL' if args.expanded_contract else 'TINY_ASSISTANT_LABELLED_PILOT'),'seed':args.seed,'micro_steps':steps,'optimizer_updates':updates,'examples_seen':len(seen),'unique_train_articles':len(train_labels),'trainable_parameters':sum(v.size for v in trainable.values()),'seconds':time.time()-start,'peak_memory_gb':mx.get_peak_memory()/1e9,'nonzero_gradient':nonzero,'frozen_base_unchanged':base_before==base_after,'adapter_changed':adapter_before!=adapter_after,'base_before_sha256':base_before,'base_after_sha256':base_after,'adapter_before_sha256':adapter_before,'adapter_after_sha256':adapter_after,'selected_epoch':selected,'development':evals,'reload_logit_max_error':error,'reload_greedy_equal':answer_before==answer_after,'independent_review_passed':False,'predictive_improvement_claim':False}
 dump(out/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
