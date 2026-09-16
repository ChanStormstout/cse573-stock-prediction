"""Three fixed probes to distinguish mask bugs from prompt/model behavior; no tuning."""
import os,json,string
from pathlib import Path
B=Path(__file__).resolve().parent;os.environ['HF_HOME']=str(B.parents[1]/'work/stock-data/llm-cache')
import pandas as pd
import mlx.core as mx
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler
p=json.loads((B/'protocol.json').read_text())['LLM'];model,tok=load(p['model'],revision=p['revision']);letters=string.ascii_uppercase[:9];ids=[tok.encode(x,add_special_tokens=False)[0] for x in letters];allowed=mx.array(ids)
def restrict(logits):return mx.where(mx.any(mx.arange(logits.shape[-1])[:,None]==allowed[None,:],axis=1)[None,:],logits,-float('inf'))
fake=mx.zeros((1,max(ids)+20));fake=fake.at[0,ids[5]].add(10);masked=restrict(fake);assert int(mx.argmax(masked))==ids[5];assert int(mx.sum(mx.isfinite(masked)))==9
q=pd.read_csv(B.parent/'stock_structured/pilot_100_inputs.csv');prompt=(B/'prompt_v3.txt').read_text();out=[]
for i in [4,6,12]:
 r=q.iloc[i];source=[r.title]+[s for s in r.target.split('\n') if s.strip()];text=f'TARGET: {r.symbol}\n'+'\n'.join(f'S{j}: {s}' for j,s in enumerate(source));rendered=tok.apply_chat_template([{'role':'system','content':prompt},{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 logits=model(mx.array(tok.encode(rendered,add_special_tokens=False))[None,:])[:,-1,:];v=logits[0,allowed].tolist();winner=letters[v.index(max(v))];answer=generate(model,tok,prompt=rendered,max_tokens=48,sampler=make_sampler(temp=0),verbose=False)
 out.append(dict(index=i,allowed_letter_logits=dict(zip(letters,v)),constrained_argmax=winner,unconstrained_probe=answer));print(out[-1],flush=True)
(B/'results/decoding_diagnostics.json').write_text(json.dumps(dict(mask_unit_check='passed; exactly nine finite choices, prescribed synthetic maximum retained',token_ids=dict(zip(letters,ids)),probes=out,limitation='Diagnostic probes only; no prompt retuning or new performance run'),indent=2))
