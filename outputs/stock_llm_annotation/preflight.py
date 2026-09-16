"""Check all prepared inputs fit the registered student budget, without training."""
import argparse
import os
from pathlib import Path
from labels import ROOT, load_panel, dump, sha
from model_contract import messages, SYSTEM

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--max-length',type=int,choices=[2048,3072],default=2048);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HOME']=str(ROOT/'work/stock-data/llm-cache')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('mlx-community/Qwen3-1.7B-4bit',revision='3b1b1768f8f8cf8351c712464f906e86c2b8269e',local_files_only=True)
    rs,manifest=load_panel(a.data);result=[]
    for r in rs:
        prefix=tok.apply_chat_template(messages(r),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        training_ids=tok.encode(prefix,add_special_tokens=False)
        # Match the installed MLX stream_generate special-token inference rule.
        generation_special=tok.bos_token is None or not prefix.startswith(tok.bos_token)
        inference_ids=tok.encode(prefix,add_special_tokens=generation_special)
        if training_ids!=inference_ids:raise ValueError('Training/inference prefix token mismatch')
        count=len(training_ids)
        result.append({'id':r['id'],'prompt_tokens':count,'fits_with_512_output':count+512<=a.max_length})
    dump(out/'tokens.json',result)
    summary={'n':len(rs),'max_prompt_tokens':max(x['prompt_tokens'] for x in result),'over_budget':sum(not x['fits_with_512_output'] for x in result),'input_sha256':manifest['inputs_sha256'],'student_contract_sha256':sha(SYSTEM.encode()),'training_run':False,'context_tokens':a.max_length,'reserved_output':512,'truncation':False,'all_training_inference_prefix_tokens_identical':True}
    dump(out/'summary.json',summary);print(summary)
    if summary['over_budget']: raise ValueError('Input budget exceeded; do not silently truncate')

if __name__=='__main__':main()
