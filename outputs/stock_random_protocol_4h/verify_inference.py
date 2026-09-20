"""Read-only integrity check for all frozen LLM outputs; no inference or fit."""
import json,math
from common import PRIVATE,OUT,sha,dump
def main():
 src=PRIVATE/'analogy';complete=json.loads((src/'inference_complete.json').read_text());seal=json.loads((src/'inference_seal.json').read_text())
 prompts=[json.loads(s) for s in (src/'prompts.jsonl').read_text().splitlines()];rows=[json.loads(s) for s in (src/'generated.jsonl').read_text().splitlines()]
 checks=dict(output_hash=sha(src/'generated.jsonl')==complete['output_sha256'],seal_hash=sha(src/'inference_seal.json')==complete['seal_sha256'],input_manifest_hash=seal['manifest']==sha(src/'manifest.json'),unique_exhaustive_prompts=len(rows)==len({r['prompt_hash'] for r in rows})==len(prompts) and {r['prompt_hash'] for r in rows}=={p['prompt_hash'] for p in prompts},probabilities=True,choice_mass=True,token_budget=True)
 maxerr=0.
 for r in rows:
  lp=[r['logp_up'],r['logp_down']];a=max(lp);z=a+math.log(sum(math.exp(v-a) for v in lp));p=math.exp(lp[0]-z);err=abs(p-r['p']);maxerr=max(maxerr,err)
  checks['probabilities'] &= math.isfinite(r['p']) and 0<=r['p']<=1 and err<1e-12
  checks['choice_mass'] &= abs(math.exp(z)-r['choice_mass'])<1e-12 and 0<=r['choice_mass']<=1+1e-6
  checks['token_budget'] &= 0<r['prompt_tokens']<=6000
 result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,calls=len(rows),max_token_probability_reconstruction_error=maxerr,model_calls_by_verifier=0,estimator_fits=0,predictive_scores_computed=False)
 dump(OUT/'LLM_VERIFICATION.json',result);print(json.dumps(result,indent=2))
 if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
