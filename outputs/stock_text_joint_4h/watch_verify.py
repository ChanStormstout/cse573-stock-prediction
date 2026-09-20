"""Overlap independent read-only replay with completed immutable train blocks."""
from core import *
import subprocess,time

def main():
 verifier=Path(__file__).with_name('verify.py');names=[f'{seed}_{stock}_{fold}' for seed in SEEDS for stock in ['AAPL','AMZN'] for fold in range(10)]
 for name in names:
  root=WORK/'training'/name
  while not (root/'complete.json').exists():time.sleep(10)
  fragment=WORK/'verification_fragments'/f'{name}.json'
  if fragment.exists():
   cached=json.loads(fragment.read_text())
   if cached.get('verifier_sha256')==sha(verifier) and cached.get('complete_sha256')==sha(root/'complete.json'):continue
  subprocess.run([sys.executable,str(verifier),'--block',name],check=True)
 subprocess.run([sys.executable,str(verifier)],check=True)
 print('Full no-fit verification completed',flush=True)
if __name__=='__main__':main()
