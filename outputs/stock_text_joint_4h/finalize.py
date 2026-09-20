"""Wait for frozen training/no-fit verification, then run only report checks."""
from core import *
import time,subprocess,datetime,io,unittest,importlib.util

def main():
 while not (PUBLIC/'VERIFICATION.json').exists():
  if not any('/stock_text_joint_4h/watch_verify.py' in line for line in subprocess.check_output(['ps','-axo','command'],text=True).splitlines()):raise RuntimeError('Verification watcher stopped without a final audit; preserve outputs')
  time.sleep(15)
 assert json.loads((PUBLIC/'VERIFICATION.json').read_text())['status']=='PASS'
 while not (PUBLIC/'training_evidence.json').exists():time.sleep(1)
 for file in ['review_checks.py','feature_diagnostics.py']:
  subprocess.run([sys.executable,str(Path(__file__).with_name(file))],check=True)
 spec=importlib.util.spec_from_file_location('text_joint_contract_tests',Path(__file__).with_name('test_contracts.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);buffer=io.StringIO();result=unittest.TextTestRunner(stream=buffer,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module));assert result.wasSuccessful()
 dump(PUBLIC/'TEST_RESULTS.json',dict(status='PASS',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),details=buffer.getvalue()))
 stamps={p.name:datetime.datetime.fromtimestamp(p.stat().st_birthtime,datetime.timezone.utc).isoformat() for p in [PUBLIC/'protocol.json',PUBLIC/'INPUT_AUDIT.json',WORK/'training_seal.json']}
 first=min(p.stat().st_birthtime for p in (WORK/'training').glob('*/*.joblib'))
 assert (PUBLIC/'protocol.json').stat().st_birthtime<first and (PUBLIC/'INPUT_AUDIT.json').stat().st_birthtime<first
 dump(PUBLIC/'REPRODUCIBILITY.json',dict(frozen_training_seal=json.loads((WORK/'training_seal.json').read_text()),file_birth_times_utc=stamps,first_saved_new_model_utc=datetime.datetime.fromtimestamp(first,datetime.timezone.utc).isoformat(),note='Local filesystem timestamps plus frozen hashes; not an external timestamp attestation',raw_data_or_weights_published=False))
 subprocess.run([sys.executable,str(Path(__file__).with_name('report.py'))],check=True)
 print('Finite study verified and reported; no advanced continuation',flush=True)
if __name__=='__main__':main()
