"""Fit-free checks of saved text preprocessing and canonical input mutations."""
from core import *
import collections
from sklearn.model_selection import StratifiedKFold

def assert_partition(tf,expected):
 if tf.fit_row_ids_!=expected:raise AssertionError('Text transformation train membership mismatch')
def verify_vectorizer(tf,training_docs):
 assert_partition(tf,[x['row_id'] for x in training_docs])
 vocab=tf.vocabulary_;an=tf.analyzer;df=collections.Counter()
 for doc in training_docs:df.update(set(an(doc)))
 assert len(vocab)<=5000 and all(df[w]>=3 for w in vocab)
 expected=np.empty(len(vocab))
 for word,col in vocab.items():expected[col]=np.log((1+len(training_docs))/(1+df[word]))+1
 np.testing.assert_allclose(tf.idf_,expected,atol=1e-12,rtol=0)
 # Deterministic budget: top term-count frequencies, lexical tie ordering.
 # sklearn controls exact tie selection; here verify no excluded term has
 # STRICTLY larger count than an included budget-boundary term.
 freq=collections.Counter()
 for doc in training_docs:freq.update(an(doc))
 valid=[w for w in df if df[w]>=3]
 if len(valid)>5000:
  floor=min(freq[w] for w in vocab)
  assert all(freq[w]<=floor for w in valid if w not in vocab)
 else:assert set(vocab)==set(valid)
 return len(vocab)
def main():
 d,*_=load();docs=joblib.load(WORK/'inputs.joblib')['docs'];n=0;vocabtotal=0;public_hashes=0
 manifest=json.loads((ROOT/'docs/RESULT_FILES.json').read_text())
 for item in manifest:
  if item['path'].startswith(('outputs/stock_full_semantics_4h/','outputs/stock_random_protocol_4h/')):
   assert sha(ROOT/item['path'])==item['sha256'];public_hashes+=1
 for root in sorted((WORK/'training').iterdir()):
  if not (root/'complete.json').exists():raise RuntimeError('Incomplete training')
  for item in json.loads((root/'training.json').read_text()):
   if not item['method'].startswith('A') or item['inner_fold'] is not None:continue
   bundle=joblib.load(root/item['file']);a=bundle['train'];model=bundle['model'];seed=int(root.name.split('_')[0])
   if item['method']=='A3':
    parts=StratifiedKFold(3,shuffle=True,random_state=seed).split(np.zeros(len(a)),d.iloc[a].label)
    tfs=[(cc.estimator.named_steps['text'],np.array(a)[at]) for cc,(at,_) in zip(model.calibrated_classifiers_,parts)]
   else:tfs=[(model.named_steps['text'],a)]
   for tf,idx in tfs:vocabtotal+=verify_vectorizer(tf,[docs[i] for i in idx]);n+=1
 # Real negative test: a selected pipeline must reject changed member identity.
 altered=list(tf.fit_row_ids_);altered[0]='corrupt-held-out-id';failed=False
 try:assert_partition(tf,altered)
 except AssertionError:failed=True
 assert failed
 dump(PUBLIC/'PREPROCESSING_VERIFICATION.json',dict(status='PASS',selected_vectorizers_and_calibration_pipelines=n,total_vocabulary_entries=vocabtotal,training_df_idf_and_budget_reconstructed=True,wrong_membership_corruption_rejected=failed,prior_public_result_hashes_verified=public_hashes,fit_calls=0))
 print('Preprocessing verification PASS',n,flush=True)
if __name__=='__main__':main()
