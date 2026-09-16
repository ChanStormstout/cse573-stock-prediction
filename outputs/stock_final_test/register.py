"""Freeze this comparison before preparing/predicting the final holdout."""
import json,hashlib,datetime
from pathlib import Path
from importlib.metadata import version
import pandas as pd,joblib
B=Path(__file__).resolve().parent;O=B.parent;ROOT=O.parent
assert not (B/'protocol.json').exists(),'The registered comparison must not be overwritten.'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['before_sha256.json','new_experiment_sha256.json']:
    for p,h in json.loads((O/'stock_review_fixes'/name).read_text()).items():assert sha(ROOT/p)==h,p
linear=json.loads((O/'stock_robust/results/linear/results.json').read_text());l2=json.loads((O/'stock_text_regularization/results/results.json').read_text());cal=json.loads((O/'stock_calibration/results/results.json').read_text());dev=pd.read_pickle(O/'stock_robust/results/data.pkl')
models={};paths=[B/'register.py',B/'prepare.py',B/'run.py',O/'stock_robust/common.py',O/'stock_robust/prepare.py',O/'stock_robust/results/prepared.json',O/'stock_robust/results/data.pkl',O/'stock_baseline/run_baseline.py',O/'stock_baseline/results/results.json',O/'stock_review_fixes/guards.py',O/'stock_content/results/source_bodies.json',O/'stock_robust/results/linear/results.json',O/'stock_text_regularization/results/results.json',O/'stock_calibration/results/results.json',ROOT/'work/stock-data/audit/news_index.pkl',ROOT/'work/stock-data/audit/xnys_schedule.csv']
paths+=list((ROOT/'work/stock-data/raw/news').glob('*.zip'))
paths += [ROOT/f'work/stock-data/raw/CHARTS/{prefix}5.csv' for prefix in ['APPLE','AMAZON']]
for s in ['AAPL','AMZN']:
    models[s]={};paths.append(O/f'stock_baseline/results/{s}/samples.csv')
    for label,kind,folder,results in [('price','price','stock_robust/results/linear',linear),('title_l1','paper_stem_title','stock_robust/results/linear',linear),('body_l1','paper_stem_body','stock_robust/results/linear',linear),('title_l2','paper_stem_title','stock_text_regularization/results',l2),('body_l2','paper_stem_body','stock_text_regularization/results',l2)]:
        file=O/f'{folder}/{s}/{kind}.joblib';fit=joblib.load(file);assert fit['model'].C==results['stocks'][s][kind]['C']
        models[s][label]=dict(path=str(file.relative_to(ROOT)),sha256=sha(file),C=fit['model'].C,l1_ratio=fit['model'].l1_ratio)
        paths.extend([file,O/f'{folder}/{s}/{kind}_grid.csv'])
P=dict(id='E12_FINAL_TEST',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User explicitly requested evaluation of the two full-body methods on the final test set.',
    comparison='Primary full-body L1 and L2; matched title L1/L2 and price controls. Supplementary previously trained positive-temperature calibration of full-body L1, both stocks, not selected by holdout.',
    weights='Use original saved Jan-Aug2018 fitted models exactly. No refit on Sep-Oct; no C, vocabulary, feature, threshold, stock selection or temperature changes.',
    preparation='Rebuild same complete regular-session hours, original exclusions and six historical completed-hour prices; available=max(published,crawled), English/body/time/title/company filtering and earliest dedup identical to baseline; exact body/token preparation from E09. Prove parity with all2504 train/development rows before final prediction.',
    test_start='2018-11-01',test_end_exclusive='2019-02-02',expected_test_windows={'AAPL':364,'AMZN':365},
    threshold=.5,models=models,
    body_temperature={s:cal['stocks'][s]['paper_stem_body']['temperature'] for s in ['AAPL','AMZN']},
    training_up_prior={s:float(dev.loc[dev.symbol.eq(s)&dev.split.eq('train'),'label'].mean()) for s in ['AAPL','AMZN']},
    evaluation='BA primary, MCC/accuracy/F1/Brier, all stocks and all months including partial February, with/no-news slices. Mean across stocks explicit; no favorable stock/month filtering.',
    primary_contrasts=[['body_l1','title_l1'],['body_l2','title_l2']],secondary_contrasts=[['body_l2','body_l1'],['body_l1','price'],['body_l2','price']],
    uncertainty='2000 paired joint-trading-day bootstrap draws, seed583; 5-day moving-block sensitivity; percentile95 intervals conditional on frozen models, descriptive, not multiplicity-adjusted and not measuring full training/design uncertainty.',
    after_test='Preserve all scores. No further tuning from these results and no claim this exposed period remains an untouched holdout for later experiments. New inference of these fixed weights for reproducibility is not a new selection.',
    versions={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','nltk','joblib']},
    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},test_predictions_computed_at_registration=False)
(B/'protocol.json').write_text(json.dumps(P,ensure_ascii=False,indent=2));print('Comparison frozen before holdout preparation or prediction:',P['registered_utc']);print(json.dumps(models,indent=2))
