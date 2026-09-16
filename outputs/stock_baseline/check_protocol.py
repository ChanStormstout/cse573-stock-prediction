"""Check time boundaries, split integrity and train-only feature fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from run_baseline import window_indices

base=Path(__file__).resolve().parent/'results'
cutoff=pd.Timestamp('2018-09-04 14:25',tz='UTC')
times=pd.DatetimeIndex([cutoff-pd.Timedelta(hours=4),cutoff-pd.Timedelta(seconds=1),cutoff,cutoff+pd.Timedelta(seconds=1)]).astype('int64').to_numpy()
assert window_indices(times,cutoff)==(1,3), 'Future news or wrong lower boundary'
for symbol in ['AAPL','AMZN']:
    df=pd.read_csv(base/symbol/'samples.csv')
    cutoff_series=pd.to_datetime(df.cutoff_utc,utc=True)
    for cutoff_value,ends in zip(cutoff_series,df.history_ends):
        assert all(pd.Timestamp(e)<=cutoff_value for e in ends.split('|'))
    assert not df.start_utc.duplicated().any()
    assert np.all(pd.to_datetime(df.start_utc,utc=True)-cutoff_series==pd.Timedelta(minutes=5))
    assert df.target_return.ne(0).all()
    model=joblib.load(base/symbol/'combined_model.joblib')
    tr=df[df.split.eq('train')].copy();va=df[df.split.eq('validation')].copy()
    numeric=model.named_steps['features'].transformers_[0][2]
    scaler=model.named_steps['features'].named_transformers_['price']
    np.testing.assert_allclose(scaler.mean_,tr[numeric].mean().to_numpy())
    # Mutating every outcome must have zero effect on inference features/probabilities.
    before=model.predict_proba(va.fillna({'text':''}))
    va['label']=1-va.label;va['target_return']=999
    np.testing.assert_array_equal(before,model.predict_proba(va.fillna({'text':''})))
    vectorizer=model.named_steps['features'].named_transformers_['text']
    analyzer=vectorizer.build_analyzer()
    train_tokens=set(token for title in tr.text.fillna('') for token in analyzer(title))
    assert set(vectorizer.vocabulary_).issubset(train_tokens)
assert json.loads((base/'results.json').read_text())['protocol']['test_evaluated'] is False
print('PASS: news time boundaries; historical availability; unique samples; train-only scaling/vocabulary; outcome isolation. No test predictions computed.')
