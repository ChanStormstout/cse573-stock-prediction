from pathlib import Path
import ast,hashlib,json
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;ROOT=B.parents[2];A=B/'analysis_v1'
checks={}
for p in B.glob('*.py'):ast.parse(p.read_text())
checks['python_syntax']=True
for path,h in json.loads((A/'sources.json').read_text()).items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h
checks['original_inputs_predictions_and_protocol_unchanged']=True
seal=json.loads((A/'content_notes_seal.json').read_text());assert hashlib.sha256((A/'content_notes_before_reveal.json').read_bytes()).hexdigest()==seal['sha256']
checks['pre_outcome_notes_unchanged']=True
p=pd.read_pickle(B.parent/'runs/zero_bias_v1/predictions.pkl');p=p[(p.symbol=='AMZN')&(p.phase=='frozen')]
assert len(p)==305;np.testing.assert_allclose(p.gate,p.price,rtol=0,atol=1e-15);np.testing.assert_array_equal(p.gate>=.5,p.price>=.5)
checks['all_305_gate_equal_price_within_float_roundoff']=True
checks['gate_price_max_probability_difference']=float(abs(p.gate-p.price).max())
m=pd.read_csv(A/'metrics.csv');dc=pd.read_csv(A/'paired_decomposition.csv')
for name,first,second in [('body_vs_title','old_body','title'),('body_vs_old_mix','old_body','old_integrated'),('title_vs_new_gate','title','gate'),('old_mix_vs_title','old_integrated','title')]:
    for period in ['validation','test']:
        h=m[(m.period==period)&(m.slice=='all')].set_index('model')
        actual=dc[(dc.comparison==name)&(dc.period==period)].BA_contribution.sum()
        assert abs(actual-(h.loc[first,'BA']-h.loc[second,'BA']))<1e-12
checks['exact_additive_BA_reconciliation']=True
panel=pd.read_csv(A/'case_panel_with_outcomes.csv');content=json.loads((A/'content_before_outcome.json').read_text())
assert len(panel)==24 and panel.key.nunique()==24 and panel.day.nunique()==24
assert len(content)==24 and sum(len(c['articles']) for c in content)==20
assert not any(set(c)&{'label','target_return','price','old_body'} for c in content)
checks['24_unique_dates_20_excerpts_content_copy_hides_outcomes']=True
attr=json.loads((B/'addendum_v2/case_attributions.json').read_text());assert len(attr)==24
np.testing.assert_allclose([r['probability'] for r in attr],panel.old_body,rtol=0,atol=1e-12)
assert all(r['text_logit']==0 for r in attr if r['news_count']==0)
checks['24_saved_model_contributions_reproduce_predictions']=True
choices=json.loads((B/'addendum_v2/choices_before_evaluation.json').read_text());assert choices=={'AAPL':'body','AMZN':'price'}
assert choices==json.loads((B/'addendum_v1/choices_before_evaluation.json').read_text())
checks['single_branch_choices_unchanged_across_loading_fix']=True
receipt=json.loads((B/'sources_receipt.json').read_text());assert len(receipt['items'])==3
assert '/Users/' not in json.dumps(receipt)
checks['sources_receipt_safe_labels']=True
for name in ['REPORT.md','MECHANISM_STATUS.md','CASE_INTERPRETATIONS.md','RECENT_PAPERS.md']:
    assert (B/name).stat().st_size>1000
checks['all_four_reports_present']=True
report={'passed':True,'checks':checks,'new_training':False,'note':'Verifies diagnostic calculations and provenance, not predictive generalization.'}
(B/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
files=[p for p in B.rglob('*') if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts]
(B/'manifest.json').write_text(json.dumps({str(p.relative_to(B)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print(json.dumps(report,indent=2))
