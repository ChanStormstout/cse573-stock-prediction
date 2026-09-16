"""Assemble source-backed delivery metadata; does not train or alter model runs."""
from pathlib import Path
import json, hashlib, re
import pandas as pd
B=Path(__file__).resolve().parent
ROOT=B.parents[1]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
rows=[]
for version,folder,names in [('primary','v1',['title','price','old_integrated','gate','gate_cal','body']),('zero_bias','zero_bias_v2',['body','static','gate','gate_cal','online_weights','selected_system'])]:
    d=pd.read_csv(B/'results'/folder/'metrics.csv');d=d[(d.phase=='frozen')&(d.period=='test')&d.method.isin(names)]
    for r in d[['symbol','method','n','BA','MCC','Brier']].to_dict('records'):rows.append({'version':version,**r})
pd.DataFrame(rows).to_csv(B/'COMPARISON.csv',index=False)
coverage=pd.read_csv(B/'coverage/v3/coverage_summary.csv')
quality=json.loads((B/'coverage/v3/quality_status.json').read_text())
training=[]
for run in ['v3','zero_bias_v1']:
    s=json.loads((B/'runs'/run/'status.json').read_text());v=json.loads((B/'runs'/run/'verification.json').read_text())
    training.append({'run':run,'actual_fits':s['actual_fits'],'raw_labels_checked':v['raw_4h_labels_checked'],'frozen_predictions_checked':v['frozen_predictions_checked'],'reload_error':v['maximum_reload_error'],'verification':v['status']})
write(B/'STATUS.json',{'status':'CORE_EXPERIMENTS_AND_DELIVERY_COMPLETE_CONDITIONAL_INPUT_PENDING','horizon':'4h','rows':1607,'exposed_history':True,'selected_method':'title','stable_two_stock_improvement':False,'stages':{'N00':'COMPLETE','N01':'COMPLETE_TWO_TARGETED_VERSIONS','N02':'COMPLETE_GUARDED_ROUTER','N03':'COMPLETE','N04':'AUDIT_EXTRACTION_AND_REVIEW_MATERIALS_COMPLETE_INDEPENDENT_REVIEW_PENDING','N05':'COMPLETE_TWO_VARIANTS_MATCHED_INFORMATION_CONTROLS','N06':'NOT_ENTERED_CONTINUATION_AND_QUALITY_GATES_NOT_MET'},'training':training,'regression_tests_passed':27,'quality_status':quality,'new_qualified_holdout':False,'manual_case_review':{'generated_cards':64,'reviewed_windows':16,'article_excerpts':23,'independent_gold':False},'main_report':'REPORT.md','demo':'http://127.0.0.1:8769'})
old=ROOT/'outputs/stock_integrated_4h/next_plan/STATUS.json'
s=json.loads(old.read_text());s.setdefault('status_at_plan_creation',s['status']);s.setdefault('training_executed_in_plan_creation_turn',s.pop('training_executed_this_turn',False));s['status']='IMPLEMENTED_CORE_EXPERIMENTS_COMPLETE_CONDITIONAL_QUALITY_PENDING';s['execution_report']='../../stock_adaptive_4h/REPORT.md';s['execution_status']='../../stock_adaptive_4h/STATUS.json';write(old,s)
write(B/'results/zero_bias_v1/status.json',{'status':'REPORT_PARTIAL_PATH_ERROR','cause':'Relative path display failed after tables and cases were written. No model changes.','completed_report':'../zero_bias_v2/REPORT.md'})
items=[{'id':'four-hour-comparison','title':'两版完整四小时系统与 baseline','queries':[{'id':'saved-results','source':{'label':'四小时模型回放结果','files':[{'label':'COMPARISON.csv'},{'label':'metrics.csv'},{'label':'system_selection.json'}],'caveats':['所有时期已用于设计；这是探索性历史回放。两版的预设规则均保留标题 baseline。AAPL正文54.09%是诊断结果，未被事后升格为最终方法。']},'reportingPeriod':'2018年11月至2019年2月初','columns':['version','symbol','method','n','BA','MCC','Brier'],'rows':rows,'methods':[{'language':'python','code':"d = pd.read_csv('metrics.csv')\ncomparison = d[(d.phase == 'frozen') & (d.period == 'test')]\n# BA: average of up and down recall; Brier: mean squared probability error."}]}]},
{'id':'coverage','title':'AMZN潜在覆盖与独立质量状态','queries':[{'id':'coverage-audit','source':{'label':'原始正文候选审计','files':[{'label':'coverage_summary.csv'},{'label':'independent_check.csv'},{'label':'quality_status.json'}],'caveats':['仅规则候选，未加入预测；独立复核0/65，accepted_types为空。抽取正确不等于预测有效。']},'reportingPeriod':'原四小时训练、开发及后续时期','columns':coverage.columns.tolist(),'rows':coverage[coverage.symbol=='AMZN'].to_dict('records'),'methods':[{'language':'python','code':"d = pd.read_csv('coverage_summary.csv')\nresult = d[d.symbol == 'AMZN']\n# Candidate coverage is audited separately from predictor article membership."}]}]},
{'id':'training','title':'真实训练与保存模型核验','queries':[{'id':'model-verification','source':{'label':'主运行日志与独立重载检查','files':[{'label':'status.json'},{'label':'fits.json'},{'label':'verification.json'},{'label':'tests.txt'}],'caveats':['FinBERT编码器冻结；464次主拟合包括传统模型、修正器、校准与选择器，并非464次大模型微调。工程核验不等于泛化有效。']},'columns':['run','actual_fits','raw_labels_checked','frozen_predictions_checked','reload_error','verification'],'rows':training}]}]
write(B/'sources_receipt.json',{'schemaVersion':1,'items':items})
# Delivery documents only: check local links without pretending to certify external URLs.
missing=[]
for p in B.glob('*.md'):
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if '://' not in target and not (p.parent/target).exists():missing.append({'file':p.name,'target':target})
assert not missing,missing
print(json.dumps({'source_rows':len(rows),'training_fits':sum(x['actual_fits'] for x in training),'local_document_links':'PASS','quality':quality['status']},ensure_ascii=False))
