"""Aggregate fixed comparisons and build a deterministic private case panel."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter
from contracts import ROOT, messages, ORIGINAL
sys.path.insert(0,str(ROOT/'outputs/stock_llm_4h'))
from common import rows,dump,file_sha,evaluate,signature,check_panel,validate

B=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--rules-v2',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    data=ROOT/'work/stock-data/annotation/student_v3';check_panel(data)
    inputs={r['id']:r for r in rows(data/'inputs.jsonl')};labels=rows(data/'labels.jsonl');gold={r['id']:r for r in labels}
    manifest=json.loads((a.run/'manifest.json').read_text());run_summary=json.loads((a.run/'summary.json').read_text())
    assert run_summary['completed'] and not manifest['training'] and not manifest['smoke_only']
    assert manifest['input_sha256']==file_sha(data/'inputs.jsonl') and manifest['labels_sha256']==file_sha(data/'labels.jsonl')
    old_manifest=json.loads((ROOT/'work/stock-data/annotation/runs/frozen_v3/manifest.json').read_text())
    for key in ['input_sha256','labels_sha256','temperature','thinking']:assert old_manifest[key]==manifest[key]
    assert old_manifest['max_tokens']==manifest['max_output_per_call']
    assert old_manifest['system_sha256']==hashlib.sha256(ORIGINAL.encode()).hexdigest()
    for name,value in manifest['code'].items():assert file_sha(a.run/'code'/name)==value
    expected={r['id'] for r in labels if r['split'] in {'valid','check'}}
    models={'old_frozen':rows(ROOT/'work/stock-data/annotation/runs/frozen_v3/predictions.jsonl'),
            'old_tuned':rows(ROOT/'work/stock-data/annotation/runs/tuned_v3/predictions.jsonl'),
            'rules':rows(ROOT/'work/stock-data/annotation/runs/rules_v3/predictions.jsonl')}
    table=[];quality={}
    for variant in ['original','prompt','staged']:
        ps=rows(a.run/variant/'predictions.jsonl');assert len(ps)==len(expected) and {r['id'] for r in ps}==expected
        for r in ps:
            if variant=='original': assert r['calls'][0]['messages']==messages(inputs[r['id']],'original')
            assert all(c['prompt_tokens']+512<=4096 for c in r['calls'])
        models[variant]=ps
        quality[variant]={'raw_json_parse_failures':sum(c['parsed'] is None for r in ps for c in r['calls']),
                          'max_prompt_tokens':max(c['prompt_tokens'] for r in ps for c in r['calls']),
                          'all_original_inputs_fit_old_2048_budget':all(c['prompt_tokens']+512<=2048 for r in ps for c in r['calls']) if variant=='original' else None,
                          'final_invalid':sum(not r['valid'] for r in ps),
                          'component_errors':dict(Counter(e for r in ps for e in r['component_errors'])),
                          'rule_rejections':dict(Counter(x['reason'] for r in ps for x in r['rejected'])),
                          'gate_states':dict(Counter(r['calls'][0]['parsed'].get('status','invalid') if isinstance(r['calls'][0]['parsed'],dict) else 'invalid' for r in ps)) if variant=='staged' else {},
                          **run_summary['variants'][variant]}
    if a.rules_v2:
        repair_manifest=json.loads((a.rules_v2/'manifest.json').read_text())
        assert repair_manifest['source_predictions_sha256']==file_sha(a.run/'staged/predictions.jsonl')
        repaired=rows(a.rules_v2/'predictions.jsonl');original={r['id']:r for r in models['staged']}
        assert len(repaired)==len(expected) and {r['id'] for r in repaired}==expected
        for r in repaired:assert r['calls']==original[r['id']]['calls']
        models['staged_v2']=repaired
        repair_manifest['repaired_by_phase']=dict(Counter(gold[r['id']]['split'] for r in repaired if r['repairs']))
        dump(a.out/'rule_repair.json',repair_manifest)
    for name,ps in models.items():
        for phase in ['valid','check']:
            for cohort in (['all'] if phase=='valid' else ['all','original','enriched']):
                for symbol in ['ALL','AAPL','AMZN']:
                    sub=[r for r in labels if r['split']==phase and (symbol=='ALL' or r['symbol']==symbol) and (cohort=='all' or r['id'].startswith('N' if cohort=='original' else 'Q'))]
                    present={r['id'] for r in ps}
                    if not sub or not {r['id'] for r in sub}<=present:continue
                    m=evaluate(ps,sub)[phase];den=2*m['fact_tp']+m['fact_fp']+m['fact_fn'];m['f1']=2*m['fact_tp']/den if den else None
                    table.append({'model':name,'phase':phase,'cohort':cohort,'symbol':symbol,**m})
    with (a.out/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    # Post-hoc diagnostic only: do strict article rejections hide some correct
    # events? This neither changes saved predictions nor the registered metric.
    eventwise=[]
    for name,ps in models.items():
        salvaged=[]
        for pred in ps:
            if pred['id'] not in inputs:continue
            obj=pred.get('parsed');events=[]
            if isinstance(obj,dict) and isinstance(obj.get('events'),list):
                for event in obj['events']:
                    try:ok,_=validate({'events':[event]},inputs[pred['id']])
                    except (TypeError,ValueError,AttributeError):ok=False
                    if ok:events.append(event)
            salvaged.append({'id':pred['id'],'parsed':{'events':events},'valid':True})
        for phase,m in evaluate(salvaged,labels).items():
            if phase in {'valid','check'} and m['n']:
                eventwise.append({'model':name,'phase':phase,'n':m['n'],'tp':m['fact_tp'],'fp':m['fact_fp'],'fn':m['fact_fn']})
    dump(a.out/'eventwise_diagnostic.json',{'status':'POST_HOC_DIAGNOSTIC_NOT_REGISTERED_PRIMARY_METRIC',
         'meaning':'Well-parsed JSON only; validate each event separately, bypass whole-article rejection and max-six limit. No raw JSON repair. Does not alter inference or promote a new method.', 'results':eventwise})
    by_model={k:{r['id']:r for r in ps} for k,ps in models.items()}
    pairs=[('old_frozen','original'),('original','prompt'),('prompt','staged')];panel=[];cells=[]
    def exact(pred,g):
        return bool(pred['valid'] and {signature(e) for e in pred['parsed']['events']}=={signature(e) for e in g['answer']['events']})
    for first,second in pairs:
        name=first+'__'+second
        for symbol in ['AAPL','AMZN']:
            for state in ['fixed','regressed','both_wrong','both_correct']:
                ids=[]
                for g in labels:
                    if g['split']!='check' or g['symbol']!=symbol:continue
                    x,y=exact(by_model[first][g['id']],g),exact(by_model[second][g['id']],g)
                    group='both_correct' if x and y else 'fixed' if y else 'regressed' if x else 'both_wrong'
                    if group==state:ids.append(g['id'])
                selected=min(ids,key=lambda x:hashlib.sha256((x+name).encode()).hexdigest()) if ids else None
                cells.append(dict(comparison=name,symbol=symbol,state=state,count=len(ids),selected=selected))
                if selected:panel.append(selected)
    for category in ['rule_rejection','gate_lost_positive']:
        ids=[]
        for r in models['staged']:
            if gold[r['id']]['split']!='check':continue
            gate=r['calls'][0]['parsed']
            condition=bool(r['rejected']) if category=='rule_rejection' else bool(gold[r['id']]['answer']['events'] and isinstance(gate,dict) and gate.get('status')!='events')
            if condition:ids.append(r['id'])
        selected=sorted(ids,key=lambda x:hashlib.sha256(x.encode()).hexdigest())[:5];panel+=selected
        cells.append(dict(category=category,count=len(ids),selected=selected))
    if a.rules_v2:
        ids=[r['id'] for r in models['staged_v2'] if r['repairs']]
        panel+=ids;cells.append({'category':'all_rule_v2_repairs','selected':ids})
    with (a.out/'case_panel.jsonl').open('w') as f:
        for key in sorted(set(panel)):
            f.write(json.dumps({'id':key,'input':inputs[key],'reference':gold[key],
                                'models':{name:ps[key] for name,ps in by_model.items() if key in ps}},ensure_ascii=False)+'\n')
    dump(a.out/'case_selection.json',cells)
    dump(a.out/'quality.json',quality)
    dump(a.out/'verification.json',{'all_expected_inputs':len(expected),'same_original_message_content':True,
                                   'cached_reference_fingerprints_and_decoding_match':True,
                                   'analysis_sha256':file_sha(__file__),'case_protocol_sha256':file_sha(B/'CASE_PROTOCOL.md'),
                                   'no_truncation':True,'snapshots_match':True,'new_training':False,
                                   'independent_human_review':False,'fresh_holdout':False,'case_articles':len(set(panel))})
    print(json.dumps([r for r in table if r['cohort']=='all' and r['symbol']=='ALL'],indent=2))

if __name__=='__main__':main()
