"""Unified supported public CLI. Config values outside registered protocol fail closed."""
import argparse
from core import *
def validate_config(c):
    fixed={'stocks':['AAPL','AMZN'],'selection_months':['2018-06','2018-07','2018-08'],'C':[.01,.1,1],'seeds':[573,574,575],'capacity_k':[25,100,500],'learning_curve_fractions':[.25,.5,1],'semantic_dimensions':[8,16,32],'fusion_weights':[0,.25,.5,.75,1]}
    for k,v in fixed.items():
        if c.get(k)!=v:raise ValueError('Unregistered config change requires new protocol implementation: '+k)
    if c['RL']['cost_bps']!=10 or c['RL']['sensitivity_bps']!=[5,20]:raise ValueError('Unregistered cost parameters')
    if c['RL']['steps']!=20000 or c['RL']['hidden']!=[16] or c['RL']['learning_rate']!=.0003:raise ValueError('Unregistered RL parameters')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate','report','verify']);p.add_argument('--config',type=Path,default=B/'configs/main.json');p.add_argument('--run',type=Path,required=True);p.add_argument('--mechanism',choices=['M02','M03','M04','M05','M06','M07','M08','M09','all'],default='all');a=p.parse_args();r=a.run.resolve();c=json.loads(a.config.read_text());validate_config(c)
    if a.stage=='prepare':
        prepare(r,c)
        raw=list((W/'raw/news').glob('*.zip'))+list((W/'raw/CHARTS').glob('*5.csv'))+[W/'audit/news_index.pkl',W/'audit/xnys_schedule.csv',O/'stock_event_facts/results/bodies.json',O/'stock_event_facts_v2/extract.py',O/'stock_event_facts/extract.py']
        dump(r/'prepared_raw_sources.json',{str(p):sha(p) for p in raw})
        return
    registered=json.loads((r/'config.json').read_text())
    if c!=registered:raise ValueError('Config mismatch with prepared run')
    d=load(r)
    for manifest in ['prepared_raw_sources.json','sealed_inputs.json']:
        if (r/manifest).exists():verify(json.loads((r/manifest).read_text()))
    if a.stage=='train':
        from diagnostics import capacity,updates
        from news_diagnostics import timing,coverage
        from events import run as event_run
        from correction import run as correction_run
        from semantic import run as semantic_run
        from trading import run as rl_run
        actions={'M02':lambda o:capacity(d,o),'M03':lambda o:updates(d,o),'M04':lambda o:timing(d,o),'M05':lambda o:coverage(d,o),'M06':lambda o:event_run(d,o),'M07':lambda o:correction_run(o,r),'M08':lambda o:semantic_run(d,o),'M09':rl_run}
        for name in actions if a.mechanism=='all' else [a.mechanism]:
            stage(r,name,actions[name])
        if (r/'M02/cv.csv').exists() and (r/'M08/selected_oof.csv').exists():
            from gate import assess
            assess(r)
    elif a.stage=='verify':
        from verification import run_checks
        run_checks(r)
    else:
        from reporting import evaluate,report
        (evaluate if a.stage=='evaluate' else report)(r)
if __name__=='__main__':main()
