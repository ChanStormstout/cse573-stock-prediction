import argparse
from core import *
p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate','report']);p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--mechanism',default='M02');args=p.parse_args();run=args.run.resolve();config=json.loads(args.config.read_text())
if args.stage=='prepare':prepare(run,config)
elif args.stage=='train':
    d=load(run)
    if args.mechanism=='M02':
        from diagnostics import capacity
        stage(run,'M02',lambda out:capacity(d,out))
    elif args.mechanism=='M03':
        from diagnostics import updates
        stage(run,'M03',lambda out:updates(d,out))
    else:raise ValueError('Unknown mechanism')
else:
    from reporting import evaluate,report
    (evaluate if args.stage=='evaluate' else report)(run)
