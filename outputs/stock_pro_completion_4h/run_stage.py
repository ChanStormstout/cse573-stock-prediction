"""Stage-isolated orchestration; frozen run.py fitting functions unchanged."""
import models as m
import importlib.util
spec=importlib.util.spec_from_file_location('completion_runner',m.Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
globals().update({k:v for k,v in vars(runner).items() if not k.startswith('_')})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['lexical','kernel','table','residual']);ap.add_argument('--limit',type=int);args=ap.parse_args();seal_check();db=initialize();d=db['d'];blocks=db['inp']['blocks']
 with (LOCAL/(args.stage+'.lock')).open('a') as lock,threadpool_limits(2):
  fcntl.flock(lock,fcntl.LOCK_EX)
  for b in blocks[:args.limit]:
   root=LOCAL/args.stage/b['name']
   if (root/'complete.json').exists():
    saved=json.loads((root/'complete.json').read_text());assert saved['seal']==sha(LOCAL/'seal.json')
    for p,h in saved['files'].items():assert sha(root/p)==h
    continue
   assert not root.exists(),'Incomplete block preserved: '+str(root);root.mkdir(parents=True);TIMES.clear();m.GEOMETRY.clear();m.MATRICES.clear()
   out=read(PRIVATE/'baseline'/b['name']/'outer_predictions_SEALED.csv');assert out.row_id.tolist()==d.iloc[b['test']].row_id.tolist();out['seed']=b['seed'];out['block']=b['name']
   if args.stage in ['lexical','kernel']:rs,ip=run_svm(args.stage,b,root,out)
   elif args.stage=='table':rs,ip=run_table(b,root,out)
   else:rs,ip=run_residual(b,root,out)
   out.to_csv(root/'predictions.csv',index=False,float_format='%.17g');pd.DataFrame(rs).to_csv(root/'selection.csv',index=False,float_format='%.17g');pd.DataFrame(ip).to_csv(root/'inner_predictions.csv',index=False,float_format='%.17g');dump(root/'training.json',TIMES)
   dump(root/'complete.json',dict(seal=sha(LOCAL/'seal.json'),files={p.name:sha(p) for p in root.iterdir() if p.is_file()}));print(args.stage,b['name'],'complete',len(TIMES),'fits',flush=True)
if __name__=='__main__':main()
