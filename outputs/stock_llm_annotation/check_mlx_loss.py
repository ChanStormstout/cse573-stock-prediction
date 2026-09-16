"""Small exact-gradient checks for the optional event decision loss."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stock_llm_4h'))
import mlx.core as mx
import mlx.nn as nn
from train_adapter import loss
class Toy(nn.Module):
 def __init__(self):super().__init__();self.weight=mx.zeros((7,12))
 def __call__(self,x):return self.weight[None,:x.shape[1],:]
def main():
 model=Toy();tokens=[1]*8
 f=nn.value_and_grad(model,loss)
 old,g=f(model,tokens,4)
 new,h=f(model,tokens,4,6,2.)
 mx.eval(old,new,g,h)
 assert abs(float(old)-float(mx.log(mx.array(12.))))<1e-5
 assert abs(float(new)-3*float(old))<1e-5
 assert bool(mx.all(g['weight'][:3]==0)) and bool(mx.all(h['weight'][:3]==0))
 assert bool(mx.allclose(h['weight'][5],9*g['weight'][5]))
 assert bool(mx.allclose(h['weight'][4],g['weight'][4]))
 print(json.dumps({'completion_only_mask':True,'auxiliary_only_at_branch':True,'default_loss_unchanged':True,'finite_nonzero_gradient':True}))
if __name__=='__main__':main()
