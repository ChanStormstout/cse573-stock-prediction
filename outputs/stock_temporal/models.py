"""Small controlled predictors. No model reads outcomes as input."""
import random
import numpy as np,torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.preprocessing import StandardScaler
class TemporalNet(nn.Module):
 def __init__(self,kind,features=24,steps=6,hidden=8):
  super().__init__();self.kind=kind
  if kind=='gru':self.encoder=nn.GRU(features,hidden,batch_first=True);self.head=nn.Linear(hidden,1)
  elif kind=='mlp':
   gru_params=3*(features*hidden+hidden*hidden+2*hidden)+hidden+1
   width=max(1,round((gru_params-1)/(features*steps+2)))
   self.encoder=nn.Sequential(nn.Flatten(),nn.Linear(features*steps,width),nn.Tanh());self.head=nn.Linear(width,1)
  else:raise ValueError(kind)
 def forward(self,x,lengths):
  if self.kind=='gru':_,h=self.encoder(pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False));h=h[-1]
  else:h=self.encoder(x)
  return self.head(h).squeeze(-1)
def fit_scaler(x):return StandardScaler().fit(x[:,:,:-1][x[:,:,-1]>0])
def transform(x,scaler):
 mask=x[:,:,-1]>0;z=np.zeros_like(x,dtype=np.float32);z[:,:,:-1][mask]=scaler.transform(x[:,:,:-1][mask]);z[:,:,-1]=mask;return z
def predict(model,x,lengths):
 model.eval()
 with torch.inference_mode():return torch.sigmoid(model(torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(lengths,dtype=torch.long))).numpy().astype(float)
def train_net(kind,x,y,lengths,seed,wd,config):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 net=TemporalNet(kind,x.shape[-1],x.shape[1],config['GRU_hidden']);opt=torch.optim.AdamW(net.parameters(),lr=config['learning_rate'],weight_decay=wd)
 X=torch.as_tensor(x,dtype=torch.float32);Y=torch.as_tensor(y,dtype=torch.float32);L=torch.as_tensor(lengths,dtype=torch.long);trace=[]
 for epoch in range(config['epochs']):
  net.train();total=0
  for ids in torch.randperm(len(X)).split(config['batch_size']):
   opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(net(X[ids],L[ids]),Y[ids]);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),config['gradient_clip']);opt.step();total+=float(loss.detach())*len(ids)
  if epoch==0 or (epoch+1)%10==0:trace.append({'epoch':epoch+1,'training_bce':total/len(X)})
 return net,trace
