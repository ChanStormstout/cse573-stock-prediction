"""Window-level low-rank adaptation; cached layers0:10 are frozen constants."""
import copy,math,random
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer,BertPooler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class LoRALinear(nn.Module):
    def __init__(self,base,rank=4,alpha=4):
        super().__init__();self.base=base;self.rank=rank;self.scale=alpha/rank
        self.A=nn.Parameter(torch.empty(rank,base.in_features));self.B=nn.Parameter(torch.zeros(base.out_features,rank));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        self.base.requires_grad_(False)
    def forward(self,x):return self.base(x)+(x@self.A.T@self.B.T)*self.scale


class Tail(nn.Module):
    def __init__(self,config,state,adapt=False):
        super().__init__();config=BertConfig.from_dict(config);config._attn_implementation='eager'
        self.layers=nn.ModuleList([BertLayer(config,layer_idx=i) for i in [10,11]]);self.pooler=BertPooler(config)
        self.load_state_dict(state);self.requires_grad_(False)
        if adapt:
            for layer in self.layers:
                sa=layer.attention.self
                sa.query=LoRALinear(sa.query);sa.value=LoRALinear(sa.value)
        self.eval()
    def forward(self,h,mask):
        ext=(1-mask[:,None,None,:].to(h.dtype))*torch.finfo(h.dtype).min
        for layer in self.layers:h=layer(h,attention_mask=ext)[0]
        return self.pooler(h)


def pool_bags(vectors,bags):
    valid=bags>=0;out=vectors[np.maximum(bags,0)].copy();out[~valid]=0
    return out.sum(1)/np.maximum(valid.sum(1,keepdims=True),1)


def fit_transform(pooled,numeric,has,ids):
    pca=PCA(n_components=16,svd_solver='full').fit(pooled[ids][has[ids]>0])
    z=pca.transform(pooled);z[has==0]=0;sc=StandardScaler().fit(np.column_stack([numeric,z])[ids])
    return dict(pca_mean=pca.mean_,pca_components=pca.components_,mean=sc.mean_,scale=sc.scale_,fitted_rows=np.asarray(ids),pca_n=int((has[ids]>0).sum()))


class WindowModel(nn.Module):
    def __init__(self,kind,config,state,transform,train_prior,seed,device):
        super().__init__();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self.kind=kind;self.device=device;self.tail=Tail(config,state,adapt=True) if kind=='lora' else None
        for name in ['pca_mean','pca_components','mean','scale']:self.register_buffer(name,torch.tensor(transform[name],dtype=torch.float32))
        self.head=nn.Linear(len(transform['mean']),1,bias=False);nn.init.zeros_(self.head.weight)
        self.stock_bias=nn.Parameter(torch.tensor(np.log(train_prior/(1-train_prior)),dtype=torch.float32))
        self.to(device);self.eval()
    def forward(self,numeric,pooled,stock,has):
        projected=(pooled-self.pca_mean)@self.pca_components.T;projected=projected*has[:,None]
        x=(torch.cat([numeric,projected],dim=1)-self.mean)/self.scale
        return self.head(x).squeeze(-1)+self.stock_bias[stock]


def encode_bags(model,cache,bags,device):
    valid=bags>=0;unique,inverse=np.unique(bags[valid],return_inverse=True)
    if not len(unique):return torch.zeros((len(bags),768),device=device)
    lengths=torch.tensor([len(cache[i]) for i in unique],device=device)
    h=pad_sequence([cache[i] for i in unique],batch_first=True).to(device=device,dtype=torch.float32)
    mask=torch.arange(h.shape[1],device=device)[None,:]<lengths[:,None]
    v=model.tail(h,mask);positions=torch.tensor(np.where(valid.ravel())[0],device=device)
    flat=torch.zeros((bags.size,768),device=device);flat=flat.index_copy(0,positions,v[torch.tensor(inverse,device=device)])
    return flat.reshape(len(bags),bags.shape[1],768).sum(1)/torch.tensor(np.maximum(valid.sum(1),1),device=device)[:,None]


def forward_rows(model,data,ids):
    dev=model.device;bag=data['bags'][ids]
    pooled=encode_bags(model,data['prefix'],bag,dev) if model.kind=='lora' else torch.tensor(data['pooled'][ids],dtype=torch.float32,device=dev)
    return model(torch.tensor(data['numeric'][ids],dtype=torch.float32,device=dev),pooled,torch.tensor(data['stock'][ids],device=dev),torch.tensor(data['has'][ids],dtype=torch.float32,device=dev))


def probabilities(model,data,ids,batch_size=16):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(ids),batch_size):out.extend(torch.sigmoid(forward_rows(model,data,ids[start:start+batch_size])).cpu().tolist())
    return np.asarray(out)


def fit(kind,data,train_ids,monitor_ids,seed,protocol,fixed_epochs=None):
    tr=protocol['training'];dev='mps' if torch.backends.mps.is_available() else 'cpu'
    transform=fit_transform(data['pooled'],data['numeric'],data['has'],train_ids)
    priors=np.asarray([data['y'][train_ids][data['stock'][train_ids]==s].mean() for s in [0,1]])
    net=WindowModel(kind,data['config'],data['tail_state'],transform,priors,seed,dev)
    groups=[dict(params=list(net.head.parameters())+[net.stock_bias],lr=tr['head_learning_rate'],weight_decay=tr['head_weight_decay'])]
    if kind=='lora':groups.append(dict(params=[p for p in net.tail.parameters() if p.requires_grad],lr=protocol['LoRA']['learning_rate'],weight_decay=protocol['LoRA']['weight_decay']))
    opt=torch.optim.AdamW(groups);rng=np.random.default_rng(seed);trace=[];best=float('inf');best_epoch=0;stale=0
    for epoch in range(1,(fixed_epochs or tr['max_epochs'])+1):
        net.eval();total=0.;order=rng.permutation(train_ids)
        for start in range(0,len(order),tr['batch_size']):
            ids=order[start:start+tr['batch_size']];opt.zero_grad();logits=forward_rows(net,data,ids);y=torch.tensor(data['y'][ids],dtype=torch.float32,device=dev)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,y);assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.);opt.step();total+=float(loss.detach().cpu())*len(ids)
        row=dict(epoch=epoch,training_bce=total/len(train_ids))
        if monitor_ids is not None:
            p=probabilities(net,data,monitor_ids,tr['batch_size']);p=np.clip(p,1e-7,1-1e-7);y=data['y'][monitor_ids];bce=float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)));row['inner_bce']=bce
            if bce<best-tr['min_delta']:best=bce;best_epoch=epoch;stale=0
            else:stale+=1
        trace.append(row)
        if monitor_ids is not None and stale>=tr['patience']:break
    chosen=fixed_epochs or best_epoch;assert chosen>0
    return net,transform,chosen,trace
