"""Separate daily trading task. Decisions at original cutoff, execution next open."""
from core import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import gymnasium as gym
from gymnasium import spaces

def raw_prices(stock):
    prefix={'AAPL':'APPLE','AMZN':'AMAZON'}[stock];p=W/f'raw/CHARTS/{prefix}5.csv'
    m=pd.read_csv(p,header=None,names=['date','time','open','high','low','close','activity']);m['ts']=pd.to_datetime(m.date+' '+m.time,format='%Y.%m.%d %H:%M',utc=True)
    if m.ts.duplicated().any():raise ValueError('Duplicate raw prices')
    return m.set_index('ts').sort_index()

def build_trading(stock):
    m=raw_prices(stock);cal=pd.read_csv(W/'audit/xnys_schedule.csv',index_col=0);cal.index=pd.to_datetime(cal.index).strftime('%Y-%m-%d');bars=[];days=[]
    for day,c in cal.iterrows():
        start,end=pd.to_datetime(c['open'],utc=True),pd.to_datetime(c['close'],utc=True)
        if start<m.index.min() or start>m.index.max():continue
        starts=pd.date_range(start,end-pd.Timedelta(hours=1),freq='h');valid=[]
        for t in starts:
            stamps=pd.date_range(t,periods=12,freq='5min');x=m.reindex(stamps)
            if x[['open','high','low','close']].isna().any().any():continue
            valid.append(dict(start_utc=t,end_utc=t+pd.Timedelta(hours=1),open=x.open.iloc[0],close=x.close.iloc[-1],high=x.high.max(),low=x.low.min(),day=day))
        bars.extend(valid);days.append(dict(symbol=stock,day=day,expected=len(starts),complete=len(valid),eligible=len(starts)==len(valid)))
    b=pd.DataFrame(bars);coverage=pd.DataFrame(days);validdays=set(coverage.loc[coverage.eligible,'day']);rows=[]
    for day,g in b.groupby('day'):
        if day not in validdays:continue
        daily=[]
        for i,h in enumerate(g.itertuples()):
            cutoff=h.start_utc-pd.Timedelta(minutes=5);hist=b[b.end_utc<=cutoff].tail(6)
            if len(hist)!=6:break
            row=dict(symbol=stock,day=day,start_utc=h.start_utc,end_utc=h.end_utc,cutoff_utc=cutoff,open=h.open,close=h.close,flat=h.close==h.open,label=int(h.close>h.open),split='train' if day<'2018-09-01' else 'validation' if day<'2018-11-01' else 'test')
            for k,x in enumerate(hist.iloc[::-1].itertuples(),1):row[f'return_{k}']=np.log(x.close/x.open);row[f'range_{k}']=(x.high-x.low)/x.open
            row.update(history_age_hours=(cutoff-hist.end_utc.iloc[-1]).total_seconds()/3600,return_mean=np.mean([row[f'return_{k}'] for k in range(1,7)]),return_std=np.std([row[f'return_{k}'] for k in range(1,7)]),ny_hour=h.start_utc.tz_convert('America/New_York').hour+.5)
            # Includes closing-to-next-opening five-minute-bar boundary changes.
            exitprice=g.open.iloc[i+1] if i+1<len(g) else h.close
            row['trade_return']=exitprice/h.open-1;row['time_fraction']=i/max(1,len(g)-1);daily.append(row)
        if len(daily)==len(g):rows.extend(daily)
    d=pd.DataFrame(rows);coverage['used_with_history']=coverage.day.isin(d.day);return d,coverage

def reward_step(ret,position,previous,cost,terminal):
    turnover=abs(position-previous);entry=cost*turnover;gross=1+position*ret
    if gross<=0:raise ValueError('Insolvency')
    # Fees charged as fraction of contemporaneous wealth at entry and liquidation.
    exitfee=cost*abs(position) if terminal else 0.
    factor=(1-entry)*gross*(1-exitfee)
    if factor<=0:raise ValueError('Costs exceed wealth')
    return np.log(factor),turnover+(abs(position) if terminal else 0),entry+(1-entry)*gross*exitfee

class TradingEnv(gym.Env):
    def __init__(self,d,scaler,cost=.001):
        self.d=d.reset_index(drop=True);self.scaler=scaler;self.cost=cost;self.days=list(self.d.groupby('day',sort=True).indices.values());self.action_space=spaces.Discrete(3);self.observation_space=spaces.Box(-np.inf,np.inf,shape=(len(PRICE)+3,),dtype=np.float32);self.pos=0;self.j=0
        self.x=scaler.transform(self.d[PRICE]);self.probs=self.d.lr_probability.to_numpy();self.r=self.d.trade_return.to_numpy();self.tm=self.d.time_fraction.to_numpy()
    def obs(self):
        i=self.ids[self.j];return np.r_[self.x[i],2*self.probs[i]-1,self.pos,self.tm[i]].astype(np.float32)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);idx=(options or {}).get('day_index',int(self.np_random.integers(len(self.days))));self.ids=self.days[idx];self.j=0;self.pos=0;return self.obs(),{}
    def step(self,action):
        i=self.ids[self.j];a=int(action)-1;last=self.j==len(self.ids)-1;reward,turnover,fee=reward_step(self.r[i],a,self.pos,self.cost,last);self.pos=0 if last else a;self.j+=1
        info=dict(row=int(i),action=a,turnover=turnover,fee_fraction=fee,end_position=self.pos,log_return=reward)
        return np.zeros(self.observation_space.shape,dtype=np.float32) if last else self.obs(),float(reward),last,False,info

def evaluate_policy(env,policy):
    rows=[]
    for j in range(len(env.days)):
        obs,_=env.reset(options={'day_index':j});done=False
        while not done:
            idx=env.ids[env.j];action=policy(obs,env,idx);obs,r,done,_,info=env.step(action);row=env.d.iloc[idx];rows.append(dict(day=row.day,start_utc=row.start_utc,split=row.split,**info))
        assert env.pos==0
    return pd.DataFrame(rows)

def stats(z):
    w=np.r_[1,np.exp(np.cumsum(z.log_return.to_numpy()))];peak=np.maximum.accumulate(w)
    return dict(steps=len(z),days=z.day.nunique(),net_return=w[-1]-1,max_drawdown=float(np.max(1-w/peak)),turnover=float(z.turnover.sum()),sum_fee_fractions=float(z.fee_fraction.sum()),cash_fraction=float(z.action.eq(0).mean()),terminal_positions_zero=bool(z.groupby('day').end_position.last().eq(0).all()))

def run(out):
    import torch,stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(1);allresults=[];summary=[];coverage=[];direction=[]
    dump(out/'protocol.json',dict(ppo=dict(hidden=[16],lr=.0003,n_steps=100,batch_size=50,n_epochs=5,gamma=.99,gae_lambda=.95,clip_range=.2,total_timesteps=20000,seeds=[573,574,575]),reason_rollout='100 divides exactly 20000; no extra steps',cost_bps=[5,10,20],reward='log((1-c*abs(a-prev))*(1+a*r)*(1-c*abs(a) if terminal else 1))',limitations='Simulated execution; no depth/impact/borrow; exposed historical replay',sb3=stable_baselines3.__version__,torch=torch.__version__))
    for stock in ['AAPL','AMZN']:
        d,cov=build_trading(stock);coverage.append(cov);d['lr_probability']=np.nan;folder=out/stock;folder.mkdir()
        for month in sorted(d.loc[d.day.ge('2018-03-01'),'start_utc'].dt.strftime('%Y-%m').unique()):
            a=d[(d.end_utc<pd.Timestamp(month+'-01',tz='UTC'))&~d.flat];b=d[d.start_utc.dt.strftime('%Y-%m').eq(month)];temporal(a,b);m=price(.1).fit(a,a.label);pr=m.predict_proba(b)[:,1];d.loc[b.index,'lr_probability']=pr;save_model(m,a,b,folder/'lr'/month,pr);direction.append(dict(symbol=stock,month=month,**metrics(b.loc[~b.flat,'label'],pr[~b.flat.to_numpy()])))
        d=d[d.day.ge('2018-03-01')].copy();assert d.lr_probability.notna().all();d.to_csv(folder/'samples.csv',index=False)
        train=d[d.split.eq('train')];scaler=StandardScaler().fit(train[PRICE]);joblib.dump(scaler,folder/'scaler.joblib')
        # Same state information in a one-step predictor; fit on training-period past-only LR scores.
        expected=Ridge(alpha=1).fit(np.c_[train[PRICE],train.lr_probability,train.time_fraction],train.trade_return);joblib.dump(expected,folder/'return_model.joblib')
        # Standardize price features before ridge to avoid units dominating penalty.
        expected=Ridge(alpha=1).fit(np.c_[scaler.transform(train[PRICE]),train.lr_probability,train.time_fraction],train.trade_return);joblib.dump(expected,folder/'return_model_scaled.joblib')
        policies={'cash':lambda o,e,i:1,'long':lambda o,e,i:2,'threshold':lambda o,e,i:2 if e.probs[i]>.55 else 0 if e.probs[i]<.45 else 1}
        def greedy(o,e,i):
            mu=expected.predict(np.r_[e.x[i],e.probs[i],e.tm[i]][None])[0];a=np.array([-1,0,1]);v=a*mu-e.cost*np.abs(a-e.pos)
            if e.j==len(e.ids)-1:v-=e.cost*np.abs(a)
            # Favor cash on ties.
            return int(np.flatnonzero(v==v.max())[0]) if np.ptp(v)>0 else 1
        policies['one_step']=greedy
        env=TradingEnv(train,scaler);check_env(env,warn=True)
        for seed in [573,574,575]:
            dst=folder/f'ppo_{seed}';dst.mkdir();agent=PPO('MlpPolicy',env,learning_rate=3e-4,n_steps=100,batch_size=50,n_epochs=5,policy_kwargs={'net_arch':dict(pi=[16],vf=[16])},seed=seed,device='cpu',verbose=0);agent.set_logger(configure(str(dst),['csv']));agent.learn(total_timesteps=20000);agent.save(dst/'model');assert agent.num_timesteps==20000
            reloaded=PPO.load(dst/'model',device='cpu');obs,_=env.reset(seed=seed);np.testing.assert_array_equal(agent.predict(obs,deterministic=True)[0],reloaded.predict(obs,deterministic=True)[0]);policies[f'PPO_{seed}']=lambda o,e,i,a=reloaded:int(a.predict(o,deterministic=True)[0]);print('trained PPO',stock,seed,flush=True)
        for cost in [.0005,.001,.002]:
            for name,policy in policies.items():
                e=TradingEnv(d,scaler,cost);z=evaluate_policy(e,policy);z['symbol']=stock;z['method']=name;z['cost_bps']=cost*10000;allresults.append(z)
                for split,s in z.groupby('split'):
                    summary.append(dict(symbol=stock,method=name,cost_bps=cost*10000,period=split,**stats(s)))
                for month,s in z.groupby(z.day.str[:7]):summary.append(dict(symbol=stock,method=name,cost_bps=cost*10000,period=month,**stats(s)))
    pd.concat(coverage).to_csv(out/'coverage.csv',index=False);pd.concat(allresults).to_csv(out/'transitions.csv',index=False);pd.DataFrame(summary).to_csv(out/'summary.csv',index=False);pd.DataFrame(direction).to_csv(out/'lr_direction.csv',index=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();stage(args.run.resolve(),'M09',run)
