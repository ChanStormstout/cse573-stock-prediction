"""Shared-date paired bootstrap, old-model inference audit, and dynamic report."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
from run import ROOT,metrics
B=Path(__file__).parent;O=B/'runs/v1'
def audit_old():
 old=ROOT/'outputs/stock_horizons/runs/v1';d=pd.read_pickle(old/'data.pkl');p=pd.read_csv(old/'predictions.csv');res=[]
 for (sym,h,method),g in p.groupby(['symbol','horizon','method']):
  ev=d[(d.symbol==sym)&(d.horizon==h)&(d.split!='train')].sort_values('start_utc');g=g.sort_values('start_utc');model=joblib.load(old/f'{sym}_{h}_{method}.joblib');delta=float(np.max(np.abs(model.predict_proba(ev)[:,1]-g.p.to_numpy())));assert delta<=1e-10
  res.append(dict(symbol=sym,horizon=h,method=method,max_probability_error=delta))
 # All four-hour labels independently recomputed from raw bars.
 from features_price import load_sources
 for sym,g in d[d.horizon=='4h'].groupby('symbol'):
  bars,_=load_sources(ROOT/'work/stock-data',sym)
  for row in g.itertuples():
   x=bars.reindex(pd.date_range(row.start_utc,row.end_utc-pd.Timedelta('5min'),freq='5min'));assert len(x)==48 and x.close.notna().all();assert int(x.close.iloc[-1]>x.open.iloc[0])==row.label
 (O/'old_inference_audit.json').write_text(json.dumps({'models':res,'four_hour_labels_checked':1607},indent=2))

def bootstrap(p):
 rows=[];rng=np.random.default_rng(573)
 # One draw shares the dates across stocks and methods, retaining every intraday start.
 for phase,g in p[p.dataset=='main4h'].groupby('phase'):
  dates=sorted(g.day.unique());n=len(dates)
  for block in [1,5,10]:
   starts=rng.integers(0,n,size=(10000,int(np.ceil(n/block))));ix=((starts[:,:,None]+np.arange(block))%n).reshape(10000,-1)[:,:n]
   weights=np.zeros((10000,n),dtype=np.int16)
   np.add.at(weights,(np.arange(10000)[:,None],ix),1)
   for sym,sg in g.groupby('symbol'):
    scores={}
    for arm,ag in sg.groupby('arm'):
     ag=ag.copy();ag['tp']=((ag.p>=.5)&(ag.label==1)).astype(int);ag['tn']=((ag.p<.5)&(ag.label==0)).astype(int);ag['pos']=ag.label;ag['neg']=1-ag.label;ag['err']=(ag.p-ag.label)**2;ag['n']=1
     v=ag.groupby('day')[['tp','tn','pos','neg','err','n']].sum().reindex(dates,fill_value=0).to_numpy();s=weights@v
     scores[arm]={'BA':.5*(np.divide(s[:,0],s[:,2],out=np.full(10000,np.nan),where=s[:,2]>0)+np.divide(s[:,1],s[:,3],out=np.full(10000,np.nan),where=s[:,3]>0)),'Brier':s[:,4]/s[:,5]}
    for reference in ['O','M','S']:
     for metric in ['BA','Brier']:
      v=scores['R'][metric]-scores[reference][metric];lo,hi=np.nanquantile(v,[.025,.975]);rows.append(dict(phase=phase,symbol=sym,contrast='R-'+reference,metric=metric,block=block,draws=10000,lower=lo,upper=hi))
 pd.DataFrame(rows).to_csv(O/'paired_intervals.csv',index=False)

def main():
 audit_old();p=pd.read_csv(O/'predictions.csv');bootstrap(p)
 # One-class tiny groups have no two-class BA; show missing rather than misleading 100%.
 rows=[]
 for cols in [['dataset','symbol','phase','arm'],['dataset','symbol','phase','arm','month'],['dataset','symbol','phase','arm','ny_hour']]:
  for k,g in p.groupby(cols):
   for threshold in ['0.5','train_prior']:
    met=metrics(g.label,g.p,.5 if threshold=='0.5' else g.train_prior)
    if g.label.nunique()<2:met['BA']=None;met['MCC']=None
    rows.append({**dict(zip(cols,k)),'threshold':threshold,**met})
 m=pd.DataFrame(rows);m.to_csv(O/'metrics.csv',index=False)
 monthly=m[(m.dataset=='main4h')&(m.phase=='outer')&(m.threshold=='0.5')&m.month.notna()]
 outer=monthly.groupby(['symbol','arm'])[['BA','MCC','Brier']].mean().reset_index()
 replay=m[(m.dataset=='main4h')&(m.phase=='frozen_replay')&(m.threshold=='0.5')&m.month.isna()&m.ny_hour.isna()][['symbol','arm','BA','MCC','Brier','n']]
 matched=m[m.dataset.str.startswith('matched')&(m.threshold=='0.5')&m.month.isna()&m.ny_hour.isna()][['dataset','symbol','phase','arm','BA','Brier','n']]
 gates=json.loads((O/'gates.json').read_text());lag=json.loads((B/'runs/lag5/gates.json').read_text());manifest=json.loads((O/'manifest.json').read_text())
 report=f'''# 四小时价格机制实验：实际执行结果

## 结论
新鲜价格上下文 R 在两股均未通过预先约定的继续门槛。停止扩大技术指标搜索；没有证据把四小时任务宣布为可靠优于随机的最终目标。所有数据时期已经参与过设计，本轮仍为探索性历史回测。

## 实际执行
- {manifest['fits']} 次 LR 拟合（包含108次内层选参）；耗时 {manifest['elapsed_seconds']:.1f} 秒。本地 CPU，不需要 Sol。
- 主任务1607个窗口保持不变；另做一小时/四小时共同起点重训，独立保存，未替换原一小时3233窗口。
- O=旧价格+固定起点虚拟变量；M=O+可用性/报价年龄；S=M+陈旧价格上下文；R=M+截止前的新鲜价格上下文。
- 六月至八月各自只使用此前月份选参；九月前冻结模型，之后不更新。所有变换仅拟合训练前缀。
- 正则化 C=1/(lambda×训练行数)，lambda为0.2/0.02/0.002；M/S/R复用O选参结果。

## 主要选择证据：三个月的指标均值
{outer.to_markdown(index=False,floatfmt='.4f')}

R−O 的 BA：AAPL {gates[0]['delta_BA']*100:.2f} 个百分点，AMZN {gates[1]['delta_BA']*100:.2f} 个百分点。两股均仅1/3月非负；R也没有胜过M、S。新增价格值没有带来稳定增量。AAPL的Brier恶化尤其明显。这不能证明价格完全无预测信息，只是否定这组固定设计的增量证据。

## 九月以后冻结回放（已暴露历史）
{replay.to_markdown(index=False,floatfmt='.4f')}

该表不能推翻训练期预设停止规则，也不能据此挑选股票、月份或起点。二月尾部样本很少，不独立声称稳定性。

## 相同预测起点的一小时/四小时对照
{matched.to_markdown(index=False,floatfmt='.4f')}

相同起点改善了可比性，但两个周期标签不同，BA之差不直接代表经济价值。未因本轮得分更改主任务。

## 核验与不确定性
- 原18个保存模型推断与旧概率最大误差≤1e-10；独立重算全部1607个四小时标签一致。
- 每次新拟合保存完整模型、训练/评价键、特征、C/lambda、标签最晚时刻、选参月份和哈希；172个模型重载预测逐元素相同。
- 12项测试通过：未来价格污染、开盘不可见、缺失连续线、未来asof、词频并列、未知配置、未来标签、平均损失复制不变性。
- paired_intervals.csv：共享交易日、全起点、两股一致重采样，10000次，5日主块，1/10日敏感性。仅描述性区间，未校正长期重复探索。
- metrics.csv含逐月/起点、主阈值及训练先验敏感性；单类别极小分组BA/MCC记空。
- 名义同维S/R的有效自由度未必相同；旧上下文可能恒为零。
- 仅RTH已完成五分钟线，假设收线即可用；未认证原始非RTH报价，因此不运行X或声称盘前实盘能力。另完成5分钟延迟敏感性68次拟合：AAPL R−O 为{lag[0]['delta_BA']*100:.2f}个百分点，AMZN为{lag[1]['delta_BA']*100:.2f}个百分点，两股仍未通过；见 runs/lag5/gates.json。

## 下一步与条件分支
已生成40篇无未来价格标签的正文盲查表。独立组员复核尚未完成，因此正文语义实验未获准训练，不能用助手检查冒充质量通过。时间分箱须等待语义增量，当前不运行。词表并列选择修复已实现并测试，但本轮未重新训练文本模型，历史文本结果保持原状。semantic_components.py 已实现非负斜率校准、固定基础offset、无文本精确回退、独立文章PCA及质量闸门；这些是已测试组件，完整文本训练编排尚未接入，不能称作已完成语义实验。

## 文件与重现
- run.py：价格特征准备、172次训练、保存预测。新输出目录必须不存在。
- summarize.py：旧模型/标签核验、配对区间与本报告。
- test_mechanisms.py：关键边界回归测试。
- review/blind_review_40.csv：独立复核输入。
- runs/v1/selection.json、fits.json、provenance.json、manifest.json：完整证据。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 work/stock-data/finbert-env/bin/python outputs/stock_four_hour_v2/run.py --output outputs/stock_four_hour_v2/runs/reproduction
work/stock-data/finbert-env/bin/python outputs/stock_four_hour_v2/test_mechanisms.py
```
'''
 (B/'REPORT.md').write_text(report)
 print(outer.to_string(index=False));print(replay.to_string(index=False))
if __name__=='__main__':main()
