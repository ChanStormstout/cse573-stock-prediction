"""Evaluate only after immutable inference completes; no evaluation-based fitting."""
from common import *

METHODS=['F0','R1','F1','F2','F6','P0','P1','P2','P3']

def main():
    check_prepared();execution=json.loads((PRIVATE/'completed.json').read_text())
    d=pd.read_csv(PRIVATE/'prepared.csv',float_precision='round_trip')
    scores={(r['key'],r['variant']):r['p'] for r in jsonl(PRIVATE/'generated.jsonl')}
    for m in ['P0','P2','P3']:
        d[m]=[np.nan if r.phase=='warmup' else (r.R1 if not r.has_excerpt else scores[(r.key,'P0' if m!='P0' and r.n_cases==0 else m)]) for r in d.itertuples()]
    d.loc[d.phase=='warmup',METHODS]=np.nan
    d.to_csv(OUT/'predictions.csv',index=False);ev=d[d.phase!='warmup'];metrics=[];months=[];strata=[];trans=[]
    for m in METHODS:
        for (s,p),g in ev.groupby(['symbol','phase']):metrics.append(dict(method=m,symbol=s,phase=p,**metric(g.label,g[m])))
        for (s,month),g in ev.groupby(['symbol','month']):months.append(dict(method=m,symbol=s,month=month,**metric(g.label,g[m])))
        for (s,p),g in ev.groupby(['symbol','phase']):
            for name,mask in [('has_analog',g.n_cases>0),('no_analog',g.n_cases==0),('no_original_news',g.has_original_news==0)]:
                a=g[mask]
                if len(a):strata.append(dict(method=m,symbol=s,phase=p,stratum=name,**metric(a.label,a[m])))
    for (s,p),g in ev.groupby(['symbol','phase']):
        for b in ['P0','P1','P2','R1','F2']:
            for name,mask in [('all',np.ones(len(g),bool)),('has_analog',g.n_cases>0),('no_original_news',g.has_original_news==0)]:
                a=g[mask];new=(a.P3>=.5)==a.label;old=(a[b]>=.5)==a.label
                trans.append(dict(symbol=s,phase=p,base=b,stratum=name,n=len(a),corrected=int((new&~old).sum()),
                                  broken=int((~new&old).sum()),changed=int(((a.P3>=.5)!=(a[b]>=.5)).sum())))
    met=pd.DataFrame(metrics);monthly=pd.DataFrame(months)
    met.to_csv(OUT/'metrics.csv',index=False);monthly.to_csv(OUT/'monthly_metrics.csv',index=False)
    pd.DataFrame(strata).to_csv(OUT/'strata_metrics.csv',index=False);pd.DataFrame(trans).to_csv(OUT/'transitions.csv',index=False)
    intervals=[]
    for phase,g in ev.groupby('phase'):
        days=sorted(g.day.unique());n=len(days);idx={day:i for i,day in enumerate(days)}
        for block in [1,5]:
            rng=np.random.default_rng(573);weights=[]
            for _ in range(500):
                starts=rng.integers(0,n,size=int(np.ceil(n/block)))
                draw=np.concatenate([(start+np.arange(block))%n for start in starts])[:n]
                weights.append(np.bincount(draw,minlength=n))
            weights=np.array(weights)
            for s,a in g.groupby('symbol'):
                w=weights[:,[idx[day] for day in a.day]];y=a.label.to_numpy()
                def ba(q):
                    with np.errstate(invalid='ignore',divide='ignore'):
                        return .5*((w[:,y==1]@q[y==1])/w[:,y==1].sum(1)+(w[:,y==0]@(~q[y==0]))/w[:,y==0].sum(1))
                for b in ['P0','P1','P2','R1','F2']:
                    delta=ba(a.P3.to_numpy()>=.5)-ba(a[b].to_numpy()>=.5)
                    lo,hi=np.nanquantile(delta,[.025,.975])
                    intervals.append(dict(symbol=s,phase=phase,base=b,block_days=block,BA_delta_low=lo,BA_delta_high=hi))
    pd.DataFrame(intervals).to_csv(OUT/'paired_intervals.csv',index=False)
    # Descriptive mechanism checks only; never route or tune from these outcomes.
    audits={r['key']:r for r in json.loads((PRIVATE/'retrieval.json').read_text())}
    source=pd.read_pickle(PRIVATE/'rows.pkl');mechanism=[]
    for (s,phase),g in ev[ev.n_cases>0].groupby(['symbol','phase']):
        first=np.array([source.iloc[audits[k]['case_indices'][0]].label for k in g.key])
        last=np.array([source.iloc[audits[k]['case_indices'][-1]].label for k in g.key])
        q=g.P3.to_numpy()>=.5;change=q!=(g.P2.to_numpy()>=.5)
        mechanism.append(dict(symbol=s,phase=phase,n=len(g),P3_agrees_nearest_case=float((q==first).mean()),
            P3_agrees_last_case=float((q==last).mean()),P3_agrees_vote=float((q==(g.P1.to_numpy()>=.5)).mean()),
            directions_changed_vs_P2=int(change.sum()),mean_abs_probability_change=float(abs(g.P3-g.P2).mean())))
    pd.DataFrame(mechanism).to_csv(OUT/'mechanism_diagnostics.csv',index=False)
    # Diagnostic upper bound: a non-deployable oracle corrects ONLY activated windows.
    ceilings=[]
    for (s,phase),g in ev.groupby(['symbol','phase']):
        q=(g.P0.to_numpy()>=.5).astype(float);active=g.n_cases.to_numpy()>0
        q[active]=g.label.to_numpy()[active]
        ceilings.append(dict(symbol=s,phase=phase,modifiable_windows=int(active.sum()),
                             P0_BA=metric(g.label,g.P0)['BA'],oracle_BA=metric(g.label,q)['BA'],
                             deployable=False))
    pd.DataFrame(ceilings).to_csv(OUT/'coverage_ceiling.csv',index=False)
    outer=monthly[monthly.month.between('2018-06','2018-08')];gates=[]
    for b in ['P0','R1']:
        for s in ['AAPL','AMZN']:
            aa=outer[(outer.method=='P3')&(outer.symbol==s)].set_index('month')
            bb=outer[(outer.method==b)&(outer.symbol==s)].set_index('month')
            delta=aa.BA-bb.BA;db=aa.Brier-bb.Brier
            gates.append(dict(base=b,symbol=s,monthly_BA_gain=float(delta.mean()),Brier_change=float(db.mean()),
                positive_months=int((delta>0).sum()),passed=bool(delta.mean()>=.01 and db.mean()<=.002 and (delta>0).sum()>=2)))
    dump(OUT/'advancement.json',dict(passed=all(g['passed'] for g in gates),checks=gates,selected_fusion=False))
    old=jsonl(W/'analogy_4h/v1/generated.jsonl')
    dump(OUT/'execution.json',{**execution,'aborted_v1_calls':len(old),'aborted_v1_seconds':sum(r['seconds'] for r in old),
                            'trained_parameters':0,'learned_retriever':False,'label_fits':0})
    lines=['# 历史新闻＋已实现收益：有限四小时对照','',
           '**全部时期是已暴露的探索性回测；没有独立新时期验证。**','',
           '结论解释见[CONCLUSIONS.md](CONCLUSIONS.md)；固定案例见[CASE_NOTES.md](CASE_NOTES.md)。','',
           'P0当前新闻＋价格；P1历史案例加权投票；P2给LLM历史输入但隐藏历史结果；P3给相同案例及实际四小时收益。',
           'F0价格＋标题是课程baseline；F1价格＋全文、F2价格＋冻结FinBERT、F6旧融合是既有系统参照，其文本预算不同，不是本轮机制的匹配消融。','',
           '| 方法 | AAPL训练OOF | AMZN训练OOF | AAPL开发 | AMZN开发 | AAPL后续 | AMZN后续 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        vals=[met[(met.method==m)&(met.symbol==s)&(met.phase==p)].BA.iloc[0] for p in ['train_oof','development','later'] for s in ['AAPL','AMZN']]
        lines.append('| '+m+' | '+' | '.join(f'{v*100:.2f}%' for v in vals)+' |')
    lines+=['','## Brier（越低越好）','',
            '| 方法 | AAPL开发 | AMZN开发 | AAPL后续 | AMZN后续 |','|---|---:|---:|---:|---:|']
    for m in METHODS:
        vals=[met[(met.method==m)&(met.symbol==s)&(met.phase==p)].Brier.iloc[0] for p in ['development','later'] for s in ['AAPL','AMZN']]
        lines.append('| '+m+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','## 覆盖和输入','', (OUT/'coverage.csv').read_text(),
            '- 未改变原定窗口。无合格当前段落时所有新方法严格R1；有段落而没有合格历史案例时P2/P3严格P0、P1严格R1。',
            '- 只使用更早训练月份的同股案例，开发和后续的案例库冻结于August。每例结果使用原四小时窗口，而非新闻发布后四小时。',
            '- 当前/案例最多一篇文章、120词完整句摘录、最多3例。检索是词汇＋标题事件动作启发式，不是深度语义检索或正式事实抽取。',
            '- 近似转载按过去标题/摘录在线归组，每组最早窗口、每个检索集合一天至多一例。不能保证不同日期都属于独立经济事件。','',
            '## 预注册晋级检查（只用June–August）','']
    for g in gates:lines.append(f"- {g['symbol']} P3对{g['base']}：月均BA {100*g['monthly_BA_gain']:+.2f}pp；Brier {g['Brier_change']:+.4f}；正增益月份{g['positive_months']}/3；{'通过' if g['passed'] else '未通过'}。")
    lines+=['','## 实际执行与限制','',
            f"冻结Qwen3.5-9B完成{execution['calls']}次本地推理，已记录调用耗时{execution['seconds']/60:.1f}分钟，MLX峰值{execution['peak_mlx_gb']:.2f}GB。耗时不含准备/模型加载/检查，也不含V1中断时未写完的批次。没有新梯度训练、LLM微调、检索器训练或付费API。",
            'UP/DOWN条件token偏好作为分数，尚未校准。单个确定性运行不虚构三种子波动。模型未输出解释；案例中的语义分析是助手输入审阅，不是模型思维过程。',
            'V1因输入卡显示市值/工资、产品/减持错配而中止，未计算或查看该版BA；保留部分调用成本、输入和源码。V2改为细分标题事件动作、删除通用词、拒绝未知类，不按结果继续改网格。',
            '现代预训练模型可能见过这些历史事件；提示限制无法证明消除记忆。新闻结果相关不等于因果，案例并不保证同市场状态。',
            '完整BA/MCC/Brier、逐月、覆盖、恒定预测比例、改对改错和1/5日块差值区间见CSV。区间未校正反复探索；不是独立显著性证明。',
            'mechanism_diagnostics只描述P3与最近/最后案例和投票方向的一致程度，不能据此证明LLM在复制标签或进行某种内部推理。',
            'coverage_ceiling是假设所有有案例窗口都能事后改对、其余窗口保持P0的BA上限，仅用于诊断本版覆盖约束，不能部署或作为模型成绩。',
            '本轮结果不能排除更好的语义检索、更丰富事件状态或任务微调的作用，但没有根据后续单股最高分选择赢家。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(met.pivot(index='method',columns=['phase','symbol'],values='BA').round(4).to_string())

if __name__=='__main__':main()
