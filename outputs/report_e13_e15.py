"""Build source-backed Chinese reports and inspectable case panels."""
from pathlib import Path
import json,sys,platform,importlib.metadata
import pandas as pd,numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'stock_adaptive'))
from common import *
def table(df):
    cols=list(df.columns);lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for row in df.itertuples(index=False,name=None):lines.append('| '+' | '.join(str(x).replace('|',' / ').replace('\n',' ') for x in row)+' |')
    return '\n'.join(lines)
def formatted(df):
    df=df.copy()
    for c in ['balanced_accuracy','accuracy']: 
        if c in df:df[c]=df[c].map(lambda x:f'{x:.2%}')
    for c in ['brier','mcc']:
        if c in df:df[c]=df[c].map(lambda x:f'{x:.4f}')
    return df
def stock_table(df,keys):
    a=[]
    for key,g in df.groupby(keys):
        key=key if isinstance(key,tuple) else (key,);r=dict(zip(keys,key))
        for s in ['AAPL','AMZN']:
            row=g[g.symbol.eq(s)].iloc[0];r[s+' BA']=f'{row.balanced_accuracy:.2%}';r[s+' Brier']=f'{row.brier:.4f}'
        a.append(r)
    return table(pd.DataFrame(a))
def contrast_table(path):
    d=pd.DataFrame(json.loads(path.read_text()));d=d[d.period.eq('former_holdout')].copy();d['delta_pp']=d.delta.map(lambda v:f'{v*100:+.2f}');d['95%五日块区间']=d.CI95.map(lambda v:f'[{v[0]*100:+.2f}, {v[1]*100:+.2f}]');return table(d[[c for c in ['symbol','penalty','feature','contrast','delta_pp','95%五日块区间'] if c in d]])
if __name__=='__main__':
    A=O/'stock_adaptive';F=O/'stock_event_facts_v3';T=O/'stock_small_boost';a=pd.read_csv(A/'results/summary.csv');f=pd.read_csv(F/'prediction/summary.csv');t=pd.read_csv(T/'results/summary.csv');a0=a[a.period.eq('former_holdout')];f0=f[f.period.eq('former_holdout')];t0=t[t.period.eq('former_holdout')];coverage=pd.read_csv(F/'prediction/coverage.csv')
    am=pd.read_csv(A/'results/monthly.csv');fm=pd.read_csv(F/'prediction/monthly.csv');tm=pd.read_csv(T/'results/monthly.csv');verification=json.loads((O/'E13_E15_VERIFICATION.json').read_text())
    runtime=dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','joblib','nltk']},threads='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1',recorded_after_run=True)
    (O/'E13_E15_RUNTIME.json').write_text(json.dumps(runtime,indent=2))
    common='所有新结果均为 E12 已暴露后的探索性历史回测，不是新的独立测试。预测当月只使用此前已实现标签；阈值固定 0.5。BA 是上涨/下跌召回率的平均，Brier 越低越好，固定 0.5 概率的 Brier 为 0.25。区间为 1,500 次配对五日移动块重采样的描述性区间，未校正整个多次选方案过程。'
    ar=['# E13：冻结、扩展与六个月滚动训练','','2026-09-15｜已完成并核验。',common,'','## 设计与实际工作','','同一全文二值词干、卡方 500 特征、L1/L2 LR；C 候选 0.01/0.1/1。两股、两种正则、三种训练策略、六个月，共 72 份月度模型记录。冻结臂复用 24 份原模型副本；48 次更新各做 9 个过去内部验证拟合与一次最终拟合。','',
    '扩展臂使用自 1 月开始的全部过去数据；滚动臂使用预测月之前六个日历月。每次选参在该次训练范围最后三个完整月向前验证，词表、筛选、缩放随训练折重新拟合。9–10 月与原 11 月之后的保留期分开汇总；2 月仅 1 日，每股 5 窗口。','',
    '## 原保留期结果（11 月—2 月 1 日）','',stock_table(a0,['penalty','policy']),'','## 配对差值','',contrast_table(A/'results/intervals.json'),'','## Insight：52.53% 不能直接解释为词语规律变稳定了','','AAPL 滚动 L1 从原冻结 49.24% 到 52.53%，差 +3.29 个百分点；五日块区间约 [−6.03,+8.54]，跨零。AMZN 滚动 L1 为 50.36%，没有超过冻结 50.92%；L2 的 AMZN 更新臂均明显低于原冻结 55.04%。','','进一步核对 AAPL 滚动 L1：12 月与 1 月均选择 C=0.01，系数与截距全部为零，所有概率恰为 0.5，按固定阈值始终判上涨。这两个月各自 BA 都为 50%。三个完整月份 BA 为 51.88%、50%、50%，不支持“跨月明显改善”的说法。整体合并 BA 与月度平均不同，因为各月类别比例及预测行为不同。','','概率损失改善也不能全归因于更好的语义学习：强正则化使模型在部分月份退回无信息概率，避免了原模型过度自信。滚动训练同时改变样本量、词表和所选正则强度；本对照检验整个更新策略，不单独识别哪个因素造成变化。','','## 全部逐月结果','',table(formatted(am[['symbol','penalty','policy','month','C','n_train','n','balanced_accuracy','brier']])),'','## 复现与失败记录','','72 份保存模型重现全部 7,398 条策略预测；原 E12 冻结预测逐项一致；选参表、训练键和标签截止检查通过；另独立重训四个代表更新模型，系数重现。初次启动发生模块同名导致的循环导入，在登记协议/训练之前退出；修正加载方式后重新启动，日志保留于 work/e13_run.log 与 work/e13_run_attempt2.log。','','[协议](protocol.json) · [逐窗口预测](results/predictions.csv) · [检查](results/audit.json) · [总体报告](../EXPERIMENT_PROGRESS_E13_E15.md)']
    (A/'REPORT.md').write_text('\n'.join(ar)+'\n')
    # Human review file is a copy; immutable assistant results and model inputs are separate.
    reviews=json.loads((F/'results/assistant_review.json').read_text());rv=pd.DataFrame(reviews);rv['human_key_facts_correct']='';rv['human_note']='';rv.to_csv(F/'HUMAN_REVIEW_TEMPLATE.csv',index=False)
    fr=['# E14：具体事件抽取与时效聚合','','2026-09-15｜实现了三版抽取；完成限定字段 B0–B3 回测。**完整事件系统未通过质量验收。**',common,'','## 三版质量检查与范围变化','','| 版本 | 助手检查 | 观察 | 决定 |','|---|---|---|---|','| v1 | 18/37 个接受案例的完整字段合格 | 历史旁支事件、动作丢失、机构错配 | 未进入预测 |','| v2 | 25/31 合格 | 标题 upgrade/downgrade 不一定是评级变化；机构范围错误 | 未进入预测 |','| v3 | 27/30 完整字段合格 | 目标价机构字段 16/18，低于 90%；评级 11/12 | 完整结构与机构关系图仍不通过 |','','三次使用不同检查样本，不能将比率画成同一测试集上的改进曲线。新记录不一定是独立新事件，存在近似转载。标签来自助手逐条阅读，没有冒充组员独立金标准。','','### 为什么仍运行了一个限缩版预测实验','','另行登记 E14_LIMITED_PROJECTION：只使用目标公司、事件类型、动作、旧新数值或评级变化；彻底排除机构名称/机构是否存在等字段，不将失败的完整 schema 宣布通过。这些实际输入字段在 v3 的 30 个 AAPL 检查记录上均有证据支持；这只是助手检查、样本相关且小，不能证明全语料准确率 100%。AMZN 没有新的质量检查样本，其结果只作覆盖受限的探索性记录。完整结构门槛失败状态保留在 quality.json。','','业绩指引抽取已实现为保守的公司收入区间：AAPL 训练期仅 8 条，AMZN 为 0；没有足够新质量样本和独立披露，不进入本轮主预测。未声称实现完整指引修订、预期差或首次披露识别。','','## 关键案例','','- 评级标题与正文纠正冲突：Morgan Stanley 标题称 downgrade，正文末尾明确仍为 overweight，只下调目标价。v2 误判评级下降，v3 要求正文明确旧评级→新评级，未找到就弃权。','- Amazon 的 upgrade：正文是目标价 1,850→2,500，并维持 overweight。v2 评级上调错误，v3 删除这条不受支持的评级事实；未擅自把它补成全部可用的完整事件。','- 目标价正确配对：Maxim 的 Apple 193→204、Barclays 的 168→157等，保留对应原句和变化比率。','- 机构名仍有错误：Wells Fargo & Co 后面的句子被接入机构字段，或标题 from Over 被误当机构。因此本轮完全排除机构特征，关系图保持待做。','','## 输入覆盖','',table(coverage.assign(event_coverage=coverage.event_coverage.map(lambda x:f'{x:.2%}'))),'','AAPL 训练只有 162/996 窗口含本轮接受事件，其中成对旧新目标价仅来自 24 篇文章；AMZN 训练仅 8/1004 窗口、2 篇文章，原保留期仅 4/365 窗口、1 篇文章。这限制了估计稳定性；不能根据 AMZN 的极少事件判断所有事件方法无效。','','## B0–B3：同样的每月扩展训练','','- B0：16 个原价格特征。','- B1：价格＋事件数量、是否存在及目标价/评级类别比例。','- B2：B1＋11 个动作/变化/已知标记特征，包含目标价变化比例和评级变化方向。','- B3：相同接受记录，按文章发布时间代理年龄，H=4 小时、λ=1、冻结规则 q=.5 或 1，使用收缩聚合；追加发布时间年龄、抓取延迟、标题支持比例。没有搜索 H，没有用未来信息追溯事件簇。','','B1–B3 接受记录相同，全部 3,233 原样本保留。新闻列仅除以训练标准差，不做减均值，保留无事件为零；价格正常标准化。B3 含 18 个事件/时效特征。机构字段不进入模型。模型是小型 L2 LR，每臂相同三个 C 候选和向前选参。','','## 原保留期结果','',stock_table(f0,['kind']),'','## 逐项贡献与区间','',contrast_table(F/'prediction/intervals.json'),'','## 结论与限制','','这次限定事实字段和收缩时效处理没有带来明显收益。AAPL 从 B1 的 47.35% 到 B2 的 47.20%，B3 降至 45.65%；AMZN 从 50.11% 降至 48.74% 和 48.19%。时效收缩有明确数学行为，但在这套输入/标签与更新方案下没有转换为更好的预测。','','q 不是已校准的正确概率；发布时间不是可靠首次披露时间；重复报道仍可能累积权重。本轮没有同时增加事件去重、更多分类器或图结构，因而没有把多项变化混成一个未经控制的提升。由于 B3 未见收益，暂不继续拆分扩大其超参数与聚合搜索。','','无事件窗口的预测也可能改变：不同特征联合训练会改变价格系数与截距。这不能直接解释为该窗口的新闻产生了影响。','','## 全部逐月结果','',table(formatted(fm)),'','## 交付','','[限缩版协议](prediction_protocol.json) · [抽取结果](results/events.jsonl) · [完整质量状态](results/quality.json) · [助手检查](results/assistant_review.json) · [组员复核表副本](HUMAN_REVIEW_TEMPLATE.csv) · [逐窗口预测](prediction/predictions.csv) · [总体报告](../EXPERIMENT_PROGRESS_E13_E15.md)']
    (F/'REPORT.md').write_text('\n'.join(fr)+'\n')
    tr=['# E15：浅层梯度提升树与同输入 LR','','2026-09-15｜实际完成并核验。',common,'','## 设计','','每月扩展训练，分别使用价格、价格＋原新闻数量，两种特征各比较 LR 和小型 HistGradientBoosting。树深最多 2、叶子最多 4、每叶至少 30 个训练样本、100 次提升、学习率 0.05，关闭随机早停。LR 比较三个 C；树比较三种 L2 强度。全部选择仅使用过去三个月的向前验证；同样的输入、样本和候选数。','','## 原保留期结果','',stock_table(t0,['feature','kind']),'','## 配对差值','',contrast_table(T/'results/intervals.json'),'','## Insight','','AAPL 提升树比相同输入 LR 高约 2–2.5 个百分点，但两种特征下 BA 都没有超过 50%；AMZN 两种提升树低于相应 LR。Brier 略降仍普遍高于 0.25。有限非线性容量没有解决跨股稳定性，不能将某一股相对一个较弱对照的改善视作成功系统。','','## 全部逐月结果','',table(formatted(tm)),'','[协议](protocol.json) · [预测](results/predictions.csv) · [统一核验](../E13_E15_VERIFICATION.json) · [总体报告](../EXPERIMENT_PROGRESS_E13_E15.md)']
    (T/'REPORT.md').write_text('\n'.join(tr)+'\n')
    main=['# E13–E15：定期更新、具体事件与浅层提升树的实际实验','','2026-09-15。**主要阶段已执行；目前未找到稳定、明显的预测提升。**','','本轮完成：模型更新策略对照；三版具体事件抽取和逐条质量检查；一个排除不可靠机构字段及稀少指引的 B0–B3 限缩版回测；浅层提升树对照。完整事件系统、机构关系图、学习注意力与更大微调没有宣布完成。',common,'','## 1. 定期更新：有单股表面改善，尚无稳定收益','',stock_table(a0,['penalty','policy']),'','AAPL 滚动 L1 的 52.53% 比冻结高 3.29 个百分点，配对五日块区间 [−6.03,+8.54]。12 月、1 月模型的全部系数和截距却为零，恒定输出 0.5；两个整月 BA 都是 50%。这提示部分概率改善来自避免过度自信，而不是已经学到了稳定的新词语规律。AMZN 更新没有超过冻结基线，全文 L2 反而从 55.04% 降至约 48%。','','[E13 完整报告](stock_adaptive/REPORT.md)','',
    '## 2. 事件事实：质量收窄后仍无预测收益','',stock_table(f0,['kind']),'','B0=价格；B1=事件数量/类别；B2=加入对象对应的动作和数值变化；B3=再加入时效收缩。它们全部使用相同的每月扩展训练，不能与 E12 的静态模型直接当作单一特征消融比较。','','### 质量关口没有被隐藏','','v1、v2 因历史旁支事件、动作丢失和标题歧义未进入预测。v3 完整字段仍有机构错误，未达到完整结构的门槛；因此另行登记限缩实验，排除机构，使用助手已检查的公司/动作/数值字段。30 个 v3 AAPL 记录的这些输入字段均有原文支持，尚无独立组员质量验收，也没有 AMZN 新质量样本。','','**覆盖是现实限制：**AAPL 训练 162/996 窗口有接受事件；AMZN 只有 8/1004。AMZN 原保留期仅 1 篇接受事件、4 个窗口。业绩指引训练记录仅 AAPL 8 条、AMZN 0 条，未纳入预测。该实验不能代表完整两类事件系统或事件方法的性能上限。','','[E14 实现、案例、质量与完整报告](stock_event_facts_v3/REPORT.md)','',
    '## 3. 新增浅层提升树：未解决稳定性','',stock_table(t0,['feature','kind']),'','AAPL 对同输入 LR 有小幅相对改善，AMZN 下降；没有形成跨两股的明确收益。[E15 完整报告](stock_small_boost/REPORT.md)','',
    '## 4. 本轮得到的 insight','','1. **定期更新不能预设有效。** 近期样本更少、词表变化和选参波动会与适应新关系同时发生。','2. **标题也可能语义不准确。** Apple 的 downgrade、Amazon 的 upgrade 在正文中实际指目标价变动；对象正确仍不代表事件类型正确。','3. **输入更保守不保证更会预测。** 本轮动作/数值投影没有提高方向表现；全部语义问题并未解决，也不能据此判定新闻没有信息。','4. **更少但较可靠的事件会付出覆盖代价。** 两股尤其 AMZN 的监督非常稀疏；堆完整图或更大模型缺少当前实验支持。','5. **数学设计需要接受实验结果。** λ+总权重确实能使弱证据特征收缩，但本轮 B3 未优于对照。','','## 5. 验证与可复现性','','- 168 份月度保存模型/副本的预测核验：E13 72，E14 48，E15 48；四个 E13 代表模型和四个数值模型独立重训重现。','- 检查全部 3,233 样本的事件成员和截止；引文、数值变化、无事件零值与有限特征检查通过。','- B0 与提升树实验的相同价格 LR 逐窗口概率一致，保证两个实验的基础对照对齐。','- 204 个早期产物、171 个上一轮实验文件、23 个 E12 产物哈希均未变化。','- 不覆盖旧结果；每一版有独立目录和协议，失败抽取保留。没有付费模型调用。','','[核验结果](E13_E15_VERIFICATION.json) · [运行环境](E13_E15_RUNTIME.json) · [模型更新核验](stock_adaptive/results/audit.json)','',
    '## 6. 当前停止决定与剩余工作','','本轮没有把不稳定的单股高分挑出来作为最终系统，也没有继续扩大网格。暂不自动增加注意力、全参数微调、完整 StockNet 或 GNN。机构和首次披露识别仍未解决，组员独立复核表已交付；这阻止完整事件/关系系统被称为完成，但不影响已完成限缩回测的保存与报告。','','用于课程展示，可以把主线收敛为“为什么静态全文模型的开发高分没有保持，以及更新/事实/非线性三种针对性改进的实际效果”。原最终测试仍完整保留；新方案若要确认泛化提升，需要未参与本轮设计的后续数据。','','[完整项目日志](PROJECT_LOG.md)']
    (O/'EXPERIMENT_PROGRESS_E13_E15.md').write_text('\n'.join(main)+'\n')
    # A compact case panel covers every outcome type; selection is descriptive, not causal attribution.
    preds=pd.read_csv(F/'prediction/predictions.csv');parts=[];cards=['# E14：B3 相对 B2 的案例面板','','按股票和改对/改错/共同正确/共同错误分层，每层最多三例，优先有事件和概率变化较大者。全部原保留期窗口都有完整预测；这些选例不能代表总体频率，也不能证明因果。']
    events=[json.loads(x) for x in (F/'results/events.jsonl').read_text().splitlines()];lookup={(x['symbol'],x['record_key']):x for x in events if x['kind']!='revenue_guidance'};D=pd.read_pickle(F/'prediction/data.pkl');D['start_utc']=D.start_utc.astype(str)
    for s,g in preds[preds.split.eq('test')].groupby('symbol'):
        w=g.pivot(index=['start_utc','label','event_has'],columns='kind',values='p').reset_index();correct2=(w.B2>=.5)==w.label.astype(bool);correct3=(w.B3>=.5)==w.label.astype(bool);w['group']=np.select([~correct2&correct3,correct2&~correct3,correct2&correct3],['fix','break','both_correct'],default='both_wrong');w['abs_delta']=(w.B3-w.B2).abs();w['symbol']=s
        for group,h in w.groupby('group'):
            pick=h.sort_values(['event_has','abs_delta'],ascending=[False,False]).head(3);parts.append(pick)
            for r in pick.itertuples():
                cards+=['',f'## {s} {r.start_utc}：{group}',f'标签={r.label}；B2 p={r.B2:.4f}；B3 p={r.B3:.4f}；有接受事件={r.event_has}。']
                row=D[D.symbol.eq(s)&D.start_utc.eq(r.start_utc)].iloc[0];fs=[lookup[(s,k)] for k in row.news_record_keys.split('|') if (s,k) in lookup]
                if not fs:cards+=['无接受事件；预测差异来自联合拟合的参数变化，不能归因于该窗口读懂了新闻。']
                for ev in fs[:3]:cards += [f'- {ev["kind"]} / {ev["action"]}；旧值 {ev.get("old_value")} → 新值 {ev.get("new_value")}；发布时间 {ev["published_utc"]}；可用时间 {ev["available_utc"]}。',f'  证据：{ev["evidence"]}',f'  记录键：`{ev["record_key"]}`']
    pd.concat(parts).to_csv(F/'prediction/case_panel.csv',index=False);(F/'CASES.md').write_text('\n'.join(cards)+'\n')
    print('Reports and case panel written.')
