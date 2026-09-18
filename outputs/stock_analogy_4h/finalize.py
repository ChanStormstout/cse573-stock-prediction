"""Update human-facing project entrypoints only after complete verified results."""
from common import *

def main():
    execution=json.loads((OUT/'execution.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['inference_calls']==execution['calls']
    assert (OUT/'model_reload.json').exists() and (OUT/'CASE_NOTES.md').exists()
    m=pd.read_csv(OUT/'metrics.csv');trans=pd.read_csv(OUT/'transitions.csv')
    def score(method,s,phase):return m[(m.method==method)&(m.symbol==s)&(m.phase==phase)].BA.iloc[0]*100
    table=['| 方法 | AAPL开发 | AMZN开发 | AAPL后续 | AMZN后续 |','|---|---:|---:|---:|---:|']
    for method in ['F0','R1','F1','F2','P0','P1','P2','P3']:
        vals=[score(method,s,p) for p in ['development','later'] for s in ['AAPL','AMZN']]
        table.append('| '+method+' | '+' | '.join(f'{v:.2f}%' for v in vals)+' |')
    changes=[]
    for s in ['AAPL','AMZN']:
        r=trans[(trans.symbol==s)&(trans.phase=='later')&(trans.base=='P2')&(trans.stratum=='all')].iloc[0]
        changes.append(f"{s}后续：加入历史收益比相同案例不带收益改对{r.corrected}、改错{r.broken}")
    gate=json.loads((OUT/'advancement.json').read_text())
    result='通过预注册探索线，但尚无独立泛化证明' if gate['passed'] else '未通过预注册探索线，不新增融合或按后续成绩挑股票赢家'
    body=(f"[报告](../outputs/stock_analogy_4h/v2/REPORT.md)、[案例](../outputs/stock_analogy_4h/v2/CASE_NOTES.md)。"
          f"实际完成{execution['calls']}次冻结Qwen3.5-9B本地推理，约{execution['seconds']/60:.1f}分钟；没有新LLM微调。"
          "P0当前新闻＋价格；P1相似案例投票；P2历史案例无结果；P3同样案例加已实现四小时收益。\n\n"+
          '\n'.join(table)+'\n\n'+
          '- '+'；'.join(changes)+'。\n'+
          '- '+result+'。\n'+
          '- 检索严格限于更早训练月份，September之后冻结August案例库。所有1,607窗口保留，1,374个OOF/开发/后续窗口评价。\n'+
          f"- V1因输入卡显示工资/市值、产品/减持错配中止，保留{execution['aborted_v1_calls']}次调用及源码；未计算V1 BA。V2收紧标题事件动作和未知拒绝，仍有主体/竞争对手、模板背景混淆。\n"+
          '- AMZN开发/后续各仅2个窗口触发案例；95个原无新闻后续窗口精确使用R1。因此结果不证明已解决新闻缺失，也不能排除更好的语义检索。\n'+
          '- 时间、未来相似度干预、标签翻转检索不变、缓存拒绝、精确回退、指标独立复算和模型重载通过。原文/prompt/权重留本地，独立抽取验收未完成；所有时期仍是暴露回测。\n')
    status=ROOT/'docs/CURRENT_STATUS.md';old=status.read_text();title='## 最新完成：历史相似新闻与已实现收益的LLM对照'
    if title not in old:
        first,rest=old.split('\n',1);status.write_text(first+'\n\n'+title+'\n\n'+body+'\n'+rest.lstrip())
    log=ROOT/'outputs/PROJECT_LOG.md';heading='## 2026-09-17：历史新闻案例＋已实现收益的冻结LLM对照'
    if heading not in log.read_text():
        log.write_text(log.read_text()+'\n\n'+heading+'\n\n'+
            '**为什么做：**检验历史相似新闻与当时完整四小时结果能否帮助LLM预测当前窗口，区别于单篇预测及单纯加入旧新闻。先保存协议，按输入卡修正一次检索，不用开发/后续分数设计。\n\n'+
            body.replace('../outputs/stock_analogy_4h/','stock_analogy_4h/')+'\n'+
            '**解释边界：**匹配对象/动作和市场状态尚不完善、训练库小且时间跨度有限，是有输入证据的限制；并非已证明所有失败都由它们导致。区分覆盖、匹配质量、预测增量；模型不生成解释，案例解释来自助手。\n')
    course=ROOT/'docs/COURSE_REPORT_4H.md';h='## 补充探索：历史案例收益增强预测'
    if h not in course.read_text():
        course.write_text(course.read_text()+'\n\n'+h+'\n\n'+body)
    files=sorted(OUT.glob('*.csv'))+sorted(OUT.glob('*.json'))+[HERE/'v1/abort.json']
    registry=ROOT/'scripts/repository_results.txt';names=registry.read_text().splitlines()
    for p in files:
        name=str(p.relative_to(ROOT))
        if name not in names:names.append(name)
    registry.write_text('\n'.join(names)+'\n')
    print(body)

if __name__=='__main__':main()
