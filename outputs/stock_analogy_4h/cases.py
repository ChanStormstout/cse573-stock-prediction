"""Reveal outcomes on the same pre-inference case panel; publish no source text."""
from common import *

def main():
    assert (PRIVATE/'completed.json').exists()
    d=pd.read_csv(OUT/'predictions.csv',float_precision='round_trip').set_index('key')
    docs=json.loads((PRIVATE/'documents.json').read_text());audit=json.loads((PRIVATE/'retrieval.json').read_text())
    rows=pd.read_pickle(PRIVATE/'rows.pkl');a={r['key']:r for r in audit}
    notes=['# 固定案例：输入审阅后揭示预测','',
           '输入阶段的发现见[预先审阅](CASE_INPUT_REVIEW.md)。这些案例按key哈希选出，不按预测是否改对挑选。',
           '以下解释属于助手分析，不是LLM生成理由；单词评分接口未生成推理过程。原文与prompt仅留本地。','',
           '| 窗口UTC | 历史案例数 | 实际方向 | P0 | P1 | P2 | P3 | P3对P2 |',
           '|---|---:|---|---:|---:|---:|---:|---|']
    case_rows=[]
    for key in json.loads((PRIVATE/'case_selection.json').read_text()):
        r=d.loc[key];old=(r.P2>=.5)==r.label;new=(r.P3>=.5)==r.label
        change='改对' if new and not old else ('改错' if old and not new else '方向未变' if (r.P2>=.5)==(r.P3>=.5) else '其他')
        notes.append('| '+key.replace('|',' ')+' | '+str(r.n_cases)+' | '+('涨' if r.label else '跌')+' | '+
                     ' | '.join(f'{r[m]:.3f}' for m in ['P0','P1','P2','P3'])+' | '+change+' |')
        case_rows.append(dict(key=key,n_cases=int(r.n_cases),label=int(r.label),
                              **{m:float(r[m]) for m in ['P0','P1','P2','P3']},transition=change))
    notes+=['','数值为上涨分数。P1为平滑案例投票；LLM分数为UP/DOWN条件token概率，未经校准。',
            '无案例时P2/P3逐位相同；无可靠段落时四组逐位等于R1。改对/改错不代表新闻因果效应。',
            '即便动作匹配，目标价改变后四小时也可能相反；即便方向猜中，实体角色匹配错误仍是输入问题，不能用猜中替它验收。']
    (OUT/'CASE_NOTES.md').write_text('\n'.join(notes)+'\n');dump(OUT/'cases.json',case_rows)

if __name__=='__main__':main()
