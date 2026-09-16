"""Local offline demo: real saved-model inference, optional separate outcome reveal."""
from core import *
# core loads legacy transforms and changes the search path; select this demo's
# inference module explicitly rather than the older experiment's predict.py.
sys.path.insert(0, str(B))
from predict import predict
from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib.parse import urlparse,parse_qs
import argparse,zipfile

PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CSE 573 · 四小时预测</title>
<style>body{font:16px system-ui,sans-serif;background:#f3f6fb;color:#172336;margin:0}main{max-width:1080px;margin:42px auto;padding:0 24px}h1{font-size:34px;letter-spacing:-1px;margin:8px 0}small,.muted{color:#58677b}header{margin-bottom:28px}.tag{display:inline-block;background:#e3ecff;color:#244ba1;border-radius:20px;padding:5px 12px;font-size:13px}.controls,.panel{background:white;padding:22px;border:1px solid #dfe5ef;border-radius:14px;margin:16px 0}.controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap}select,button{font:inherit;padding:10px;border:1px solid #cad5e7;border-radius:8px}button{background:#2457c5;color:white;cursor:pointer}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.cards div{background:white;border:1px solid #dfe5ef;border-radius:14px;padding:22px}.value{font-size:36px;display:block;font-weight:650;margin:10px 0}.row{display:grid;grid-template-columns:160px 1fr 65px;gap:14px;align-items:center;margin:14px 0}.track{background:#e9eef6;height:12px;border-radius:10px}.fill{background:#527cd4;height:100%;border-radius:10px}pre{white-space:pre-wrap;line-height:1.65;font-family:inherit}.article{border-top:1px solid #e3e8ef;padding-top:14px;margin-top:18px}#outcome{color:#274d94;font-weight:600}code{font-size:13px}footer{padding:20px 0;font-size:13px;color:#58677b}@media(max-width:650px){.cards{grid-template-columns:1fr}.row{grid-template-columns:100px 1fr 55px}}</style>
<main><header><span class="tag">CSE 573 · 历史回放</span><h1>新闻能修正价格预测吗？</h1><p class="muted">AAPL / AMZN · 固定四小时 · 从已保存权重实际推断</p></header>
<div class="controls"><label>股票 <select id="stock"><option>AAPL</option><option>AMZN</option></select></label><label>起点 UTC <select id="date"></select></label><button id="go">运行预测</button><label><input type="checkbox" id="reveal">显示实际结果</label></div>
<p id="state" class="muted">正在读取可用时点…</p><div id="result" hidden><div class="cards"><div>标题 baseline<span class="value" id="baseline"></span><small>经典价格＋标题模型</small></div><div>条件新闻修正<span class="value" id="gate"></span><small>探索模型；不代表验收通过</small></div><div>预设规则选择的系统<span class="value" id="selected"></span><small id="selectedName"></small></div></div>
<section class="panel"><h2>上涨概率与新闻贡献</h2><div id="bars"></div><p id="weights"></p><p class="muted" id="quality"></p><p id="outcome"></p></section><section class="panel"><h2>当时可用的新闻证据</h2><p class="muted">只展示最新两篇原文节选；并非完整窗口内容。</p><div id="news"></div></section></div><footer>预测函数不接收未来标签。所有时期已用于项目探索；这些结果不是新的独立泛化检验。FinBERT 编码器冻结。</footer></main>
<script>const el=id=>document.getElementById(id);let choices;const pct=x=>(100*x).toFixed(2)+'%';function dates(){el('date').replaceChildren(...choices[el('stock').value].map(v=>new Option(v,v)))}async function run(){el('state').textContent='正在加载权重并计算…';try{const q=new URLSearchParams({symbol:el('stock').value,start:el('date').value,reveal:el('reveal').checked?'1':'0'});const r=await fetch('/predict?'+q).then(x=>x.json());if(r.error)throw Error(r.error);el('result').hidden=false;el('baseline').textContent=pct(r.baseline_probability);el('gate').textContent=pct(r.gate_probability);el('selected').textContent=pct(r.selected_probability);el('selectedName').textContent=r.selected_method==='title'?'新方法未通过继续条件，保留标题 baseline':r.selected_method;el('state').textContent='截止 '+r.cutoff+' ｜ 目标 '+r.start+' → '+r.end;el('bars').replaceChildren();for(const [name,key] of [['价格基础','base_probability'],['正文修正','body_probability'],['语义修正','semantic_probability'],['静态组合','static_probability'],['条件组合','gate_probability']]){const row=document.createElement('div');row.className='row';const label=document.createElement('span');label.textContent=name;const track=document.createElement('div');track.className='track';const fill=document.createElement('div');fill.className='fill';fill.style.width=pct(r[key]);track.append(fill);const value=document.createElement('span');value.textContent=pct(r[key]);row.append(label,track,value);el('bars').append(row)}el('weights').textContent='零修正 / 正文 / 语义权重：'+Object.values(r.weights).map(pct).join(' / ')+'。新闻 '+r.news_count+' 篇。';el('quality').textContent='状态：'+r.gate_status+'。关系抽取仍待独立复核，本次只用原文章集合。';el('outcome').textContent=r.realized_return===undefined?'实际结果尚未显示。':'实际四小时收益：'+pct(r.realized_return);el('news').replaceChildren();for(const n of r.news){const a=document.createElement('article');a.className='article';const h=document.createElement('h3');h.textContent=n.title;const t=document.createElement('small');t.textContent='可用时间 '+n.available;const p=document.createElement('pre');p.textContent=n.excerpt;a.append(h,t,p);el('news').append(a)}if(!r.news.length)el('news').textContent='此窗口没有入选新闻，新闻修正为零。'}catch(e){el('state').textContent='错误：'+e.message}}fetch('/choices').then(x=>x.json()).then(x=>{choices=x;dates();run()});el('stock').onchange=()=>{dates();run()};el('go').onclick=run;el('reveal').onchange=run;</script></html>'''

def main(run,port):
    d=pd.read_pickle(run/'inputs.pkl');d=d[d.month>='2018-09'];raw=pd.read_pickle(ROOT/'work/stock-data/audit/news_index.pkl');raw['key']=raw.archive+'::'+raw.member;raw=raw.set_index('key')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url=urlparse(self.path)
            try:
                if url.path=='/':content=PAGE.encode();typ='text/html; charset=utf-8'
                elif url.path=='/choices':content=json.dumps({s:[str(x) for x in g.start_utc] for s,g in d.groupby('symbol')}).encode();typ='application/json'
                elif url.path=='/predict':
                    args=parse_qs(url.query);sym=args['symbol'][0];start=args['start'][0];r=predict(run,sym,start)
                    row=d[d.symbol.eq(sym)&d.start_utc.eq(pd.Timestamp(start))].iloc[0]
                    ids=sorted([k for k in row.news_record_keys.split('|') if k],key=lambda k:raw.loc[k,'available_utc'],reverse=True)[:2];news=[]
                    for k in ids:
                        meta=raw.loc[k]
                        with zipfile.ZipFile(ROOT/'work/stock-data/raw/news'/meta.archive) as z:text=json.loads(z.read(meta.member)).get('text','')
                        news.append({'title':meta.title,'available':str(meta.available_utc),'excerpt':text[:1400]})
                    r['news']=news
                    if args.get('reveal',['0'])[0]=='1':r['realized_return']=float(row.target_return)
                    content=json.dumps(r,ensure_ascii=False).encode();typ='application/json; charset=utf-8'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',typ);self.end_headers();self.wfile.write(content)
            except Exception as e:
                self.send_response(400);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps({'error':str(e)}).encode())
    print(f'http://127.0.0.1:{port}',flush=True);HTTPServer(('127.0.0.1',port),Handler).serve_forever()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=B/'runs/zero_bias_v1');p.add_argument('--port',type=int,default=8769);a=p.parse_args();main(a.run,a.port)
