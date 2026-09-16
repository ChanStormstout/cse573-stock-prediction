"""Execute registered N00--N03 and separate N05; immutable run directories."""
from core import *
import argparse, os, time, shutil, datetime
import joblib
from sklearn.linear_model import LogisticRegression

def choose_base(cv,symbol,kind,month):
    a=[r for r in cv if r['symbol']==symbol and r['kind']==kind and '2018-03'<=r['month']<month]
    if not a:return .1,[]
    v=pd.DataFrame(a).groupby('C')[['BA','Brier']].mean().reset_index()
    return float(v.sort_values(['BA','Brier','C'],ascending=[False,True,True]).iloc[0].C),sorted({r['month'] for r in a})

def choose_res(cv,symbol,kind,month):
    a=[r for r in cv if r['symbol']==symbol and r['kind']==kind and r['month']<month]
    if not a:return .1,[]
    v=pd.DataFrame(a).groupby('C')[['BA','Brier','base_Brier']].mean().reset_index()
    v=v[v.Brier<=v.base_Brier+.002]
    return (None if v.empty else float(v.sort_values(['BA','Brier','C'],ascending=[False,True,True]).iloc[0].C)),sorted({r['month'] for r in a})

def choose_router(cv,month):
    a=[r for r in cv if r['month']<month and r['trained']]
    if not a:return 10.,.25,[]
    t=pd.DataFrame(a); candidates=[]
    for (alpha,rho),g in t.groupby(['alpha','rho']):
        stats=g.groupby('symbol')[['BA','Brier','static_Brier']].mean()
        if len(stats)==2 and (stats.Brier<=stats.static_Brier+.002).all():
            candidates.append((float(stats.BA.mean()),float(stats.Brier.mean()),float(rho),float(alpha)))
    if not candidates:return 10.,0.,sorted(t.month.unique())
    ba,br,rho,alpha=sorted(candidates,key=lambda x:(-x[0],x[1],x[2],-x[3]))[0]
    return alpha,rho,sorted(t.month.unique())

def main(out,news_bias='free'):
    out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();(out/'code_snapshot').mkdir()
    start=time.monotonic();d,keys,emb=prepare_inputs();d.to_pickle(out/'inputs.pkl')
    inputs=[OLD/'prepared'/s for s in ['data.pkl','articles.npz','embedding_inputs.json','manifest.json']]+[ROOT/'work/stock-data/audit/news_index.pkl',B/'PROTOCOL.md',B/'core.py',B/'run.py']
    inputs += [OLD/'run.py',OLD/'runs/v1/predictions.csv',OLD/'runs/v1/oof.csv',B.parent/'stock_four_hour_v2/text_vectorizer.py']
    if news_bias=='zero':inputs.append(B/'ZERO_BIAS_PROTOCOL.md')
    dump(out/'config.json',{'news_bias':news_bias,'post_diagnostic_variant':news_bias=='zero'})
    sources={str(p):sha(p) for p in inputs};dump(out/'sources.json',sources)
    for p in list(B.glob('*.py'))+[B/'PROTOCOL.md']:shutil.copy2(p,out/'code_snapshot'/p.name)
    price_ledgers=[];full_ledgers=[];base_cv=[];res_cv=[];gate_cv=[];selection=[];fits=[];static_grids=[];all_rows=[]
    def save_fit(obj,name,tr,ev,layer,details):
        temporal_check(tr,ev);path=out/'models'/f'{name}.joblib';joblib.dump(obj,path)
        fits.append({'name':name,'layer':layer,'pid':os.getpid(),'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'train_keys':tr.key.tolist(),'eval_keys':ev.key.tolist(),'max_label_end':str(tr.end_utc.max()),'evaluation_cutoff':str(ev.cutoff_utc.min()),'model_hash':sha(path),**details})
        return path
    def fit_branches(month,final=False):
        rows=[];bundles={}
        for sym,g in d.groupby('symbol'):
            tr=g[g.month<month];ev=g[g.month>=month] if final else g[g.month==month]
            temporal_check(tr,ev);row=ev.copy();bundle={}
            for kind in ['price','title']:
                c,months=choose_base(base_cv,sym,kind,month);tf=Transform(kind).fit(tr,keys,emb)
                a=tf.transform(tr,keys,emb);b=tf.transform(ev,keys,emb)
                for candidate in ([c] if final else CS):
                    clock=time.monotonic();model=LogisticRegression(C=candidate,solver='liblinear',max_iter=3000,random_state=573,tol=1e-7).fit(a,tr.label)
                    if model.n_iter_.max()>=3000:raise RuntimeError('Base failed to converge')
                    obj={'transform':tf,'model':model};p=model.predict_proba(b)[:,1]
                    path=save_fit(obj,f'{sym}_{month}_{kind}_{candidate}',tr,ev,'base',{'C':candidate,'iterations':model.n_iter_.tolist(),'seconds':time.monotonic()-clock,'train_score':score(tr.label,model.predict_proba(a)[:,1])})
                    check=joblib.load(path);np.testing.assert_allclose(p,check['model'].predict_proba(check['transform'].transform(ev,keys,emb))[:,1],atol=1e-12,rtol=0)
                    if not final:base_cv.append({'symbol':sym,'month':month,'kind':kind,'C':candidate,**score(ev.label,p)})
                    if candidate==c:row[kind]=p;bundle[kind]=obj
                selection.append({'symbol':sym,'month':month,'kind':kind,'C':c,'selection_months':months})
            row['prior']=float(tr.label.mean());row['constant']=.5
            past=pd.concat(price_ledgers,ignore_index=True) if price_ledgers else row.iloc[:0]
            history=past[(past.symbol==sym)&(past.month<month)]
            ready=len(history)>=60 and history.day.nunique()>=20 and history.label.nunique()==2
            row['residual_ready']=ready
            for kind in ['body','semantic']:
                tf=NewsTransform(kind).fit(history,keys,emb) if ready else None
                x=StandardScaler().fit(history[PRICE]) if ready else None
                a=tf.transform(history,keys,emb) if ready else None;b=tf.transform(ev,keys,emb) if ready else None
                for joint in [False,True]:
                    name='joint_'+kind if joint else kind;c,months=choose_res(res_cv,sym,name,month);selected=None;delta=np.zeros(len(ev));raw_delta=delta.copy()
                    if ready:
                        for candidate in ([c] if final and c is not None else CS if not final else []):
                            clock=time.monotonic();m=OffsetModel(candidate,joint,news_bias=news_bias).fit(a,history.has_news.to_numpy(),history.price,history.label,x.transform(history[PRICE]) if joint else None)
                            dd,raw=m.correction(b,ev.has_news.to_numpy(),x.transform(ev[PRICE]) if joint else None)
                            pp=add_delta(row.price,dd,np.ones(len(ev),bool) if joint else ev.has_news)
                            obj={'transform':tf,'model':m,'price_scaler':x if joint else None}
                            path=save_fit(obj,f'{sym}_{month}_{name}_{candidate}',history,ev,'residual',{'C':candidate,'iterations':m.iterations_,'seconds':time.monotonic()-clock,'offset_source':'earlier_month_price_oof','offset_keys':history.key.tolist(),'features':a.shape[1],'coefficient_norm':float(np.linalg.norm(m.coef_))})
                            loaded=joblib.load(path);dd2,_=loaded['model'].correction(loaded['transform'].transform(ev,keys,emb),ev.has_news.to_numpy(),loaded['price_scaler'].transform(ev[PRICE]) if joint else None)
                            np.testing.assert_allclose(dd,dd2,atol=1e-12,rtol=0)
                            if not final:res_cv.append({'symbol':sym,'month':month,'kind':name,'C':candidate,**score(ev.label,pp),'base_Brier':score(ev.label,row.price)['Brier']})
                            if candidate==c:selected=obj;delta=dd;raw_delta=raw
                    bundle[name]=selected;row[name]=add_delta(row.price,delta,np.ones(len(ev),bool) if joint else ev.has_news)
                    if not joint:row['db' if kind=='body' else 'ds']=delta;row['raw_b' if kind=='body' else 'raw_s']=raw_delta
                    selection.append({'symbol':sym,'month':month,'kind':name,'C':c,'selection_months':months,'status':'FIT' if selected else 'BASE_FALLBACK','training_rows':len(history)})
            if len(history):
                cal=Calibration().fit(history.price,history.label)
                save_fit(cal,f'{sym}_{month}_calibration',history,ev,'calibration',{'iterations':cal.iterations_,'coef':cal.coef_.tolist(),'offset_source':'earlier_month_price_oof'})
            else:cal=Calibration();cal.coef_=np.array([1.,0.])
            bundle['calibration']=cal;bundles[sym]=bundle;rows.append(row)
        current=pd.concat(rows,ignore_index=True)
        history=pd.concat(full_ledgers,ignore_index=True) if full_ledgers else current.iloc[:0]
        complete=history[history.residual_ready.astype(bool)]
        alpha,rho,months=choose_router(gate_cv,month);routers={}
        for a in ([alpha] if final else [1.,10.,100.]):
            router=Router(a).fit(complete);routers[a]=router
            if router.ready_:save_fit(router,f'BOTH_{month}_router_{a}',complete,current,'router',{'alpha':a,'support':router.support_,'stock_support':router.stock_support_,'source':'earlier_month_complete_chain_oof'})
        selection.append({'symbol':'BOTH','month':month,'kind':'router','alpha':alpha,'rho':rho,'selection_months':months})
        output=[]
        for sym,row in current.groupby('symbol',sort=True):
            row=row.copy();hist=complete[complete.symbol==sym];w,grid=select_static(hist);bundles[sym].update(static=w,router=routers[alpha],rho=rho)
            selection.append({'symbol':sym,'month':month,'kind':'static','weights':w,'selection_months':sorted(hist.month.unique())})
            static_grids.extend([{'symbol':sym,'month':month,**x} for x in grid]);row['static']=combine(row,w)
            for a,router in routers.items():
                for strength in ([rho] if final else [0.,.25,.5]):
                    ww,reason=router.weights(row,w,strength);pp=combine(row,ww)
                    if not final:gate_cv.append({'symbol':sym,'month':month,'alpha':a,'rho':strength,'trained':router.ready_,**score(row.label,pp),'static_Brier':score(row.label,row.static)['Brier']})
                    if a==alpha and strength==rho:
                        row['gate']=pp;row['gate_reason']=reason
                        for i,n in enumerate(['zero','body','semantic']):row['weight_'+n]=ww[:,i]
            for method in ['price','static','gate']:row[method+'_cal']=bundles[sym]['calibration'].predict(row[method])
            row['phase']='frozen' if final else 'forward';output.append(row)
        result=pd.concat(output,ignore_index=True)
        if not final:
            price_ledgers.append(result.copy());full_ledgers.append(result.copy())
        all_rows.append(result)
        print(month,'COMPLETE','actual_fits',len(fits),'router',routers[alpha].ready_,'selected',alpha,rho,flush=True)
        return result,bundles

    for month in pd.period_range('2018-02','2018-08',freq='M').astype(str):fit_branches(month)
    frozen,bundles=fit_branches('2018-09',True)
    oof=pd.concat(full_ledgers,ignore_index=True)
    # Mechanism selection happens exclusively on chronological June--August outputs.
    outer=oof[oof.month>='2018-06'];summaries=[]
    for method in ['title','static','gate','static_cal','gate_cal']:
        detail={s:mean_months(g,g[method]) for s,g in outer.groupby('symbol')}
        base={s:mean_months(g,g.title) for s,g in outer.groupby('symbol')}
        eligible=all(detail[s]['Brier']<=base[s]['Brier']+.002 and detail[s]['BA']>=base[s]['BA']-.01 for s in detail)
        wins=sum(np.mean([score(g.label,g[method])['BA']-score(g.label,g.title)['BA'] for _,g in m.groupby('symbol')])>0 for _,m in outer.groupby('month'))
        gain=float(np.mean([detail[s]['BA']-base[s]['BA'] for s in detail]));passed=eligible and wins>=2 and gain>=.01
        summaries.append({'method':method,'detail':detail,'eligible':eligible,'positive_months':wins,'macro_BA_gain':gain,'continuation_gate':passed})
    candidates=[x for x in summaries if x['continuation_gate']]
    chosen=sorted(candidates,key=lambda x:(-x['macro_BA_gain'],np.mean([s['Brier'] for s in x['detail'].values()]),['static','static_cal','gate','gate_cal'].index(x['method'])))[0]['method'] if candidates else 'title'
    dump(out/'system_selection.json',{'selected_method':chosen,'selection_months':['2018-06','2018-07','2018-08'],'candidates':summaries,'rule':'budget gate; not independent significance'})
    full=pd.concat(all_rows,ignore_index=True);full['selected_system']=full[chosen]
    original=pd.read_csv(OLD/'runs/v1/predictions.csv')
    old=original[['key','price','title','body','semantic','integrated']].rename(columns={k:'old_'+k for k in ['price','title','body','semantic','integrated']})
    full=full.merge(old,on='key',how='left',validate='one_to_one')
    f=full[full.phase=='frozen'];errors={name:float(np.max(np.abs(f[name]-f['old_'+name]))) for name in ['price','title']}
    assert max(errors.values())<1e-12,errors
    dump(out/'baseline_reproduction.json',errors)
    for sym,bundle in bundles.items():
        bundle.update(symbol=sym,selected_method=chosen,embedding_sha256=sha(OLD/'prepared/articles.npz'),training_end='2018-08-31',source_hashes=sources)
        path=out/f'{sym}_system.joblib';joblib.dump(bundle,path)
        x=d[(d.symbol==sym)&(d.month>='2018-09')].drop(columns=['label','target_return'])
        v=predict_bundle(joblib.load(path),x,keys,emb).set_index('key');target=f[f.symbol==sym].set_index('key').loc[v.index]
        for name in ['price','title','body','semantic','joint_body','joint_semantic','static','gate','price_cal','static_cal','gate_cal']:
            np.testing.assert_allclose(v[name],target[name],atol=1e-12,rtol=0);np.testing.assert_array_equal(v[name]>=.5,target[name]>=.5)
    full.to_pickle(out/'predictions.pkl');full.drop(columns=['stem_body','text','article_keys']).to_csv(out/'predictions.csv',index=False)
    oof.to_pickle(out/'oof.pkl');pd.DataFrame(base_cv).to_csv(out/'base_cv.csv',index=False);pd.DataFrame(res_cv).to_csv(out/'residual_cv.csv',index=False);pd.DataFrame(gate_cv).to_csv(out/'gate_cv.csv',index=False)
    dump(out/'fits.json',fits);dump(out/'selection.json',selection);dump(out/'static_grid.json',static_grids)
    check_hashes(sources)
    dump(out/'status.json',{'status':'CORE_COMPLETE','horizon':'4h','rows':len(d),'frozen_rows':len(f),'actual_fits':len(fits),'fit_layers':pd.Series([a['layer'] for a in fits]).value_counts().to_dict(),'seconds':time.monotonic()-start,'pid':os.getpid(),'unlabeled_reload_verified':True,'selected_method':chosen,'interpretation':'Exploratory exposed historical replay','source_hashes_unchanged':True})
    print('FROZEN RESULTS',flush=True)
    for sym,g in f[f.split=='test'].groupby('symbol'):
        print(sym,{k:score(g.label,g[k])['BA'] for k in ['title','body','semantic','static','gate','static_cal','gate_cal']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--news-bias',choices=['free','zero'],default='free');a=p.parse_args();main(a.output,a.news_bias)
