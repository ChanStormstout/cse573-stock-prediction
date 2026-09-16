from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(r):
    out=r/'figures';out.mkdir(exist_ok=False);lc=pd.read_csv(r/'M02/learning_curve.csv');fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,stock in zip(axes,['AAPL','AMZN']):
        g=lc[lc.symbol.eq(stock)].groupby('fraction')[['train_balanced_accuracy','balanced_accuracy']].mean();ax.plot(g.index,g.train_balanced_accuracy*100,'o-',label='Training');ax.plot(g.index,g.balanced_accuracy*100,'o-',label='Forward validation');ax.axhline(50,color='gray',ls='--');ax.set(title=stock,xlabel='Fraction of complete training days',ylabel='Balanced accuracy (%)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Fixed L2 / 100 words / C=0.1; averages over months and date-sampling seeds');fig.tight_layout();fig.savefig(out/'learning_curve.png',dpi=170);plt.close(fig)
    g=pd.read_csv(r/'evaluation/monthly.csv');g=g[(g.mechanism=='M08')&g.method.isin(['fusion','price'])];fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,stock in zip(axes,['AAPL','AMZN']):
        t=g[g.symbol.eq(stock)].pivot(index='month',columns='method',values='balanced_accuracy');t.mul(100).plot(ax=ax,marker='o');ax.axhline(50,color='gray',ls='--');ax.set(title=stock,ylabel='Balanced accuracy (%)');ax.tick_params(axis='x',rotation=45)
    fig.suptitle('Frozen price vs fusion: exposed historical months; February is one day');fig.tight_layout();fig.savefig(out/'fusion_monthly.png',dpi=170);plt.close(fig)
    g=pd.read_csv(r/'M09/summary.csv');g=g[(g.period=='test')&g.method.str.startswith('PPO_')];fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,stock in zip(axes,['AAPL','AMZN']):
        t=g[g.symbol.eq(stock)]
        for name,s in t.groupby('method'):ax.plot(s.cost_bps,s.net_return*100,'o-',label=name)
        ax.axhline(0,color='black',ls='--',label='Cash');ax.set(title=stock,xlabel='One-way cost (basis points)',ylabel='Net return (%)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Same PPO weights, exposed replay; all three seeds');fig.tight_layout();fig.savefig(out/'rl_costs.png',dpi=170);plt.close(fig)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run(a.run)
