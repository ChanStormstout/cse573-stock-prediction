from dataclasses import dataclass,asdict
@dataclass(frozen=True)
class Config:
 version: str='four_hour_v2_v1'
 seed: int=573
 lambdas: tuple=(.2,.02,.002)
 inner_start: str='2018-03'
 outer_months: tuple=('2018-06','2018-07','2018-08')
 final_month: str='2018-09'
 max_price_fits: int=500
 availability_lag: str='0min'
 quote_scope: str='rth'
 bootstrap_draws: int=10000
 def __post_init__(self):
  # This registered experiment supports exactly these settings; no silently ignored fields.
  expected={'version':'four_hour_v2_v1','seed':573,'lambdas':(.2,.02,.002),'inner_start':'2018-03','outer_months':('2018-06','2018-07','2018-08'),'final_month':'2018-09','max_price_fits':500,'availability_lag':'0min','quote_scope':'rth','bootstrap_draws':10000}
  expected['availability_lag']=self.availability_lag
  if self.availability_lag not in ['0min','5min']:raise ValueError('Unsupported availability lag')
  if asdict(self)!=expected:raise ValueError('Unregistered configuration; create a separately declared experiment')
