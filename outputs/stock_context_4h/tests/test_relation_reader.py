"""Regression tests for the pre-Qwen relation reader contracts."""
from __future__ import annotations
import json
from outputs.stock_context_4h.build_news_pairs import missing_hash,node,pair_id
from outputs.stock_context_4h.run_pair_reader import validate,ENUM
def row(current='2020-01-02T00:00:00Z',past='2020-01-01T00:00:00Z'):
 return {'current_available_utc':current,'past_available_utc':past,'current_sentences':[{'id':'C0','text':'Target is $225 in 2024.'}],'past_sentences':[{'id':'P0','text':'Target was 200.'}]}
def obj(**kw):
 d={'pair_relation':ENUM[0],'current_evidence_ids':['C0'],'past_evidence_ids':['P0'],'changes':[]};d.update(kw);return d
def main():
 assert node('AAPL','x')!=node('AMZN','x')
 assert pair_id('AAPL','c','p')!=pair_id('AMZN','c','p')
 assert all(missing_hash(x) is None for x in (None,'','nan','None'))
 assert validate(obj(changes=None),row())==['changes_not_list']
 assert validate(obj(changes='bad'),row())==['changes_not_list']
 assert 'current_evidence_ids_bad_id' in validate(obj(current_evidence_ids=['C9']),row())
 assert 'old_value_ungrounded_numeric' in validate(obj(changes=[{'old_value':'2','new_value':None}]),row())
 assert validate(obj(changes=[{'old_value':'$225','new_value':'225'}]),row())==[]
 assert 'time_order' in validate(obj(),row('2020-01-01T00:00:00Z','2020-01-01T00:00:00Z'))
 assert json.dumps([{'target':'AAPL','count':1}])
 print('PASS relation-reader unit contracts')
if __name__=='__main__':main()
