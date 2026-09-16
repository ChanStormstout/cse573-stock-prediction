from collections import Counter
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
class DeterministicTfidf(TransformerMixin,BaseEstimator):
 def __init__(self,max_features=100,min_df=2):self.max_features=max_features;self.min_df=min_df
 def fit(self,X,y=None):
  proto=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True);analyze=proto.build_analyzer();tf=Counter();df=Counter()
  for text in X:
   tokens=analyze(text);tf.update(tokens);df.update(set(tokens))
  eligible=[t for t in tf if df[t]>=self.min_df];ranked=sorted(eligible,key=lambda t:(-tf[t],t))[:self.max_features]
  if not ranked:raise ValueError('No eligible training vocabulary')
  self.vocabulary_={t:i for i,t in enumerate(sorted(ranked))};self.ranking_=[{'term':t,'tf':tf[t],'df':df[t]} for t in ranked]
  self.vectorizer_=TfidfVectorizer(vocabulary=self.vocabulary_,ngram_range=(1,2),sublinear_tf=True).fit(X);return self
 def transform(self,X):return self.vectorizer_.transform(X)
 def get_feature_names_out(self,input_features=None):return self.vectorizer_.get_feature_names_out()
