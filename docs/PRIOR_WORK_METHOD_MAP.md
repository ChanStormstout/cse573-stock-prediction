# Prior-work method map

| Source | Unit / horizon | Representation / learner | Caveat | Reproduced here |
|---|---|---|---|---|
| Alostad & Davulcu, *Directional Prediction of Stock Prices using Breaking News on Twitter* | Breaking-news direction task | Sparse text, chi-square, sparse LR, Twitter breakout component | Different corpus/task; supplied PDF is currently access-gated | Binary 1/2-grams, train-only chi2 and L1 LR; no Twitter breakout. |
| [apoorvt95](https://github.com/apoorvt95/CSE573-Directional-stock-prediction) | AAPL/AMZN course data | BERT+BiLSTM/classical | README reports accuracy/F-score/ROC, not time-safe BA. `akshayk1003` mirrors it. | Classical matrix only. |
| [baani-khurana](https://github.com/baani-khurana/SWM-stock-prediction) | Course tweets/news, 1h/4h branches | Notebook pipeline; historical chronological 4h DistilBERT branch | Processed arrays external and not copied | Classical controls; frozen FinBERT comparator. |
| [Hindawi91 2023](https://github.com/Hindawi91/CSE573_Directional_Stock-_Prediction_Using_Online_News) | Course data; supports 30m/1h/4h/1d, defaults 1d | TF-IDF/Word2Vec/BERT x LR/RF/AdaBoost/SVC/KNN/voting | TF-IDF fit before random split, so reported random values are not deployment-clean | Bounded fold-trained matrix, no copied code/voting. |

## How F1 relates to paper-style control

Both use sparse words, chi-square, logistic regression and UP/DOWN labels. F1
is not a reproduction: it adds prices, uses L2 LR, has different selection,
corpus, stocks, sample unit and strict chronological cutoffs, and lacks Twitter
breakout selection. New rows are `PAPER_STYLE_CLEAN_REPRODUCTION`.
