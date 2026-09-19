# Prior-work clean reproduction and one-day horizon study

**Status:** preregistered before any classifier fit.

Four-hour primary task: accepted AAPL/AMZN canonical 1,607 rows; March--August
chronological OOF; September--October development; November--February later.
All September-and-later values are `EXPOSED EXPLORATORY HISTORICAL BACKTEST`.
Rows, labels, cutoffs, target association and R1 inputs come unchanged from
`work/stock-data/nextgen_4h/price_v1/features.pkl`. News is accepted full
target-associated text available on or before cutoff. No-news rows remain; a
probability model emits the past class prior and a decision model the past
majority class.

The secondary one-day task uses frozen XNYS sessions and teacher daily charts.
Label is target close > open; cutoff is target open minus five minutes.
`DNEWS_OVERNIGHT` is previous regular close through cutoff; `DNEWS_24H` is
cutoff minus 24h through cutoff. DPRICE uses completed prior sessions only:
returns, prior range/open, five-session close volatility/mean, and history
flags. Jan--Feb is warmup, Mar--Aug OOF, Sep--Oct development, and Nov--Feb
later, labelled `PREREGISTERED_NEW_HORIZON_HISTORICAL_EVALUATION`.

Frozen news-only methods: `PAPER_1G_L1LR`, `PAPER_1G_LINSVM`,
`PAPER_2G_L1LR`, `TFIDF_LR`, `TFIDF_LINSVM`, `TFIDF_RF`, `TFIDF_ADABOOST`,
`TFIDF_KNN`, `TFIDF_XGBOOST`, `W2V_LR`, `W2V_LINSVM`, `W2V_RF`. Vocabulary,
IDF, chi-square, scalers and Word2Vec use only current past rows. Count features
are binary 1/2-grams min_df=3, chi2 top 500. TF-IDF is unigram min_df=3,
raw cap 10,000, chi2 top 500. Word2Vec is train-fold-only: 100d, window 5,
min_count 2, 10 epochs, seed 573, one worker.

LR/SVM C=[.01,.1,1]; RF=300 trees, depth [8,None], leaves [1,5]; AdaBoost=50/100
and [.05,.1]; KNN=[5,15,31], cosine, uniform/distance; XGBoost depth [2,3],
100 trees, .05, seed 573. Unavailable dependencies are recorded, never installed.
March uses C=.1; later months select only earlier issued OOF evidence by BA,
Brier, lower capacity, lexical tie-break. September onward freezes Mar--Aug.
At most three global text configurations per horizon are selected by weaker-stock
monthly BA, macro BA, Brier, capacity and lexical order, then get one joint
news+price fit; no averaging.

Report accuracy, BA, MCC, precision/recall/F1, class recalls, balance, AUC,
Brier when applicable, monthly and no-news metrics. Existing F0/F1/F2 are
read-only. Random 80/20 TFIDF diagnostics are labelled
`RANDOM_SPLIT_DIAGNOSTIC_ONLY_NOT_DEPLOYMENT_ESTIMATE` and never select models.
No new grid, classifier, neural/LLM/market/graph/RL or external dataset run is
permitted after results are observed.
