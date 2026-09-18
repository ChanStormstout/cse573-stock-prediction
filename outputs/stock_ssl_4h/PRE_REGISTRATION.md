# Finite past-only price self-supervision experiment

Registered before new results. Main task: unchanged 1,607 AAPL/AMZN four-hour direction windows. All evaluation periods are exposed exploratory backtests. No claim that augmentation creates independent market observations. No news annotation or browser submission in this experiment.

## Inputs and splits

Use 48 completed scheduled five-minute bars before the original cutoff, allowing exchange-session boundaries with an explicit flag. Require all needed OHLC bars and preceding close; otherwise exact float64 fallback to existing new-protocol own-price LR. Six channels: log close/open, log high/low, close position within bar, log open/previous scheduled close, new-session flag, NY time fraction. The activity column is excluded because its semantics are unresolved. All transforms fit past bars only. Never interpolate a missing interval.

For self-supervision use every twelfth scheduled bar as a sequence endpoint, independently of four-hour labels, from January2018 through the applicable training boundary. Both stocks share the encoder; their downstream LR models remain separate. All sequence ends must precede the evaluation cutoff and September-onward encoders freeze using August-or-earlier data. March–May forward predictions support C selection; June/July/August are outer training-period diagnostics. September–October development and November-onward later reported separately.

## Four cells

- B: saved P_own_lr from goal60, same task and global past-only C selection.
- RAW: B input features plus flattened 48x6 sequence reduced by training-only PCA16, then LR.
- RANDOM: same price features plus frozen randomly initialized small encoder, then LR.
- SSL: identical encoder initialization, masked reconstruction pretraining, frozen encoder, then identical LR.

The matched pretraining contrast is SSL vs RANDOM. RAW tests whether learned encoding exceeds a simple same-history representation. RANDOM is explicitly random frozen features, not a supervised-from-scratch encoder experiment. Encoder is three Conv1d layers,16 channels,kernel3,dilations1/4/16,GELU; mean and last-state concatenate to32 features. Reconstruction head is pointwise16→4. One mask-indicator input; mask25% time positions of all four price channels, preserve time/session channels. Optimize masked smooth-L1 only, AdamW lr.001,weight_decay.01,batch128,max20epochs,patience4. Seeds573/574/575, report all and mean probabilities, no best seed selection. CPU default.

Self-supervised early stopping uses last15% of past endpoint dates, purging training sequences that overlap the first validation input interval. Scaler fits internal training bars; evaluate a fixed validation mask. Reinitialize and refit all allowed past unlabeled data for the chosen epoch count with a scaler fit to unique used past bars. No stock direction labels enter encoder training. Save reconstruction loss and a zero-standardized-price prediction comparator; reconstruction improvement is not directional success.

LR C=.01/.1/1 globally selected across stocks from earlier monthly issued OOF probabilities, weakest-stock monthly BA then macro BA then smaller C; March defaults .1. Each encoder and PCA fit without future data. Missing-sequence fallback is applied before selection. Threshold .5. No fusion or threshold search in this round. At most this architecture; no grid expansion after results.

## Acceptance and outputs

Report stock/phase/month BA,MCC,Brier,constant/up-prediction rates,seed variability,coverage,common-error fixes/new errors,paired1/5-day intervals. A promising matched effect requires weakest-stock June–August meanBA≥+1pp,neither stock loses>1pp,and macro gain in at least2/3months; this is an exploration gate,not significance. Also show RAW comparison. Existing evaluation never becomes a new holdout.

Save source/data/protocol hashes, sample keys and latest input timestamps, actual gradient/trainable parameter/epoch/device/time evidence, checkpoints and reload parity privately. Public reports contain no raw OHLC/news or weights. Reject changed resume fingerprints; preserve previous experiments. Run repository refresh/check,commit,push,verify remote.
