# Is Quantitative Research Applicable to the IDX Market?

A microstructure study on industry-diversified IDX large-caps, using academic spread/order-flow estimators built from OHLCV, an XGBoost classifier under strict walk-forward validation, and an investigation into whether *Bandarmologi*-style bid-offer following survives on historical data.

The short version: on liquid IDX large-caps, **next-day direction is not predictable from OHLCV-proxied microstructure features.** The pipeline is not broken — a positive control on volatility clears chance decisively. The directional signal is simply absent. This repo documents that search end to end, including the negative result.

---

## Headline result

| Model | Target | OOF AUC | 95% bootstrap CI | Verdict |
|---|---|---|---|---|
| Directional | Next-day up/down | 0.501 | [0.484, 0.517] | Indistinguishable from chance |
| Volatility regime (control) | Next-day high/low range | 0.587 | [0.572, 0.601] | Real signal |

The directional CI straddles 0.50, and it stays there across every neutral-band threshold tested (see the sweep below). The volatility model — same features, same walk-forward, same evaluation, only a different label — clears 0.50 cleanly. That contrast is the whole point: the machinery works, so the absence of directional edge is a finding about the market, not a bug in the code.

---

## What this does

The notebook runs a single linear pipeline over a 15-ticker IDX basket:

1. **Ingest** 5-minute intraday bars (~60 days) and daily bars (~2 years) per ticker from yfinance, normalized into one canonical schema in Asia/Jakarta time.
2. **Validate** every bar against hard invariants (inverted high/low, non-positive prices, duplicate or non-monotonic timestamps, NaNs) and soft ones (OHLC outside the bar range, off-session timestamps), reported separately.
3. **Clean** with an explicit, audited policy: drop hard violations, dedup, forward-fill within ticker, and flag stale bars rather than silently patching them.
4. **Engineer microstructure features** from OHLCV alone — spread, trade signing, order imbalance, liquidity, VWAP position, volatility, noise.
5. **Detect order-flow patterns** with transparent rule-based logic (each with a smooth 0–1 confidence score, not a hard flag).
6. **Aggregate** the intraday feature matrix down to one row per `(ticker, date)` and join daily-context features, so the feature horizon matches the daily label horizon.
7. **Label** next-day close-to-close direction with a configurable neutral band.
8. **Train** XGBoost under `TimeSeriesSplit` walk-forward (no shuffling, each test fold strictly later than its train fold).
9. **Evaluate** with per-fold metrics, a bootstrap CI on out-of-fold AUC, an aggregated confusion matrix, and a decile calibration check.
10. **Stress-test** the conclusion with a neutral-threshold sweep and a volatility positive control, then attribute the volatility model with SHAP.
11. **Emit** a structured JSON signal payload with explicit provenance and uncertainty bounds.

---

## Data

- **Source:** yfinance OHLCV. No real order book, no trade-by-trade tape. Every microstructure quantity here is an *estimator* from bar data.
- **Basket:** 15 tickers spanning 9 sectors — banking (BBCA, BBRI, BMRI, BBNI), telecom (TLKM), conglomerate (ASII), consumer staples (UNVR, ICBP), tobacco (GGRM), mining/energy (ADRO, INCO, ANTM), tech (GOTO), property (BSDE), pharma (KLBF).
- **Why diversified:** an earlier homogeneous 5-bank basket let the model learn ticker identity instead of signal, and left too few effective rows. Spreading across sectors forces the model to see genuinely different microstructure regimes.
- **Excluded:** BREN and DSSA, both removed from LQ45 in April 2026 for high shareholding concentration, which would bias any historical pattern analysis.
- **Resulting matrix:** ~54k intraday bars and 7,170 daily rows (478 trading days × 15 tickers), collapsing to ~6,040 usable labeled rows after the neutral band and the final-day drop. 131 model features.

---

## Methodology

### Feature engineering

Every estimator has a peer-reviewed source. Nothing here is ad hoc.

**Spread (three estimators + consensus).** Roll (1984) from serial covariance of price changes; Corwin–Schultz (2012) from two-bar high/low ranges; Abdi–Ranaldo (2017) from close/high/low. The consensus spread is the per-bar median of the three, with a disagreement measure (dispersion ÷ consensus) carried alongside.

**Trade signing and order flow.** Bar-level Lee–Ready (1991) tick rule to sign each bar, then signed volume, intraday cumulative delta, rolling delta imbalance at three horizons, buyer/seller aggression ratios, and a microprice proxy. This recovers net direction per bar — the best obtainable without a real tape.

**Liquidity and price position.** Amihud (2002) illiquidity, VWAP distance and its rolling z-score, realized volatility (short and long), volume z-score, a variance-ratio noise metric, and a composite liquidity-pressure score combining z-scored spread, illiquidity, and absolute imbalance.

**Rule-based patterns.** Seven detectors — buyer/seller dominance, absorption, distribution, liquidity wall, spread expansion, momentum burst — each emitting a boolean trigger and a logistic confidence in [0, 1]. Running these before the ML step answers a prior question transparently: do the patterns practitioners talk about actually appear at sensible frequencies? They do (dominance and distribution fire on ~14–19% of bars; rarer patterns like momentum bursts on ~1–2%).

### Leakage discipline

This is where the project spends its care, because microstructure work dies on subtle look-ahead.

- Walk-forward only, never shuffled; each test fold is strictly later than its train fold, asserted in code.
- Scale features that encode ticker identity (spread levels, Amihud, realized vol) are normalized to per-ticker **expanding** z-scores using only past data (`shift(1)` before the expanding window).
- Forward returns are verified bar-by-bar to use the *next* day's close, with an assertion that fails the run if the computation drifts.
- Raw price levels and the label columns are explicitly excluded from the feature set.

### Model

XGBoost (`max_depth=4`, `learning_rate=0.05`, subsample/colsample 0.8, `min_child_weight=5`, L1/L2 regularization), 5-fold `TimeSeriesSplit`, binary log-loss objective. Deliberately shallow and regularized given the row count.

---

## Why the negative result holds up

A single low AUC is easy to dismiss as a tuning problem. Two checks close that door.

**Threshold sweep.** Re-labeling and retraining the full walk-forward at neutral bands from 0% to 1.2% never produces a CI lower bound above 0.50. The non-result is not an artifact of where the neutral zone is drawn.

| Neutral band ε | Labeled rows | OOF AUC | 95% CI | Significant? |
|---|---|---|---|---|
| 0.000 | 6,440 | 0.500 | [0.485, 0.515] | No |
| 0.002 | 6,416 | 0.496 | [0.480, 0.510] | No |
| 0.003 | 6,040 | 0.501 | [0.484, 0.516] | No |
| 0.005 | 5,396 | 0.494 | [0.477, 0.510] | No |
| 0.008 | 4,654 | 0.497 | [0.480, 0.514] | No |
| 0.012 | 3,738 | 0.485 | [0.464, 0.504] | No |

**Positive control.** Volatility clusters — today's range predicts tomorrow's, one of the most robust facts in finance. Pointing the identical pipeline at a next-day high-volatility label lifts OOF AUC to 0.587 [0.572, 0.601]. The feature engine, the walk-forward, and the evaluation are all sound. Direction just isn't there.

**SHAP confirms where the information lives.** On the volatility model, the top attributions are realized volatility (short and long), pattern confidence scores (liquidity wall, momentum burst), signed volume, and spread expansion. The order-flow proxies *do* carry information — about volatility, not about direction.

---

## Findings

1. IDX is efficient at the surface level for liquid large-caps. No next-day directional edge survives honest validation on OHLCV-only features.
2. Competing on names where institutions are already active, using only surface data, is a losing setup. The edge, if any, is not in the bar data.
3. Without real bid-offer flow — an actual order book, not OHLCV proxies — there is no extractable directional alpha from this approach.
4. *Bandarmologi* on historical bid-offer footprints decays. By the time a pattern is visible in the historical record, the large player who created it has already exited. Volatility is the one regime the surface features genuinely anticipate.

---

## Outputs

`output/kspm_signals.json` — one signal per ticker for the latest session, plus a provenance block and a model card. The model card states plainly that the directional model was *tested and is not predictable*, and that the volatility-regime model is the only field with demonstrated signal. The JSON is structured so a downstream narrative/LLM layer can consume it with the uncertainty already attached.

Each signal carries the imbalance read, delta read, liquidity condition, buyer aggression, detected patterns, and the volatility-regime probability — all traceable to a named feature and threshold.

---

## Running it

```bash
pip install yfinance xgboost plotly shap joblib lightgbm pandas numpy scikit-learn matplotlib
jupyter notebook main.ipynb   # run cells top to bottom
```

All tunables (basket, intervals, label horizon, neutral threshold, XGBoost params) live in a single `Config` dataclass at the top, so the whole investigation re-runs against a different basket or horizon by editing one object. Re-running requires live yfinance access; intraday history is rolling (~60 days), so exact bar counts will differ from those reported here.

---

## Caveats

- **Everything microstructural is proxied.** Spread, imbalance, delta, and aggression are estimators from OHLCV, not observed order-book data. Read the conclusions as statements about *this proxy set*, not about what a real tape would reveal.
- **Bar-level tick rule loses intrabar distribution.** On 5-minute bars it recovers net direction only.
- **The volatility result is a control, not a strategy.** It validates the pipeline; it is not packaged or backtested as a tradable system, and AUC ≈ 0.59 is modest.
- This is an honest research log, not a product. The interesting part is the discipline around the negative result.

---

## References

- Roll, R. (1984). *A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market.* Journal of Finance.
- Corwin, S. A., & Schultz, P. (2012). *A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices.* Journal of Finance.
- Abdi, F., & Ranaldo, A. (2017). *A Simple Estimation of Bid-Ask Spreads from Close, High, and Low Prices.* Review of Financial Studies.
- Lee, C. M. C., & Ready, M. J. (1991). *Inferring Trade Direction from Intraday Data.* Journal of Finance.
- Amihud, Y. (2002). *Illiquidity and Stock Returns: Cross-Section and Time-Series Effects.* Journal of Financial Markets.
