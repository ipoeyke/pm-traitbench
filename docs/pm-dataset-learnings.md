# Learnings: how the market processes are built

Notes kept while designing stage 2 (market) of the dataset pipeline, for my own reference. Not part of the repo.

## The template

Every series is a daily random walk in the right variable:

```
change = drift + exposure_to_common_shock + own_noise + pull_to_round_level + event_jump
```

The "right variable" is what practitioners quote and what moves proportionally: log price for equities, commodities and FX (a 2% move is a 2% move at any price level); yield in percent for curves (moves are in bp, not in % of the yield); log spread for credit (spreads widen proportionally, HY at 300 moves more bp than AA at 50). Picking that variable is most of the modelling.

## Common shock

One standard normal `z` per day, shared by all families. A family with correlation `c` to it gets `c·z + sqrt(1 - c²)·e`, where `e` is its own noise. Why that form: variance is `c² + (1 - c²) = 1`, so the total shock has unit variance, and its correlation with `z` is `c`. Multiply by the family's daily vol `sigma` and you have a shock with the right size and the right correlation in one line. That is also why the loadings could be sourced as correlations: they are the same number.

## Drift

Not a separate term: `z` has mean `driver_mean` in a regime, so a family's mean change is `c·sigma·driver_mean`. Equities (c = 1 by definition) fall in risk-off; anything with negative `c` (credit spreads, gold) rises. One number sets the direction and size for every family consistently.

## Equities: factor model

Sharpe's single-index model, with the Campbell et al. decomposition into market and own variance.

```
r_i = beta_i · sigma_m · z + sigma_i · t_i
```

Two independent shocks: market and own. A name's variance is `beta² sigma_m² + sigma_i²`; with 0.16 and 0.25 that is about 30% vol, a normal single name. A sector factor was in an earlier draft and was dropped: no bias fingerprint measures sector co-movement, and its vol could not be sourced; the sector tag stays as a static attribute for exclusion rules. Own noise is Student-t because single-name daily returns have fat tails (kurtosis 20 for the index; names worse). Price: `p_t = p_{t-1}·exp(r)`, so the price can never go negative and returns compound.

## Rates: level and slope

Nelson-Siegel curve, in the Diebold and Li (2006) dynamic version. Yields at all maturities move together mostly in parallel (level), and the second thing that happens is the curve steepening or flattening (slope). Those two factors explain about 95% of curve variance in every study, so:

```
y_k = L + w_k · S
```

with tenor weights `w = (-0.5, -0.15, 0.15, 0.5)`: a positive slope shock lowers the 2Y and raises the 30Y. `L` and `S` each follow the template. Yields are already in percent, so no exponent; a floor at zero because negative yields are outside this build.

## Credit: base × factor × issuer

A multiplicative spread model, the same idea as a one-factor model on log spreads.

```
s_i = base_band · F^(h_i) · exp(x_i)
```

Log form so moves are proportional. `F` is the market-wide credit factor following the template with negative `c` (widens when equities fall). Asymmetry: scale the shock by 1.1 when it widens and by 1/1.1 when it tightens, which produces the positive skew in the data. `x_i` is issuer-specific and mean-reverting (AR(1)), because an issuer's cheapness to its band does not wander off forever. Bond price from spread by duration: a bond of duration `D` loses about `D × (change in yield)`, so `p_t = p_{t-1}·(1 - D·(Δs + Δy)/10000)` with the change in bp; the treasury leg `Δy` comes from the currency's curve at 5Y.

## Commodities: spot plus a curve shape

Spot follows the template with a group shock (energy, metals) mixed with own noise: `sqrt(share)·e_g + sqrt(1 - share)·t_i` again keeps unit variance. The futures curve is not simulated tenor by tenor; each point is spot times a shape: `M_k = spot·(1 + slope·k/12)`. Slope negative = backwardation (later months cheaper), the average roll yield from Erb and Harvey (2006). A little per-tenor noise so the curve is not a ruler. Front month = `price`.

## FX: currencies, not pairs

Simulate one log value per currency against USD with the template. A pair is a ratio of two currency values, so EURGBP = EURUSD / GBPUSD by construction and no triangle can disagree. Zero drift: FX has no risk premium worth modelling here.

## Round-level pull

Ornstein-Uhlenbeck, the standard mean-reversion process: `-kappa·(log p - log round(p))` pulls toward the nearest grid level with half-life `ln 2 / kappa` (about 14 days at 0.05). Applied only in the range regime, which is what makes prices retest the same numbers and gives anchoring something to anchor on.

## Event jump

Added on the event date after the diffusion step: `surprise × jump_size` in the family's own variable (log return, bp, relative spread).

## Summary

The derivation is: choose the variable that moves proportionally, take the family's standard factor structure from the literature, express every shock as `c·z + sqrt(1 - c²)·e` so correlations are the config numbers, let the regime enter only through the mean and scale of `z`, and add the two extras the dataset needs (round-level pull, event jumps).

## Consensus and positioning

Two biases need a signal that is not the PM's own: herding (weight on the street's view) and, through positioning, the contrarian preferences. The market layer supplies a "what everyone else thinks" series per instrument that follows price with a lag like sell-side research, changes in steps because views are revised on days, and sometimes flips for no price reason so a herding PM can be caught following a flip. Positioning is the second series: how crowded the trade is, from a weekly report.

The one ingredient is trend. For each instrument and day, take the trailing 20-day return and divide by the size of a normal 20-day move (the family's vol scaled to 20 days); clip to [-1, 1]. `+1` means up a lot lately, `0` flat, `-1` down a lot. For a curve, use the 10Y yield change with the sign flipped, since falling yields are the bullish direction for a bond investor; the same flip applies to a credit issuer's spread, since tightening is bullish for the holder.

The street score is an exponential moving average of trend, updated on revision days. An EMA is a running average that weights recent values more:

```
score_new = score_old + lam · (trend - score_old)
```

`lam` between 0 and 1 says how far the score moves toward today's trend at each update; with `d` trading days since the previous update, `lam = 1 - 2^(-d/W)` makes the half-life `W` days whatever the update cadence, so after `W` days half of an old view has been replaced. With `W = 20` the street trails price by about a month. Updates happen only on the instrument's event days, on one fixed weekday (the revision day) and on flip days; every other day the score is held, which is what makes the series step instead of drift.

On a flip day the score is set to `-sign(score_old) · 2 · threshold`, the opposite view, past the label boundary, not at full conviction, and a `consensus_flip` row goes into the calendar so the flip is dated. Two flips per instrument per year. This is the herding trap: no price move justified it, so a PM who follows it is following the crowd, not information.

Labels: `street_view` is overweight above 0.25, underweight below -0.25, else neutral. They exist so a transcript sentence ("street's gone underweight") matches a row.

Positioning is the same idea, slower and weekly. On each Friday report date, `target = 50 + 40 · tanh(2 · trend60)` maps the 60-day trend to a percentile in [10, 90] with a soft saturation, then the percentile moves toward the target with the same day-based EMA at a 60-day half-life, about twelve weekly reports. Held between reports, so positioning builds over a quarter and unwinds over several reports; after a street flip it stays crowded for weeks. Labels: crowded_short below 30, crowded_long above 70; the band sits at half the target's travel from neutral, comparable to the street's 0.25 band, so the positioning label is not exited by a small dip.

Both series start neutral (0 and 50) sixty trading days before the calendar, so by day one they carry a history. The street window is the only guessed shape parameter; the positioning window is three times it by design, and persistence and noise were dropped because the EMA half-life already sets how sticky a view is, with the randomness coming from prices through trend and from flips.

## Where the numbers were checked

Magnitudes were checked on 2026-09-22 against FRED (Federal Reserve Economic Data, Federal Reserve Bank of St. Louis, fred.stlouisfed.org): VIXCLS, SP500, NASDAQCOM, DGS2, DGS10, T10Y2Y, the ICE BofA OAS series BAMLC0A0CM, BAMLH0A0HYM2, BAMLC0A2CAA, BAMLC0A3CA, BAMLC0A4CBBB, BAMLH0A1HYBB, BAMLH0A2HYB, the Moody's yields DAAA and DBAA, the H.10 exchange rates DEXUSEU, DEXUSUK, DEXJPUS, DEXUSAL, DEXSZUS, DEXCAUS, the spot prices DCOILWTICO and DHHNGSP, and the IMF monthly prices PCOPPUSDM, PWHEAMTUSDM, PMAIZMTUSDM.

Findings that changed the design: the stock-bond sign (yields fall when equities fall, correlation about +0.3 for 2000-2020), VIX about 1.6x higher in drawdowns than in rallies, the yen safe-haven correlation near zero in 2016-2026, credit spread asymmetry about 1.1 rather than 1.5, and Henry Hub natural gas vol above 100% a year, which is why natural gas is not in the energy group.
