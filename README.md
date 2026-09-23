# pm-traitbench

This repo builds the pipeline that generates PM-TraitBench, a synthetic dataset of portfolio manager
behavioural traits for evaluating a copilot's behavioural memory.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

## Usage

```sh
uv run pm-traitbench sample --config configs/demo.yaml --data-dir data
uv run pm-traitbench market --config configs/demo.yaml --data-dir data
```

The `sample` stage writes four tables to `data`: `personas`, `traits`,
`rules` and `drift_events`. The `market` stage writes six tables under
`data/market`: `instruments`, `prices`, `curves`, `consensus`, `calendar`
and `regimes`, simulating one market per configured market seed. Pass
`--force` to overwrite a table that already exists. Run `uv run
pm-traitbench --help` for the full command list.

On a full run, `market/prices`, `market/curves` and `market/consensus` are
by far the largest tables; set `output.tables` in the config to override
their format to `parquet` while leaving the rest as `jsonl`:

```yaml
output:
  tables:
    market/prices: parquet
    market/curves: parquet
    market/consensus: parquet
```

## Market model

The market simulates five instrument families (equities, rates, credit,
commodities, FX) driven by one common risk factor `z` plus per-family and
per-instrument noise, over a schedule of three regimes (range, risk-off,
risk-on) that each market seed cycles through in its own order. It trades
off realism for a model whose moments can be checked in closed form:

- No sector factor: equities load on `z` through a single beta plus
  idiosyncratic noise, with no separate sector-level co-movement.
- One common driver: every family's exposure to broad risk-on/risk-off
  moves runs through the same `z`, not a family-specific factor.
- A regime sets a driver mean, a vol multiplier, and a mean-reversion
  speed that is zero outside the range regime: switching regimes shifts
  `z`'s average level and scales every family's volatility, and in range
  it also adds a pull back toward nearby round levels.
- No vol clustering inside a regime: volatility is constant within a
  regime span, with no GARCH-style clustering on top of the regime switch.
- Rates are level and slope only: the sovereign curve is two factors, not
  a richer term-structure model, with fixed offsets giving the other tenors.
  Level vol is set per currency, and published yields are floored at
  `yield_floor_pct`, so a low-yield curve can sit at the floor.
- Credit is one factor times a rating base times issuer noise: no issuer
  defaults, a static rating band per issuer for the whole run, and issuer
  volatility is a guess since issuer-level OAS series are not public.
  The factor's asymmetric shock is centred, so it adds skew to spread moves
  without adding drift.
- Commodities are spot plus a scripted curve: the term structure follows a
  fixed roll yield by commodity group, with no seasonality and natural gas
  excluded from the simulated universe.
- FX is zero-drift per currency: each currency's log value against USD has
  no drift term, and there is no carry from interest-rate differentials.
- Consensus is a lagged moving average plus scripted flips: the street
  view tracks a trend-following average of recent price moves with
  occasional random direction flips, and both the street and positioning
  half-lives are measured in trading days, not update counts. Positioning
  is built the same way for every instrument, a slower-moving percentile
  score for crowding; credit is the one family with no public positioning
  data to check it against, so its score is a survey analogue rather than
  a proxy for a reported one.
- One sampled event class per instrument (earnings for equities, rating
  actions for credit issuers, central-bank meetings for curves, inventory
  reports for energy, crop reports for agriculture) plus a market-wide
  macro print, tied to no family, that adds directly to the common driver
  `z`; industrial metals, precious metals and FX carry no idiosyncratic
  event of their own. Contract expiries and the weekly positioning report
  are separate, deterministic calendar rows, not sampled events.
- No holidays or intraday data: the axis is every weekday, one close per
  instrument per day, with no exchange holiday calendar.

Wherever a parameter is drawn from real data, its config note names the
source: sourced values were checked against named FRED series or papers on
2026-09-22, and the credit rating-band spread medians use a calm three-year
window. Where no reliable public source exists, the config says so and
marks the value a guess: the investment-grade/high-yield credit mix and
credit duration ranges, the equity beta range, credit issuer volatility,
the commodity group-versus-idiosyncratic variance share, the rating-action
event frequencies, the consensus window and flip rate, and rates level
vol (USD is sourced, but a single field basis cannot mark only the other
currencies a guess).

Before writing its tables, the market stage checks each generated seed
against what the config implies. For every regime and family it compares
the realised annualised volatility of an equal-weight family index
against its model-implied value within a relative tolerance, and the
index's realised correlation with `z` against its model-implied
correlation within 4 standard errors of the estimate in Fisher-z space,
which keeps that error band from collapsing to nearly nothing as the
correlation approaches +-1; credit's index is the investment-grade
issuers' spread, and rates' is the mean 10Y yield change, not a price.
Sampled event types and consensus flips are checked
by count, against how many the config implies were drawn; contract
expiries and the weekly positioning report, both deterministic rather
than sampled, are checked against their exact dates instead. Round-level
tests (a close crossing into or through the band around a round level)
are counted as a family mean per instrument, for equities, commodities,
FX pairs and sovereign 10Y yields, during the range regime, with level
crossings between two closes also counted as a test. Drift is not
checked: a drift check over one regime span would pass or fail mostly by
chance, because the standard error of drift over 14-22 weeks exceeds the
drift itself.

## Development

```sh
uv run pytest
uv run ruff check
uv run ruff format
```
