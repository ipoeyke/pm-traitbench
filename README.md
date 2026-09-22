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
- A regime is a driver mean and a vol multiplier: switching regimes shifts
  `z`'s average level and scales every family's volatility, rather than
  changing the shape of the process.
- No vol clustering inside a regime: volatility is constant within a
  regime span, with no GARCH-style clustering on top of the regime switch.
- Rates are level and slope only: the sovereign curve is two factors, not
  a richer term-structure model, with fixed offsets giving the other tenors.
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
- Consensus is a lagged moving average plus scripted flips: the street view
  tracks a trend-following average of recent price moves with occasional
  random direction flips, and credit positioning is a slower-moving
  percentile score standing in for a real positioning survey.
- Each instrument sees at most one narrow class of scripted event over the
  horizon: earnings for equities, a rating action for credit issuers,
  central bank meetings for sovereign curves, and an inventory or crop
  report for energy and agriculture commodities; industrial metals,
  precious metals and FX carry no idiosyncratic event of their own.
- No holidays or intraday data: the axis is every weekday, one close per
  instrument per day, with no exchange holiday calendar.

Wherever a parameter is drawn from real data, its config note names the
source: sourced values were checked against named FRED series or papers on
2026-09-22, and the credit rating-band spread medians use a calm three-year
window. Where no reliable public source exists, the config says so and
marks the value a guess: the investment-grade/high-yield credit mix and
credit duration ranges, the equity beta range, credit issuer volatility,
the commodity group-versus-idiosyncratic variance share, the rating-action
event frequencies, and the consensus window and flip rate.

Before writing its tables, the market stage checks each generated seed
against what the config implies. For every regime and family it compares
the realised annualised volatility of an equal-weight family index against
its model-implied value within a relative tolerance, and the index's
realised correlation with `z` against its model-implied correlation within
4 standard errors of the estimate. It checks that every scheduled event
type, consensus flip, contract expiry and positioning report lands on the
exact dates and counts the model implies. Round-level tests (a close
crossing into or through the band around a round price, curve or FX level)
are counted as a family mean per instrument during the range regime, with
level crossings between two closes also counted as a test. Drift is not
checked: its standard error over a 14-22 week regime span exceeds the
drift itself, so no seed could pass a drift check by chance alone.

## Development

```sh
uv run pytest
uv run ruff check
uv run ruff format
```
