"""The PM's own signal, forecast, forecast interval and conviction for one trade idea.

The own signal blends skill-weighted knowledge of the true forward move, read
in bullish units via the series' `bullish_sign` (so a rise that is bad for a
bearish series reads as a negative signal), with idiosyncratic noise. The
forecast then blends the resulting thesis move with the trailing trend
(extrapolation bias). The forecast interval is centred on the conditional
mean of the forward move, in series units, given the signal, with width set
by overconfidence, so it is a calibrated statement about the raw market
outcome rather than about the (possibly biased) stated forecast.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pm_traitbench.config import Config
from pm_traitbench.engine.biases import extrapolation
from pm_traitbench.engine.constants import CONVICTION_CUTS
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.engine.series import Series
from pm_traitbench.enums import Side

_MAX_CONVICTION = 5


def z_for_coverage(c: float) -> float:
    """The two-sided normal z-score whose central interval covers probability `c`."""
    return float(norm.ppf((1 + c) / 2))


Z_80 = z_for_coverage(0.8)


@dataclass(frozen=True)
class SignalDraw:
    """One draw of a PM's own signal, forecast, forecast interval and conviction."""

    own_signal: float
    sd_h: float
    thesis_move: float
    forecast: float
    interval_lo: float
    interval_hi: float
    conviction: int


def conviction_bucket(own_signal: float) -> int:
    """Conviction rank from the own signal's magnitude against the fixed CONVICTION_CUTS."""
    rank = 1 + sum(1 for cut in CONVICTION_CUTS[1:] if abs(own_signal) >= cut)
    return min(rank, _MAX_CONVICTION)


def draw_signal(
    view: MarketView,
    series: Series,
    t: int,
    params: EffectiveParams,
    config: Config,
    rng: np.random.Generator,
) -> SignalDraw:
    """Draw the PM's own signal for `series` at day `t`, and its forecast, interval, conviction."""
    horizon = config.engine.horizon_days
    skill = config.engine.skill
    sd = view.sd_h(series, t, horizon)
    fd = view.forward_days(t, horizon)
    sd_fwd = sd * math.sqrt(fd / horizon)
    if fd == 0:
        z = 0.0
    else:
        # forward_move is in series units; bullish_sign reads it in bullish units so a
        # bearish series' skill correctly points the same way as its bullish counterpart.
        z = series.bullish_sign * view.forward_move(series, t, horizon) / sd_fwd

    n = float(rng.normal())
    own_signal = skill * z + math.sqrt(1 - skill**2) * n

    thesis_move = series.bullish_sign * own_signal * sd
    trailing = view.trailing_move(series, t, horizon)
    forecast = extrapolation.blend(thesis_move, trailing, params)

    coverage = params.value("overconfidence_coverage")
    z_c = z_for_coverage(coverage)
    # Centred on the conditional mean of the forward move (series units) given the
    # signal, not the stated thesis, so a neutral PM's realised coverage equals its
    # stated coverage.
    centre = series.bullish_sign * skill * own_signal * sd_fwd
    half_width = z_c * sd_fwd * math.sqrt(1 - skill**2)

    return SignalDraw(
        own_signal=own_signal,
        sd_h=sd,
        thesis_move=thesis_move,
        forecast=forecast,
        interval_lo=centre - half_width,
        interval_hi=centre + half_width,
        conviction=conviction_bucket(own_signal),
    )


def signal_sign(draw: SignalDraw) -> Side:
    """The trade direction implied by the sign of the own signal."""
    return Side.BUY if draw.own_signal > 0 else Side.SELL
