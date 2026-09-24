"""Tests for the PM's own signal, forecast, forecast interval and conviction."""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from pm_traitbench.config import Config, EngineConfig
from pm_traitbench.engine import own_signal
from pm_traitbench.engine.own_signal import (
    Z_80,
    SignalDraw,
    draw_signal,
    signal_sign,
    z_for_coverage,
)
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.engine.series import LegRef, Series
from pm_traitbench.enums import Side
from pm_traitbench.rng import stream

_INSTRUMENT_ID = "EQ-0001"


def _series(bullish_sign: int = 1) -> Series:
    return Series(legs=(LegRef(_INSTRUMENT_ID, None, 1.0),), bullish_sign=bullish_sign, unit="pct")


def _params(theta: float, coverage: float) -> EffectiveParams:
    return EffectiveParams(
        values={"extrapolation_theta": theta, "overconfidence_coverage": coverage},
        active={"extrapolation_theta": False, "overconfidence_coverage": False},
    )


def _config(skill: float, horizon_days: int = 20) -> Config:
    return Config(engine=EngineConfig(skill=skill, horizon_days=horizon_days))


@dataclass(frozen=True)
class _StubView:
    """A stand-in for MarketView exposing only what draw_signal reads, with fixed readings."""

    sd: float
    fd: int
    forward_move_value: float
    trailing: float

    def sd_h(self, series: Series, t: int, h: int) -> float:
        return self.sd

    def forward_days(self, t: int, h: int) -> int:
        return self.fd

    def forward_move(self, series: Series, t: int, h: int) -> float:
        return self.forward_move_value

    def trailing_move(self, series: Series, t: int, h: int) -> float:
        return self.trailing


class _PoisonForwardMoveView(_StubView):
    """A stub whose forward_move fails the test if ever called."""

    def forward_move(self, series: Series, t: int, h: int) -> float:
        raise AssertionError("forward_move must not be called when forward_days is 0")


class _CountingRng:
    """Wraps a generator and counts calls to normal(), to check the one-draw contract."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.calls = 0

    def normal(self, *args, **kwargs):
        self.calls += 1
        return self._rng.normal(*args, **kwargs)


@pytest.mark.parametrize("skill", [0.0, 0.15, 0.5])
def test_own_signal_correlation_matches_skill(skill: float) -> None:
    # False-alarm rate: tolerance 0.02 is about 3 standard errors (~0.007) over 20,000 draws.
    n_draws = 20_000
    horizon = 20
    config = _config(skill, horizon)
    series = _series()
    params = _params(theta=0.0, coverage=0.8)
    zs = np.empty(n_draws)
    signals = np.empty(n_draws)
    for i in range(n_draws):
        z = float(stream(1, "corr_z", str(skill), i).normal())
        view = _StubView(sd=1.0, fd=horizon, forward_move_value=z, trailing=0.0)
        rng = stream(1, "corr_n", str(skill), i)
        draw = draw_signal(view, series, 0, params, config, rng)
        zs[i] = z
        signals[i] = draw.own_signal
    corr = float(np.corrcoef(signals, zs)[0, 1])
    assert corr == pytest.approx(skill, abs=0.02)


@pytest.mark.parametrize("fd", [20, 5])
def test_interval_achieves_nominal_coverage(fd: int) -> None:
    # False-alarm rate: tolerance 0.02 is about 7 standard errors (~0.003) over 20,000 draws.
    # The interval is centred on the conditional mean skill * own_signal * sd_fwd, so
    # coverage is exact for any fd (full window or truncated), not just a tuned one.
    n_draws = 20_000
    horizon = 20
    skill = 0.15
    sd = 1.0
    sd_fwd = sd * math.sqrt(fd / horizon)
    config = _config(skill, horizon)
    series = _series()
    params = _params(theta=0.0, coverage=0.8)
    hits = 0
    for i in range(n_draws):
        z = float(stream(1, "cov_z", fd, i).normal())
        view = _StubView(sd=sd, fd=fd, forward_move_value=z * sd_fwd, trailing=0.0)
        rng = stream(1, "cov_n", fd, i)
        draw = draw_signal(view, series, 0, params, config, rng)
        realized = z * sd_fwd
        if draw.interval_lo <= realized <= draw.interval_hi:
            hits += 1
    coverage = hits / n_draws
    assert coverage == pytest.approx(0.8, abs=0.02)


def test_theta_one_makes_forecast_equal_trailing() -> None:
    config = _config(skill=0.15)
    series = _series()
    params = _params(theta=1.0, coverage=0.8)
    view = _StubView(sd=1.0, fd=20, forward_move_value=0.3, trailing=1.234)
    rng = stream(1, "theta_one", 0)
    draw = draw_signal(view, series, 0, params, config, rng)
    assert draw.forecast == pytest.approx(1.234)


def test_theta_zero_makes_forecast_equal_thesis_move() -> None:
    config = _config(skill=0.15)
    series = _series()
    params = _params(theta=0.0, coverage=0.8)
    view = _StubView(sd=1.0, fd=20, forward_move_value=0.3, trailing=1.234)
    rng = stream(1, "theta_zero", 0)
    draw = draw_signal(view, series, 0, params, config, rng)
    assert draw.forecast == pytest.approx(draw.thesis_move)


def test_bearish_series_flips_thesis_move_sign() -> None:
    config = _config(skill=0.15)
    params = _params(theta=0.0, coverage=0.8)
    view = _StubView(sd=1.0, fd=20, forward_move_value=0.3, trailing=0.0)
    bullish = _series(bullish_sign=1)
    bearish = _series(bullish_sign=-1)
    draw_bull = draw_signal(view, bullish, 0, params, config, stream(1, "flip", 0))
    draw_bear = draw_signal(view, bearish, 0, params, config, stream(1, "flip", 0))
    assert draw_bear.own_signal == pytest.approx(draw_bull.own_signal)
    assert draw_bear.thesis_move == pytest.approx(-draw_bull.thesis_move)


def test_zero_forward_days_gives_zero_z_without_calling_forward_move() -> None:
    config = _config(skill=0.5)
    series = _series()
    params = _params(theta=0.0, coverage=0.8)
    view = _PoisonForwardMoveView(sd=1.0, fd=0, forward_move_value=999.0, trailing=0.0)
    draw = draw_signal(view, series, 0, params, config, stream(1, "fd_zero", 0))
    n = float(stream(1, "fd_zero", 0).normal())
    expected_signal = math.sqrt(1 - 0.5**2) * n
    assert draw.own_signal == pytest.approx(expected_signal)


def test_draws_exactly_one_standard_normal() -> None:
    config = _config(skill=0.15)
    series = _series()
    params = _params(theta=0.0, coverage=0.8)
    view = _StubView(sd=1.0, fd=20, forward_move_value=0.1, trailing=0.0)
    counting_rng = _CountingRng(stream(1, "count", 0))
    draw_signal(view, series, 0, params, config, counting_rng)
    assert counting_rng.calls == 1


@pytest.mark.parametrize(
    ("own_signal_value", "expected"),
    [(1.0, 1), (1.13, 1), (1.14, 2), (1.86, 5), (5.0, 5)],
)
def test_conviction_buckets(own_signal_value: float, expected: int) -> None:
    assert own_signal._conviction(own_signal_value) == expected


def test_z_for_coverage_matches_expected_quantile() -> None:
    assert z_for_coverage(0.8) == pytest.approx(1.2816, abs=1e-3)
    assert Z_80 == pytest.approx(z_for_coverage(0.8))


def test_signal_sign_buy_when_positive() -> None:
    draw = SignalDraw(
        own_signal=0.5,
        sd_h=1.0,
        thesis_move=0.5,
        forecast=0.5,
        interval_lo=0.0,
        interval_hi=1.0,
        conviction=1,
    )
    assert signal_sign(draw) == Side.BUY


def test_signal_sign_sell_when_not_positive() -> None:
    negative = SignalDraw(
        own_signal=-0.5,
        sd_h=1.0,
        thesis_move=-0.5,
        forecast=-0.5,
        interval_lo=-1.0,
        interval_hi=0.0,
        conviction=1,
    )
    zero = SignalDraw(
        own_signal=0.0,
        sd_h=1.0,
        thesis_move=0.0,
        forecast=0.0,
        interval_lo=-1.0,
        interval_hi=1.0,
        conviction=1,
    )
    assert signal_sign(negative) == Side.SELL
    assert signal_sign(zero) == Side.SELL
