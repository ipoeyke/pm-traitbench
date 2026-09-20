"""Probability distribution specs used to describe bias and preference strength."""

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats


class _BoundedSpec(BaseModel):
    """Shared truncation logic for distribution specs with optional lo/hi bounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lo: float | None = None
    hi: float | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> "_BoundedSpec":
        if self.lo is not None and self.hi is not None and self.lo >= self.hi:
            raise ValueError("lo must be less than hi")
        dist = self._frozen()
        f_lo = dist.cdf(self.lo) if self.lo is not None else 0.0
        f_hi = dist.cdf(self.hi) if self.hi is not None else 1.0
        if f_hi - f_lo < 1e-9:
            raise ValueError(f"truncation window [lo={self.lo}, hi={self.hi}] is degenerate")
        return self

    def _frozen(self) -> Any:
        raise NotImplementedError

    def median_value(self) -> float:
        """Median of the untruncated distribution."""
        return float(self._frozen().median())

    def ppf(self, u: np.ndarray | float) -> np.ndarray | float:
        """Truncated inverse CDF: rescale u onto [F(lo), F(hi)] before inverting."""
        dist = self._frozen()
        is_scalar = np.isscalar(u)
        uu = np.clip(np.atleast_1d(np.asarray(u, dtype=float)), 1e-12, 1 - 1e-12)
        f_lo = dist.cdf(self.lo) if self.lo is not None else 0.0
        f_hi = dist.cdf(self.hi) if self.hi is not None else 1.0
        result = dist.ppf(f_lo + uu * (f_hi - f_lo))
        return float(result[0]) if is_scalar else result

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """Truncated CDF: inverse of the rescaling used in ppf, clipped to [0, 1]."""
        dist = self._frozen()
        is_scalar = np.isscalar(x)
        xx = np.atleast_1d(np.asarray(x, dtype=float))
        f_lo = dist.cdf(self.lo) if self.lo is not None else 0.0
        f_hi = dist.cdf(self.hi) if self.hi is not None else 1.0
        result = np.clip((dist.cdf(xx) - f_lo) / (f_hi - f_lo), 0.0, 1.0)
        return float(result[0]) if is_scalar else result


class LogNormalSpec(_BoundedSpec):
    kind: Literal["lognormal"] = "lognormal"
    median: float = Field(gt=0)
    sigma: float = Field(gt=0)

    def _frozen(self) -> Any:
        return stats.lognorm(s=self.sigma, scale=self.median)


class BetaSpec(_BoundedSpec):
    kind: Literal["beta"] = "beta"
    a: float = Field(gt=0)
    b: float = Field(gt=0)

    def _frozen(self) -> Any:
        return stats.beta(self.a, self.b)


Distribution = Annotated[LogNormalSpec | BetaSpec, Field(discriminator="kind")]
