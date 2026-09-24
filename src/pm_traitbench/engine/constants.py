"""Fixed constants for the behaviour engine: thresholds, cuts and tenor tables."""

from pm_traitbench.enums import Tenor

# Entry requires the own signal to reach one sd of view.
ENTRY_THRESHOLD = 1.0
# Conviction tiers at successive multiples of one sd of view, the first at entry.
# Lower edges for buckets 1-5: quintiles of a standard normal's absolute value given it
# exceeds 1, so ranks are equally likely.
CONVICTION_CUTS = (1.0, 1.14, 1.31, 1.53, 1.86)
# Each idea carries two or three signposts to track toward its target.
SIGNPOSTS_PER_IDEA = (2, 3)
# An add buys half the position's original size.
ADD_FRACTION = 0.5
# Size at conviction rank 1-5 as a fraction of the mandate cap, linear.
RISK_STEPS = (0.2, 0.4, 0.6, 0.8, 1.0)
# No new entries in the final five sessions, so every idea gets at least a week of life.
NO_ENTRY_LAST_SESSIONS = 5
# Realised volatility for sizing and signposts is measured over a trailing 60 days.
TRAILING_SD_DAYS = 60
# Fewer than five days of history is too short to trust a vol estimate.
MIN_SD_DAYS = 5
# Keeps a division by realised vol finite when a market has gone dead flat.
SD_FLOOR = 1e-6
# Consecutive sessions a level signpost must hold before it fires.
LEVEL_WINDOW_CHOICES = (3, 4, 5)
# Lookback for the prior-high anchor used by the anchoring bias.
TRAILING_HIGH_DAYS = 60

# DV01 per $1mm notional by sovereign tenor, in local currency.
DV01_PER_MILLION: dict[Tenor, float] = {
    Tenor.Y2: 190,
    Tenor.Y5: 450,
    Tenor.Y10: 800,
    Tenor.Y30: 1700,
}

# The on-the-run outright tenor for single-leg rates ideas.
OUTRIGHT_TENOR = Tenor.Y10
# Steepener and flattener pairs traded across the curve.
CURVE_PAIRS = ((Tenor.Y2, Tenor.Y10), (Tenor.Y5, Tenor.Y30))
# Back-month legs used for calendar spread ideas.
CALENDAR_BACK_TENORS = (Tenor.M4, Tenor.M5, Tenor.M6)

# Fixed order in which bias flags on an entry are joined.
BIAS_FLAG_ORDER = ("herding", "overconfidence", "conviction")
