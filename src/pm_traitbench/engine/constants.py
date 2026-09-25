"""Fixed constants for the behaviour engine: thresholds, cuts and tenor tables."""

from pm_traitbench.enums import Tenor

# Entry requires the blended forecast to reach one sd of view.
ENTRY_THRESHOLD = 1.0
# Lower edges of conviction buckets 1-5: quintiles of the own signal's absolute value given
# it exceeds 1, a bucketing floor on stated conviction, separate from the forecast-based entry gate.
CONVICTION_CUTS = (1.0, 1.14, 1.31, 1.53, 1.86)
# Each idea carries two or three signposts to track toward its target.
SIGNPOSTS_PER_IDEA = (2, 3)
# An add buys half the position's original size.
ADD_FRACTION = 0.5
# A neutral PM (lambda about 1.1) cuts on about 4.5% of loss-side days.
LOSS_CUT_HAZARD = 0.05
# An active PM at the 2.0 centre adds on about 10% of loss-side days.
LOSS_ADD_SLOPE = 0.1
# Caps the add hazard so no lambda adds on more than half of loss-side days.
LOSS_ADD_CAP = 0.5
# Size at conviction rank 1-5 as a fraction of the mandate cap, linear.
RISK_STEPS = (0.2, 0.4, 0.6, 0.8, 1.0)
# Overconfidence's size factor at the active coverage centre 0.4 (Z_80 / z(0.4), about
# 2.45), so a centre-planted PM's largest step still lands under the mandate cap.
SIZE_HEADROOM = 2.5
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
# The salient round-level anchor sits 40% of the way from entry to target, where
# most ideas that do not stop out reach it.
ANCHOR_FRACTION = 0.4

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
