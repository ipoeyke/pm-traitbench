"""The ordered set of pipeline stages the CLI exposes as subcommands."""

from pm_traitbench.market.stage import MARKET_STAGE
from pm_traitbench.sampling.stage import SAMPLE_STAGE
from pm_traitbench.stages import Stage

STAGES: tuple[Stage, ...] = (SAMPLE_STAGE, MARKET_STAGE)
