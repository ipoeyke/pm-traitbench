"""The ordered set of pipeline stages the CLI exposes as subcommands."""

from pm_traitbench.dialogue.stage import DIALOGUE_STAGE
from pm_traitbench.dialogue.validate.stage import VALIDATE_STAGE
from pm_traitbench.engine.stage import ENGINE_STAGE
from pm_traitbench.gates.gate1.stage import GATE1_STAGE
from pm_traitbench.gates.gate2.stage import GATE2_STAGE
from pm_traitbench.market.stage import MARKET_STAGE
from pm_traitbench.probes.stage import PROBES_STAGE
from pm_traitbench.sampling.stage import SAMPLE_STAGE
from pm_traitbench.signals.stage import PLAN_STAGE
from pm_traitbench.stages import Stage

STAGES: tuple[Stage, ...] = (
    SAMPLE_STAGE,
    MARKET_STAGE,
    ENGINE_STAGE,
    GATE1_STAGE,
    PLAN_STAGE,
    DIALOGUE_STAGE,
    VALIDATE_STAGE,
    GATE2_STAGE,
    PROBES_STAGE,
)
