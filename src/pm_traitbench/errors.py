"""Exception types for the pipeline, each carrying a process exit code."""


class PmTraitbenchError(Exception):
    """Base error for all pipeline failures."""

    exit_code: int = 1


class ConfigError(PmTraitbenchError):
    """Raised when pipeline configuration is invalid."""

    exit_code = 2


class CatalogueError(PmTraitbenchError):
    """Raised when a reference catalogue is missing or invalid."""

    exit_code = 2


class TableValidationError(PmTraitbenchError):
    """Raised when a generated table fails validation."""

    exit_code = 1


class StageIOError(PmTraitbenchError):
    """Raised when a pipeline stage fails to read or write its data."""

    exit_code = 1


class SamplingError(PmTraitbenchError):
    """Raised when a sampling operation cannot produce a valid result."""

    exit_code = 1


class MarketCheckError(PmTraitbenchError):
    """Raised when a generated market misses its own regime targets."""

    exit_code = 1


class EngineError(PmTraitbenchError):
    """Raised when the behaviour engine breaks one of its own invariants."""

    exit_code = 1


class Gate1Error(PmTraitbenchError):
    """Raised when Gate 1 cannot run or a blocking recovery test fails."""

    exit_code = 1


class PlanError(PmTraitbenchError):
    """Raised when the signal plan's inputs are inconsistent or its stance bank cannot serve a
    request.
    """

    exit_code = 1


class DialogueError(PmTraitbenchError):
    """Raised when the dialogue stage cannot run or a session cannot be narrated."""

    exit_code = 1


class DialogueBudgetError(DialogueError):
    """Raised when the dialogue stage's fresh-token budget is spent."""

    exit_code = 1


class ValidateError(PmTraitbenchError):
    """Raised when the validate stage cannot run or a session cannot be judged or regenerated."""

    exit_code = 1


class Gate2Error(PmTraitbenchError):
    """Raised when Gate 2 cannot run, a reply cannot be parsed, or a blocking recovery test
    fails.
    """

    exit_code = 1


class CorpusError(PmTraitbenchError):
    """Raised when a stage finds the validated corpus missing, stale or incomplete, or a PM's
    traits incomplete.
    """

    exit_code = 1


class ProbesError(PmTraitbenchError):
    """Raised when the probes stage cannot run or cannot build a probe its inputs promise."""

    exit_code = 1
