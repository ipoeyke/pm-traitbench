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
