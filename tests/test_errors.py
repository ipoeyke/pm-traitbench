import pytest

from pm_traitbench.errors import (
    CatalogueError,
    ConfigError,
    PmTraitbenchError,
    SamplingError,
    StageIOError,
    TableValidationError,
)


@pytest.mark.parametrize(
    ("error_cls", "expected_exit_code"),
    [
        (ConfigError, 2),
        (CatalogueError, 2),
        (TableValidationError, 1),
        (StageIOError, 1),
        (SamplingError, 1),
    ],
)
def test_subclass_is_pm_traitbench_error_with_exit_code(
    error_cls: type[PmTraitbenchError], expected_exit_code: int
) -> None:
    assert issubclass(error_cls, PmTraitbenchError)
    assert error_cls.exit_code == expected_exit_code


def test_base_error_default_exit_code() -> None:
    assert PmTraitbenchError.exit_code == 1
