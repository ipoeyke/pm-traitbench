import os
import subprocess
import sys

import pytest

from pm_traitbench.rng import stream


def test_same_seed_and_keys_give_identical_draws() -> None:
    first = stream(42, "pm", 1, "biases").random(5)
    second = stream(42, "pm", 1, "biases").random(5)
    assert list(first) == list(second)


def test_different_keys_give_different_draws() -> None:
    a = stream(42, "pm", 1, "biases").random(5)
    b = stream(42, "pm", 2, "biases").random(5)
    c = stream(42, "pm", 1, "rules").random(5)
    assert list(a) != list(b)
    assert list(a) != list(c)
    assert list(b) != list(c)


def test_string_key_hashing_is_stable_across_processes() -> None:
    in_process = stream(7, "pm", 3, "drift").random(1)[0]

    script = "from pm_traitbench.rng import stream; print(stream(7, 'pm', 3, 'drift').random(1)[0])"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "12345"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess_value = float(result.stdout.strip())

    assert subprocess_value == in_process


def test_negative_int_key_raises_value_error() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        stream(1, "pm", -1, "biases")
