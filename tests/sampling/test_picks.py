import pytest

from pm_traitbench.errors import SamplingError
from pm_traitbench.rng import stream
from pm_traitbench.sampling.picks import pick_index


def test_pick_index_is_within_range():
    rng = stream(1, "t", 0)
    for _ in range(50):
        assert 0 <= pick_index(rng, 5, "things") < 5


def test_pick_index_zero_length_raises_sampling_error_naming_what():
    rng = stream(1, "t", 0)
    with pytest.raises(SamplingError, match="empty things"):
        pick_index(rng, 0, "things")


def test_pick_index_negative_length_raises_sampling_error():
    rng = stream(1, "t", 0)
    with pytest.raises(SamplingError):
        pick_index(rng, -1, "things")


def test_pick_index_draws_exactly_one_uniform_from_the_stream():
    a = stream(1, "t", 0)
    b = stream(1, "t", 0)
    picked = pick_index(a, 5, "things")
    direct = int(b.integers(5))
    assert picked == direct
    # the stream advanced by exactly one draw for both
    assert a.random() == b.random()
