"""Tests for the per-PM voice draw."""

import pytest

from pm_traitbench.catalogues.models import Voice
from pm_traitbench.dialogue.voices import draw_voice

_VOICES = tuple(Voice(voice_id=f"v_{i:02d}", line=f"voice line {i}") for i in range(8))


def test_draw_voice_is_deterministic_per_pm() -> None:
    first = draw_voice(1, "pm_001", _VOICES)
    second = draw_voice(1, "pm_001", _VOICES)
    assert first == second


def test_draw_voice_differs_across_pms() -> None:
    drawn = {draw_voice(1, f"pm_{i:03d}", _VOICES).voice_id for i in range(40)}
    assert len(drawn) > 1


def test_draw_voice_ignores_everything_but_root_and_pm() -> None:
    # The signature carries no trait input; interleaving other draws must not
    # perturb pm_001's own draw.
    draw_voice(1, "pm_002", _VOICES)
    draw_voice(1, "pm_003", _VOICES)
    first = draw_voice(1, "pm_001", _VOICES)
    draw_voice(1, "pm_004", _VOICES)
    second = draw_voice(1, "pm_001", _VOICES)
    assert first == second


def test_draw_voice_raises_on_empty_sequence() -> None:
    with pytest.raises(ValueError):
        draw_voice(1, "pm_001", ())
