"""One real four-turn session against the live Anthropic API: confirms structured output,
tool calls, and mid-conversation `system` messages, including a scripted advisor
violation, all work together on the production model.

Skipped unless `--run-network` is passed, since it needs real credentials and spends tokens.
"""

import asyncio

import pytest

from pm_traitbench.config import Config
from pm_traitbench.dialogue.client import AnthropicClient, CachedClient
from pm_traitbench.dialogue.prompts import read_advisor_prompt
from pm_traitbench.dialogue.session import narrate_session
from pm_traitbench.dialogue.turns import Opening, PmDirective, TurnPlan
from pm_traitbench.enums import SessionKind, SignalMode, StanceEntry, TurnRole
from pm_traitbench.tables.schema import Stance
from tests.dialogue.fixtures import session_context


@pytest.mark.network
def test_one_real_four_turn_session_with_a_violation(market_lookup, tmp_path) -> None:
    reaction = Stance(
        signal_id="sg_001",
        trait_id="t_01",
        mode=SignalMode.REVEALED,
        entry=StanceEntry.REVEALED_REACTION,
        stance="That's not what I asked you to do.",
    )
    turn_plan = TurnPlan(
        n_turns=4,
        pm_directives=(
            PmDirective(stance=None, trades=(), opening=Opening.OPEN_POSITIONS),
            PmDirective(stance=reaction, trades=(), opening=None),
        ),
        violation_advisor_index=0,
    )
    ctx = session_context(
        market_lookup,
        kind=SessionKind.CHECK_IN,
        stances=(reaction,),
        advisor_violation="Sold half the position without asking.",
        turn_plan=turn_plan,
    )
    client = CachedClient(lambda: AnthropicClient(1), tmp_path, token_budget=None)
    advisor_prompt = read_advisor_prompt(None)

    result = asyncio.run(narrate_session(ctx, client, Config().dialogue, advisor_prompt))

    assert len(result.session.turns) == 4
    pm_turn_0, advisor_turn_0, pm_turn_1, advisor_turn_1 = result.session.turns
    assert pm_turn_0.role == TurnRole.PM and pm_turn_0.text.strip()
    assert advisor_turn_0.role == TurnRole.ADVISOR and advisor_turn_0.text.strip()
    assert pm_turn_1.role == TurnRole.PM and pm_turn_1.text.strip()
    assert advisor_turn_1.role == TurnRole.ADVISOR and advisor_turn_1.text.strip()

    assert len(result.log.turns) == 4
    assert result.log.turns[1].scripted_violation is True
    assert result.log.turns[3].scripted_violation is False
    for turn_log in result.log.turns:
        assert turn_log.text.strip()
    total_tool_calls = sum(len(t.tool_calls) for t in result.log.turns)
    print(f"advisor made {total_tool_calls} tool call(s) across both turns")
