"""One real two-turn session against the live Anthropic API: confirms structured output,
tools and mid-conversation `system` messages all work together on the production model.

Skipped unless `--run-network` is passed, since it needs real credentials and spends tokens.
"""

import asyncio

import pytest

from pm_traitbench.config import Config
from pm_traitbench.dialogue.client import AnthropicClient, CachedClient
from pm_traitbench.dialogue.prompts import read_advisor_prompt
from pm_traitbench.dialogue.session import narrate_session
from pm_traitbench.dialogue.turns import Opening, PmDirective, TurnPlan
from pm_traitbench.enums import SessionKind, TurnRole
from tests.dialogue.conftest import session_context


@pytest.mark.network
def test_one_real_two_turn_session(market_lookup, tmp_path) -> None:
    turn_plan = TurnPlan(
        n_turns=2,
        pm_directives=(PmDirective(stance=None, trades=(), opening=Opening.SESSION_IDEAS),),
        violation_advisor_index=None,
    )
    ctx = session_context(market_lookup, kind=SessionKind.SILENCE, turn_plan=turn_plan)
    client = CachedClient(lambda: AnthropicClient(1), tmp_path, token_budget=None)
    advisor_prompt = read_advisor_prompt(None)

    result = asyncio.run(narrate_session(ctx, client, Config().dialogue, advisor_prompt))

    assert len(result.session.turns) == 2
    pm_turn, advisor_turn = result.session.turns
    assert pm_turn.role == TurnRole.PM and pm_turn.text.strip()
    assert advisor_turn.role == TurnRole.ADVISOR and advisor_turn.text.strip()

    assert len(result.log.turns) == 2
    advisor_log = result.log.turns[1]
    assert advisor_log.role == TurnRole.ADVISOR
    assert advisor_log.text.strip()
    print(f"advisor made {len(advisor_log.tool_calls)} tool call(s)")
