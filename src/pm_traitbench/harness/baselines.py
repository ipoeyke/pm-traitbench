"""Model-backed baseline systems under test: no memory, and the full observed transcript."""

import asyncio
import datetime
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pm_traitbench.config import Config
from pm_traitbench.dialogue.client import (
    AnthropicClient,
    CachedClient,
    LlmClient,
    last_text_json,
    send_parsed,
)
from pm_traitbench.dialogue.prompts import base_request, read_advisor_prompt
from pm_traitbench.enums import ProbeForm
from pm_traitbench.errors import HarnessError
from pm_traitbench.gates.gate2.prompt_parts import mandate_line, pm_rules_section
from pm_traitbench.gates.gate2.transcript import render_pm
from pm_traitbench.harness.protocol import PublicProbe, PublicProfile, PublicSession, SutFactory

BASELINES: tuple[str, ...] = ("no-memory", "full-context")

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
MCQ_INSTRUCTION = "Reply with the letter of one option only."
OPEN_INSTRUCTION = "Reply to the PM as you would in the session."

_LETTERS = "ABCD"
_CACHED = {"cache_control": {"type": "ephemeral"}}


# The advisor prompt tells the copilot to look every figure up; a probe comes with no
# tools, so without this the copilot declines to state numbers and format checks on them
# come back not applicable.
NO_TOOLS_NOTE = (
    "This is a standalone question with no lookup tools. Work from the figures the PM "
    "quotes in the question, and say plainly when a figure is not available to you. "
    "Follow the PM's preferences for how answers are written."
)


def render_probe(probe: PublicProbe) -> str:
    """The question, its lettered options for an MCQ, then the reply instruction for its form."""
    lines = [probe.question]
    if probe.form == ProbeForm.MCQ:
        lines.extend(f"{_LETTERS[i]}. {text}" for i, text in enumerate(probe.options))
        instruction = MCQ_INSTRUCTION
    else:
        instruction = OPEN_INSTRUCTION
    return "\n".join(lines) + f"\n\n{instruction}"


def profile_text(profile: PublicProfile) -> str:
    """The mandate, self-description and PM-scope rules a copilot is told up front."""
    return "\n\n".join(
        (
            mandate_line(profile.mandate),
            f"Self-description: {profile.self_description}",
            pm_rules_section(profile.rules),
        )
    )


def parse_answer(response: Mapping[str, Any]) -> str | None:
    """The `answer` string of the reply's JSON object, or None when it is not one."""
    payload = last_text_json(response)
    if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
        return payload["answer"]
    return None


def _default_client(config: Config) -> LlmClient:
    return AnthropicClient(1, config.dialogue.api_max_retries)


class _Baseline:
    """Shared client, event loop and request building; subclasses choose the transcript part.

    Each instance owns its event loop because `AnthropicClient` binds its
    semaphore and HTTP client to the loop it first runs on, and PMs run on
    different threads.
    """

    def __init__(
        self,
        config: Config,
        profile: PublicProfile,
        run_dir: Path,
        run_name: str,
        client_factory: Callable[[Config], LlmClient] | None = None,
    ) -> None:
        self._config = config
        self._profile = profile
        self._run_name = run_name
        factory = client_factory or _default_client
        self._client = CachedClient(
            lambda: factory(config), run_dir / "cache", config.harness.pm_token_budget
        )
        self._runner = asyncio.Runner()

    def observe(self, session: PublicSession) -> None:
        """Ignore the session unless a subclass keeps memory."""

    def _memory_text(self) -> list[str]:
        """The parts of the user message that precede the date and probe."""
        return []

    def _user_content(self, as_of: datetime.date, probe: PublicProbe) -> list[dict[str, Any]]:
        """The memory as one cached block, when there is any, then the date and probe."""
        blocks: list[dict[str, Any]] = []
        memory = self._memory_text()
        if memory:
            blocks.append({"type": "text", "text": "\n\n".join(memory), **_CACHED})
        tail = f"Today is {as_of.isoformat()}.\n\n{render_probe(probe)}"
        blocks.append({"type": "text", "text": tail})
        return blocks

    def answer(self, as_of: datetime.date, probe: PublicProbe) -> str:
        """Ask the model the probe and return its answer string, or "" when it stays unparsable."""
        config = self._config
        system = f"{read_advisor_prompt(config.dialogue.advisor_prompt_path)}\n\n"
        system += f"{NO_TOOLS_NOTE}\n\n{profile_text(self._profile)}"
        request = base_request(
            config.harness.model,
            config.harness.max_answer_tokens,
            config.harness.effort,
            system,
            [{"role": "user", "content": self._user_content(as_of, probe)}],
            ANSWER_SCHEMA,
        )
        # Breakpoints end the shared prefixes (system, then memory) so a PM's probes read
        # them; the probe itself stays unmarked, as a marker there is written and never read.
        request["system"] = [{"type": "text", "text": system, **_CACHED}]
        try:
            parsed, _ = self._runner.run(
                send_parsed(
                    self._client,
                    request,
                    parse_answer,
                    scope=f"eval:{self._run_name}:{probe.probe_id}",
                    max_retries=config.dialogue.max_retries,
                    error_type=HarnessError,
                    label="probe",
                    reason="reply is not an answer object",
                )
            )
        except HarnessError:
            # Unparsable after every retry: an empty reply scores wrong; budget errors still fail.
            return ""
        return parsed

    def close(self) -> None:
        """Close the client and the event loop."""
        try:
            self._runner.run(self._client.aclose())
        finally:
            self._runner.close()


class NoMemory(_Baseline):
    """Answers from the profile alone, so it measures what the model knows without memory."""


class FullContext(_Baseline):
    """Answers with every observed transcript in the prompt, an upper bound on recall."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sessions: list[PublicSession] = []

    def observe(self, session: PublicSession) -> None:
        self._sessions.append(session)

    def _memory_text(self) -> list[str]:
        if not self._sessions:
            return []
        parts = [render_pm(self._sessions)]
        rules = {rule.rule_id: rule.text for s in self._sessions for rule in s.idea_rules}
        if rules:
            lines = "\n".join(rules[rule_id] for rule_id in sorted(rules))
            parts.append(f"Idea rules discussed:\n{lines}")
        return parts


_CLASSES: dict[str, type[_Baseline]] = {"no-memory": NoMemory, "full-context": FullContext}


def baseline_factory(
    name: str,
    config: Config,
    run_dir: Path,
    run_name: str,
    client_factory: Callable[[Config], LlmClient] | None = None,
) -> SutFactory:
    """A factory of the named baseline, one instance per PM sharing the run's response cache."""
    if name not in _CLASSES:
        raise HarnessError(f"unknown baseline '{name}'; choose from {', '.join(BASELINES)}")
    cls = _CLASSES[name]
    return lambda profile: cls(config, profile, run_dir, run_name, client_factory)
