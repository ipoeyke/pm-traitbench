"""The validate stage's per-session attempt loop: runs every check layer once, then
regenerates or drops a session that fails until it passes or the attempt cap is hit.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.dialogue.client import CachedClient, session_prefix
from pm_traitbench.dialogue.context import SessionContext
from pm_traitbench.dialogue.session import SessionResult, narrate_session
from pm_traitbench.dialogue.validate.grep import check_grep
from pm_traitbench.dialogue.validate.judge import (
    forbidden_request,
    leak_request,
    map_label,
    parse_forbidden,
    parse_leak,
    parse_stance,
    send_judged,
    stance_request,
    transcript_text,
)
from pm_traitbench.dialogue.validate.ledger import (
    check_pm_levels,
    check_trades,
    count_level_warnings,
)
from pm_traitbench.enums import REVEALING_MODES, TurnRole, ValidationStatus
from pm_traitbench.errors import ValidateError
from pm_traitbench.tables.schema import DialogueLog, LedgerRow, Session, ValidationRow

# The check layers that can reject a session; each has `<layer>_reasons` on `LayerResult`
# and `<layer>_ok` on `ValidationRow`.
LAYERS = ("ledger", "level", "grep", "leak", "forbidden", "stance")
FEEDBACK_HEADER = "A validator rejected the previous version of this session"


@dataclass(frozen=True)
class LayerResult:
    """One attempt's outcome across every check layer."""

    ledger_reasons: tuple[str, ...]
    level_reasons: tuple[str, ...]
    grep_reasons: tuple[str, ...]
    leak_judged: bool
    leak_reasons: tuple[str, ...]
    forbidden_reasons: tuple[str, ...]
    stance_reasons: tuple[str, ...]
    level_warnings: int
    warnings: tuple[str, ...]
    rejected_replies: int

    def reasons_by_layer(self) -> dict[str, tuple[str, ...]]:
        """Each layer's reject reasons, keyed by layer name in `LAYERS` order."""
        return {layer: getattr(self, f"{layer}_reasons") for layer in LAYERS}

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(r for reasons in self.reasons_by_layer().values() for r in reasons)

    @property
    def passed(self) -> bool:
        return not self.reasons


def revealed_params(ctx: SessionContext, trait_param_by_id: Mapping[str, str]) -> frozenset[str]:
    """Params of stances whose mode reveals a trait: `revealed` or `contradiction`."""
    return frozenset(
        trait_param_by_id[s.trait_id] for s in ctx.skeleton.stances if s.mode in REVEALING_MODES
    )


async def _unjudged() -> tuple[None, int]:
    return None, 0


def stanced_pm_turns(ctx: SessionContext, log: DialogueLog) -> tuple[tuple[int, str, str], ...]:
    """Each PM turn that carried a stance, as its 1-based PM turn number, text and stance line."""
    pm_turns = [turn for turn in log.turns if turn.role == TurnRole.PM]
    directives = ctx.turn_plan.pm_directives
    if len(pm_turns) != len(directives):
        raise ValidateError(
            f"{session_prefix(ctx.skeleton.session_id)}log has {len(pm_turns)} PM turns "
            f"but the turn plan has {len(directives)}"
        )
    return tuple(
        (number, turn.text, directive.stance.stance)
        for number, (turn, directive) in enumerate(zip(pm_turns, directives, strict=True), 1)
        if directive.stance is not None
    )


async def validate_once(
    ctx: SessionContext,
    log: DialogueLog,
    client: CachedClient,
    config: Config,
    catalogue: Catalogue,
    ledger: Sequence[LedgerRow],
    grep_params: Sequence[str],
    trait_param_by_id: Mapping[str, str],
) -> LayerResult:
    """Run every check layer once against `log` and return their combined outcome."""
    session_id = ctx.skeleton.session_id
    prefix = session_prefix(session_id)
    max_retries = config.dialogue.max_retries

    ledger_reasons = check_trades(log, ctx.skeleton, ledger, config.validation.size_tolerance)
    level_reasons = check_pm_levels(
        log, ctx.skeleton, ctx.lookup, config.validation.level_tolerance
    )
    grep_reasons = check_grep(log, grep_params)
    level_warnings = count_level_warnings(log, config.validation.level_tolerance)

    revealed = revealed_params(ctx, trait_param_by_id)
    leak_judged = bool(revealed)
    transcript = transcript_text(log)
    forbidden_send = send_judged(
        client,
        forbidden_request(log, ctx.avoid_lines, config.validation, transcript),
        parse_forbidden,
        session_id,
        max_retries,
    )
    leak_send = (
        send_judged(
            client,
            leak_request(log, config.validation, transcript),
            parse_leak,
            session_id,
            max_retries,
        )
        if leak_judged
        else _unjudged()
    )
    stanced = stanced_pm_turns(ctx, log)
    stance_sends = (
        send_judged(
            client,
            stance_request(text, stance, config.validation),
            parse_stance,
            session_id,
            max_retries,
        )
        for _, text, stance in stanced
    )
    (
        (verdict, leak_rejected),
        (violations, forbidden_rejected),
        *stance_results,
    ) = await asyncio.gather(leak_send, forbidden_send, *stance_sends)

    stance_reasons = tuple(
        f'stance not carried out in PM turn {number}: "{stance}": {stance_verdict.reason}'
        for (number, _, stance), (stance_verdict, _) in zip(stanced, stance_results, strict=True)
        if not stance_verdict.carried_out
    )
    stance_rejected = sum(rejected for _, rejected in stance_results)

    # A judge finding counts only with its evidence: the quote must appear in a PM turn,
    # so a verdict invented from the topic rather than the text cannot fail a session.
    pm_text = " ".join(t.text for t in log.turns if t.role == TurnRole.PM)
    leak_reasons: tuple[str, ...] = ()
    warnings: list[str] = []
    if verdict is not None and verdict.explicit:
        mapped = map_label(verdict.label, catalogue.bias_labels)
        if mapped is not None and mapped in revealed:
            if is_direct_quote(verdict.quote, pm_text):
                leak_reasons = (f'leaks {mapped}: "{verdict.quote}"',)
            else:
                warnings.append(f"{prefix}leak judge quote not in a PM turn: {verdict.quote}")
        elif mapped is None and verdict.label and not verdict.label.isspace():
            warnings.append(f"{prefix}judge label unmapped: {verdict.label}")

    forbidden_reasons: list[str] = []
    for v in violations:
        if not (1 <= v.index <= len(ctx.avoid_lines)):
            warnings.append(f"{prefix}forbidden judge index out of range: {v.index}")
        elif not is_direct_quote(v.quote, pm_text):
            warnings.append(f"{prefix}forbidden judge quote not in a PM turn: {v.quote}")
        else:
            forbidden_reasons.append(f'forbidden: {ctx.avoid_lines[v.index - 1]}: "{v.quote}"')

    return LayerResult(
        ledger_reasons=ledger_reasons,
        level_reasons=level_reasons,
        grep_reasons=grep_reasons,
        leak_judged=leak_judged,
        leak_reasons=leak_reasons,
        forbidden_reasons=tuple(forbidden_reasons),
        stance_reasons=stance_reasons,
        level_warnings=level_warnings,
        warnings=tuple(warnings),
        rejected_replies=leak_rejected + forbidden_rejected + stance_rejected,
    )


def is_direct_quote(quote: str, pm_text: str) -> bool:
    """Whether a judge's quote appears verbatim in the PM turns, ignoring case and spacing."""
    needle = " ".join(quote.split()).casefold()
    return bool(needle) and needle in " ".join(pm_text.split()).casefold()


def feedback_text(attempt: int, reasons: Sequence[str]) -> str:
    """The narrator correction text for a regenerated attempt, listing every reject reason."""
    lines = "\n".join(f"- {reason}" for reason in reasons)
    return (
        f"Attempt {attempt}. {FEEDBACK_HEADER}:\n"
        f"{lines}\n"
        "Fix these and keep everything else as instructed."
    )


@dataclass(frozen=True)
class SessionOutcome:
    """One session's full attempt history: its rows, the passing result if any, and warnings."""

    rows: tuple[ValidationRow, ...]
    final: SessionResult | None
    warnings: tuple[str, ...]
    rejected_replies: int


async def run_session(
    ctx: SessionContext,
    session: Session,
    log: DialogueLog,
    client: CachedClient,
    config: Config,
    catalogue: Catalogue,
    advisor_prompt: str,
    ledger: Sequence[LedgerRow],
    grep_params: Sequence[str],
    trait_param_by_id: Mapping[str, str],
) -> SessionOutcome:
    """Validate `session`/`log`, regenerating and re-validating until it passes or is dropped."""
    max_attempts = config.validation.max_attempts
    rows: list[ValidationRow] = []
    warnings: list[str] = []
    rejected_replies = 0

    current = SessionResult(session=session, log=log, warnings=(), rejected_replies=0)
    final: SessionResult | None = None
    for attempt in range(1, max_attempts + 1):
        layer = await validate_once(
            ctx, current.log, client, config, catalogue, ledger, grep_params, trait_param_by_id
        )
        rejected_replies += layer.rejected_replies
        warnings.extend(layer.warnings)

        if layer.passed:
            status = ValidationStatus.PASS
        elif attempt < max_attempts:
            status = ValidationStatus.REGENERATE
        else:
            status = ValidationStatus.DROPPED

        rows.append(
            ValidationRow(
                pm_id=ctx.skeleton.pm_id,
                session_id=ctx.skeleton.session_id,
                attempt=attempt,
                status=status,
                leak_judged=layer.leak_judged,
                **{f"{name}_ok": not reasons for name, reasons in layer.reasons_by_layer().items()},
                level_warnings=layer.level_warnings,
                reasons=layer.reasons,
                judge_model=config.validation.judge_model,
            )
        )

        if status == ValidationStatus.PASS:
            final = current
            break
        if status == ValidationStatus.DROPPED:
            break

        current = await narrate_session(
            ctx,
            client,
            config.dialogue,
            advisor_prompt,
            feedback=feedback_text(attempt + 1, layer.reasons),
        )
        warnings.extend(current.warnings)
        rejected_replies += current.rejected_replies

    return SessionOutcome(
        rows=tuple(rows),
        final=final,
        warnings=tuple(warnings),
        rejected_replies=rejected_replies,
    )
