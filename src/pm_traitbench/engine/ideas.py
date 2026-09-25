"""Idea generation: turns a PM's daily entry attempts into new trade ideas.

Composes the own-signal draw, extrapolation-blended with the trailing move
for entry and side, with herding, overconfidence and conviction biases, an
adapter's leg construction and sizing, and catalogue-driven thesis and
signpost text, into one new position (plus its idea, rule and ledger rows)
per attempt. `attempt_entry` and `entries_for_day` are pure: neither mutates
its `state` argument, both return a new one.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from pm_traitbench.catalogues.models import Catalogue
from pm_traitbench.config import Config
from pm_traitbench.engine.adapters import preferred_form
from pm_traitbench.engine.adapters.base import Adapter, leg_side
from pm_traitbench.engine.biases import (
    conviction,
    extrapolation,
    herding,
    join_flags,
    overconfidence,
)
from pm_traitbench.engine.constants import (
    ENTRY_THRESHOLD,
    LEVEL_WINDOW_CHOICES,
    NO_ENTRY_LAST_SESSIONS,
    RISK_STEPS,
    SIGNPOSTS_PER_IDEA,
)
from pm_traitbench.engine.market_view import MarketView
from pm_traitbench.engine.own_signal import draw_signal
from pm_traitbench.engine.params import EffectiveParams
from pm_traitbench.engine.series import Series
from pm_traitbench.engine.state import PmState, Position, idea_id, rule_id
from pm_traitbench.engine.templates import (
    idea_name,
    level_text,
    render_signpost_text,
    render_thesis,
)
from pm_traitbench.engine.triggers import ledger_rows, leg_sides
from pm_traitbench.enums import (
    Action,
    AssetClass,
    Expression,
    Op,
    RuleScope,
    RuleSource,
    Side,
    StreetView,
)
from pm_traitbench.errors import EngineError
from pm_traitbench.tables.schema import Idea, LedgerRow, Leg, Persona, Rule, Trait

_SIDE_SIGN: dict[Side, int] = {Side.BUY: 1, Side.SELL: -1}
_PRICE_QUOTED_OUTRIGHT_CLASSES = (AssetClass.EQUITIES, AssetClass.COMMODITIES)


@dataclass(frozen=True)
class NewIdea:
    """A freshly entered trade idea: its position and the rows it produced."""

    position: Position
    idea: Idea
    rules: tuple[Rule, ...]
    ledger_rows: tuple[LedgerRow, ...]
    conflict: bool
    entered_after_run: bool


def _find_pm_rule(rules: Sequence[Rule], param: str) -> Rule | None:
    """The PM-scope rule for `param`, or None if the PM holds no such rule."""
    for r in rules:
        if r.scope == RuleScope.PM and r.param == param:
            return r
    return None


def _structurally_blocked(
    state: PmState, t: int, view: MarketView, pm_rules: Sequence[Rule], universe: Sequence[str]
) -> bool:
    """Whether no attempt should be drawn today: too late, at capacity, or nothing to buy."""
    if t >= view.n_days - NO_ENTRY_LAST_SESSIONS:
        return True
    max_positions_rule = _find_pm_rule(pm_rules, "max_positions")
    if max_positions_rule is not None and state.n_positions >= float(max_positions_rule.level):
        return True
    return frozenset(universe) <= state.held_instruments


def _side_word(expression: Expression, side: Side) -> str:
    """The thesis template's `{side}` word for this idea's form and direction."""
    if expression == Expression.CURVE:
        return "steepener" if side == Side.BUY else "flattener"
    if expression == Expression.CALENDAR_SPREAD:
        return "long the front" if side == Side.BUY else "short the front"
    return "long" if side == Side.BUY else "short"


def _stop_text(level: float, unit: str, adverse_dir: int, price_quoted: bool) -> str:
    word = "above" if adverse_dir > 0 else "below"
    return f"stop {word} {level_text(level, unit, price_quoted=price_quoted)}"


def _target_text(level: float, unit: str, adverse_dir: int, price_quoted: bool) -> str:
    word = "below" if adverse_dir > 0 else "above"
    return f"target {word} {level_text(level, unit, price_quoted=price_quoted)}"


def _draw_form(
    forms: tuple[Expression, ...],
    preferred: Expression | None,
    weight: float,
    rng: np.random.Generator,
) -> Expression:
    """Draw an idea's structural form, favouring `preferred` by `weight` when it applies."""
    if preferred is not None and preferred in forms:
        n_other = len(forms) - 1
        if n_other == 0:
            probs = [1.0]
        else:
            probs = [weight if f == preferred else (1 - weight) / n_other for f in forms]
    else:
        probs = [1.0 / len(forms)] * len(forms)
    return forms[int(rng.choice(len(forms), p=probs))]


def _signpost_kinds(
    rng: np.random.Generator, event_types_present: bool, has_peers: bool
) -> list[str]:
    """The signpost kinds for one idea: 2 or 3, without replacement from those available.

    `relative` needs a peer other than the candidate, else its move is identically zero.
    """
    available = ["level"] + (["relative"] if has_peers else [])
    available += ["event"] if event_types_present else []
    n = min(int(rng.choice(SIGNPOSTS_PER_IDEA)), len(available))
    return list(rng.choice(available, size=n, replace=False))


def _build_signposts(
    *,
    pm_id: str,
    trade_idea_id: str,
    candidate: str,
    adapter: Adapter,
    view: MarketView,
    catalogue: Catalogue,
    series: Series,
    entry_level: float,
    adverse_dir: int,
    price_quoted: bool,
    signpost_k: float,
    sd: float,
    next_rid: Callable[[], str],
    rng_signposts: np.random.Generator,
    rng_templates: np.random.Generator,
) -> tuple[Rule, ...]:
    """This idea's 2 or 3 idea-scope signpost rules."""
    event_types = view.event_types(candidate)
    has_peers = bool(set(adapter.peer_ids(candidate, view.instruments)) - {candidate})
    kinds = _signpost_kinds(rng_signposts, bool(event_types), has_peers)
    peer = adapter.peer_label(candidate, view.instruments)
    rows: list[Rule] = []
    for kind in kinds:
        if kind == "event":
            field, op, window, unit = "event", Op.EQ, 1, None
            level: float | str = str(rng_signposts.choice(sorted(et.value for et in event_types)))
            text = render_signpost_text(
                catalogue, adapter.asset_class, "event", event=level, rng=rng_templates
            )
        elif kind == "level":
            window = int(rng_signposts.choice(LEVEL_WINDOW_CHOICES))
            field, unit = "level", series.unit
            level = entry_level + adverse_dir * signpost_k * sd
            op = Op.GE if adverse_dir > 0 else Op.LE
            text = render_signpost_text(
                catalogue,
                adapter.asset_class,
                "level",
                level=level,
                unit=unit,
                price_quoted=price_quoted,
                window=window,
                rng=rng_templates,
            )
        else:
            field, op, unit, window = "relative_move", Op.LE, series.unit, 1
            level = -signpost_k * sd
            text = render_signpost_text(
                catalogue,
                adapter.asset_class,
                "relative",
                level=abs(level),
                unit=unit,
                peer=peer,
                rng=rng_templates,
            )
        rows.append(
            Rule(
                pm_id=pm_id,
                rule_id=next_rid(),
                source=RuleSource.SELF,
                scope=RuleScope.IDEA,
                trade_idea_id=trade_idea_id,
                param="signpost",
                field=field,
                op=op,
                level=level,
                unit=unit,
                window=window,
                action=Action.SIGNPOST,
                text=text,
            )
        )
    return tuple(rows)


def forecast_z(own_signal: float, trail_z: float, theta: float) -> float:
    """Theta-blend of the own signal and the trailing move, in unit-variance z-units.

    Both inputs already carry unit variance, so dividing by
    `sqrt((1-theta)**2 + theta**2)` keeps the blend at unit variance for any theta.
    """
    return ((1 - theta) * own_signal + theta * trail_z) / math.sqrt((1 - theta) ** 2 + theta**2)


def attempt_entry(
    state: PmState,
    t: int,
    view: MarketView,
    adapter: Adapter,
    params: EffectiveParams,
    persona: Persona,
    pm_rules: Sequence[Rule],
    traits: Sequence[Trait],
    universe: Sequence[str],
    config: Config,
    catalogue: Catalogue,
    rng_for: Callable[..., np.random.Generator],
    attempt: int,
) -> tuple[PmState, "NewIdea | None"]:
    """Attempt one new trade idea for this PM on day `t`; None if none was entered."""
    if _structurally_blocked(state, t, view, pm_rules, universe):
        return state, None

    available = [iid for iid in universe if iid not in state.held_instruments]
    candidate = available[int(rng_for("candidate", t, attempt).integers(len(available)))]

    forms = adapter.forms(persona.mandate.sub_style)
    preferred = preferred_form(traits)
    form = _draw_form(
        forms, preferred, config.engine.preferred_form_weight, rng_for("form", t, attempt)
    )

    rng_legs = rng_for("legs", t, attempt)
    legs = adapter.build_legs(form, candidate, view, t, universe, state.held_instruments, rng_legs)
    if legs is None:
        form = Expression.OUTRIGHT
        legs = adapter.build_legs(
            form, candidate, view, t, universe, state.held_instruments, rng_legs
        )
        if legs is None:
            raise EngineError(f"adapter could not build outright legs for '{candidate}'")

    series = adapter.series(form, legs)
    draw = draw_signal(view, series, t, params, config, rng_for("signal", t, attempt))
    trailing_move = view.trailing_move(series, t, config.engine.horizon_days)
    # Extrapolation bias: entry and side follow the theta-blend of the own signal
    # with the trailing move (read in bullish units, scaled to the same z-units).
    trail_z = series.bullish_sign * trailing_move / draw.sd_h
    fz = forecast_z(draw.own_signal, trail_z, params.value("extrapolation_theta"))
    if abs(fz) < ENTRY_THRESHOLD:
        return state, None

    own_side = Side.BUY if fz > 0 else Side.SELL
    street = view.street_view(candidate, t)
    decision = herding.decide(own_side, street, params, rng_for("herding", t, attempt))
    side = decision.side
    side_sign = _SIDE_SIGN[side]

    rank, conviction_flag = conviction.size_rank(
        draw.conviction, params, rng_for("size_rank", t, attempt)
    )
    factor, overconfidence_flag = overconfidence.size_factor(params)
    cap_rule = _find_pm_rule(pm_rules, "max_risk_pct")
    if cap_rule is None:
        raise EngineError("no mandate risk cap rule found for this PM")
    cap_level = float(cap_rule.level)
    size_pct_book = min(cap_level, cap_level * RISK_STEPS[rank - 1] * factor)
    if size_pct_book > cap_level:
        raise EngineError("sized idea exceeds the mandate risk cap")

    stop_rule = _find_pm_rule(pm_rules, "stop_loss")
    if stop_rule is None:
        raise EngineError("no stop-loss rule found for this PM")
    stop_distance = adapter.stop_distance(stop_rule, series)
    rr_lo, rr_hi = config.engine.rr_range
    rr = float(rng_for("rr", t, attempt).uniform(rr_lo, rr_hi))
    entry_level = view.level(series, t)
    adverse_dir = -series.bullish_sign * side_sign
    stop_level = entry_level + adverse_dir * stop_distance
    target_level = entry_level - adverse_dir * rr * stop_distance

    price_quoted = (
        form == Expression.OUTRIGHT and adapter.asset_class in _PRICE_QUOTED_OUTRIGHT_CLASSES
    )
    trade_idea_id = idea_id(state.next_idea)
    rule_counter = state.next_rule

    def _next_rid() -> str:
        nonlocal rule_counter
        rid = rule_id(rule_counter)
        rule_counter += 1
        return rid

    stop_row = Rule(
        pm_id=persona.pm_id,
        rule_id=_next_rid(),
        source=RuleSource.SELF,
        scope=RuleScope.IDEA,
        trade_idea_id=trade_idea_id,
        param="stop",
        field="level",
        op=Op.GE if adverse_dir > 0 else Op.LE,
        level=stop_level,
        unit=series.unit,
        window=1,
        action=Action.EXIT,
        text=_stop_text(stop_level, series.unit, adverse_dir, price_quoted),
    )
    target_row = Rule(
        pm_id=persona.pm_id,
        rule_id=_next_rid(),
        source=RuleSource.SELF,
        scope=RuleScope.IDEA,
        trade_idea_id=trade_idea_id,
        param="target",
        field="level",
        op=Op.LE if adverse_dir > 0 else Op.GE,
        level=target_level,
        unit=series.unit,
        window=1,
        action=Action.TARGET,
        text=_target_text(target_level, series.unit, adverse_dir, price_quoted),
    )
    rng_templates = rng_for("templates", t, attempt)
    signpost_rows = _build_signposts(
        pm_id=persona.pm_id,
        trade_idea_id=trade_idea_id,
        candidate=candidate,
        adapter=adapter,
        view=view,
        catalogue=catalogue,
        series=series,
        entry_level=entry_level,
        adverse_dir=adverse_dir,
        price_quoted=price_quoted,
        signpost_k=config.engine.signpost_k,
        sd=draw.sd_h,
        next_rid=_next_rid,
        rng_signposts=rng_for("signposts", t, attempt),
        rng_templates=rng_templates,
    )

    name = idea_name(view, legs, form)
    # The expected move in the trade's own direction, so a herding follow never shows the conflict.
    stated_move = series.bullish_sign * side_sign * abs(draw.forecast)
    thesis = render_thesis(
        catalogue,
        adapter.asset_class,
        form,
        side=_side_word(form, side),
        name=name,
        entry=entry_level,
        target=target_level,
        move=stated_move,
        unit=series.unit,
        price_quoted=price_quoted,
        horizon=config.engine.horizon_days,
        rng=rng_templates,
    )

    # Position.legs keeps the literal traded tenor (a commodity outright still tracks M1,
    # for roll math); the idea's own legs only carry a tenor for curve/calendar-spread forms.
    position_legs = tuple(
        Leg(
            instrument_id=leg.instrument_id,
            tenor=leg.tenor,
            side=leg_side(side_sign, series.bullish_sign, leg.coeff, adapter.leg_bullish(leg)),
            weight=abs(leg.coeff),
        )
        for leg in legs
    )
    idea_leg_requires_tenor = form in (Expression.CURVE, Expression.CALENDAR_SPREAD)
    idea_legs = (
        position_legs
        if idea_leg_requires_tenor
        else tuple(leg.model_copy(update={"tenor": None}) for leg in position_legs)
    )

    idea_row = Idea(
        pm_id=persona.pm_id,
        trade_idea_id=trade_idea_id,
        instrument_id=candidate,
        expression=form,
        side=side,
        legs=idea_legs,
        entry_date=view.dates[t],
        exit_date=None,
        entry_level=entry_level,
        target_level=target_level,
        stop_level=stop_level,
        thesis=thesis,
        outcome=None,
        own_signal=draw.own_signal,
        forecast=draw.forecast,
        interval_lo=draw.interval_lo,
        interval_hi=draw.interval_hi,
        street_view_at_entry=street if street is not None else StreetView.NEUTRAL,
        conflict=decision.conflict,
        followed_street=decision.followed_street,
        conviction=draw.conviction,
        size_rank=rank,
    )

    size_at_entry, _ = adapter.size_and_risk(
        size_pct_book, legs, view, t, persona.mandate.book_size
    )
    position = Position(
        trade_idea_id=trade_idea_id,
        expression=form,
        instrument_id=candidate,
        legs=position_legs,
        series=series,
        side=side,
        entry_t=t,
        entry_level=entry_level,
        target_level=target_level,
        stop_level=stop_level,
        sd_h_at_entry=draw.sd_h,
        forecast=draw.forecast,
        size_pct_book=size_pct_book,
        original_size_pct_book=size_pct_book,
        size_at_entry=size_at_entry,
        conviction=draw.conviction,
        size_rank=rank,
        triggers_fired=0,
        consumed_rule_ids=frozenset(),
        run_counters=(),
        size_changed_t=t,
    )
    ledger = ledger_rows(
        position,
        leg_sides(position, side_sign, adapter),
        size_pct_book,
        pm_id=persona.pm_id,
        adapter=adapter,
        book_size=persona.mandate.book_size,
        view=view,
        t=t,
        bias_flag=join_flags([decision.flag, overconfidence_flag, conviction_flag]),
        rule_id=None,
    )

    entered_after_run = extrapolation.entered_after_run(
        trailing_move, draw.sd_h, side_sign, series.bullish_sign
    )

    new_state = replace(
        state.add_position(position), next_idea=state.next_idea + 1, next_rule=rule_counter
    )
    new_idea = NewIdea(
        position=position,
        idea=idea_row,
        rules=(stop_row, target_row, *signpost_rows),
        ledger_rows=ledger,
        conflict=decision.conflict,
        entered_after_run=entered_after_run,
    )
    return new_state, new_idea


def entries_for_day(
    state: PmState,
    t: int,
    view: MarketView,
    adapter: Adapter,
    params: EffectiveParams,
    persona: Persona,
    pm_rules: Sequence[Rule],
    traits: Sequence[Trait],
    universe: Sequence[str],
    config: Config,
    catalogue: Catalogue,
    rng_for: Callable[..., np.random.Generator],
) -> tuple[PmState, list["NewIdea"]]:
    """Attempt this PM's entries for one day; stop at the first structural block."""
    n_attempts = int(rng_for("arrival", t).poisson(config.engine.arrival_rate))
    new_ideas: list[NewIdea] = []
    for attempt in range(n_attempts):
        if _structurally_blocked(state, t, view, pm_rules, universe):
            break
        state, new_idea = attempt_entry(
            state,
            t,
            view,
            adapter,
            params,
            persona,
            pm_rules,
            traits,
            universe,
            config,
            catalogue,
            rng_for,
            attempt,
        )
        if new_idea is not None:
            new_ideas.append(new_idea)
    return state, new_ideas
