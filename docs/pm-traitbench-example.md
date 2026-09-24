# PM-TraitBench: worked example, one commodities PM across the year

Illustration only. Nothing here was produced by the engine or a narrator; every row is hand-written to show what the files contain and how the pieces refer to each other. The PM is `pm_066`, a commodities manager. The book is commodities only, traded through spot, forwards, options and futures; there is no rates or equity leg. Market seed C, so the year runs risk-off (2026-01-05 to 2026-04-26), then risk-on (to 2026-08-02), then range (to 2027-01-03).

Reading guide: blue files (persona, rules, ideas, ledger, rule events, sessions) are what the copilot under test can see. Red files (traits, drift events, signals, probes) are hidden ground truth used only for generation and scoring.

> **Aligned with the repo's sample stage as merged to `main` on 2026-09-22.** Sections 1-4 use the row shapes, catalogue strings and drift rule the pipeline writes. The PM trades commodities only, the roll rule `r_05` is a catalogue rule for commodities mandates, and preference `t_13` is a catalogue expression preference. The id `pm_066` is the slot the default grid gives a full-split, commodities, seed C, typical, drift PM; ids `pm_001` to `pm_032` are the pilot, which runs on seed A only. Every table is a `.jsonl` file (parquet is the alternative; CSV is not supported). Sections 5-7 describe stages that are not built yet and were edited only where they quoted a changed value, a file name, or the old rates leg.

## 1. Persona

```yaml
pm_id: pm_066
market_seed: C
split: full                        # the full split covers all three market seeds; the pilot runs on seed A only
mandate:
  asset_class: commodities
  sub_style: commodity_futures_directional
  book_size: 250000000
  risk_unit: contracts
  benchmark: commodity_index
stated_profile:
  self_description: "my entry price shapes how i see it, a strong recent run makes me expect more"
typicality: typical                # two catalogue fragments that agree with the two strongest biases by quantile: anchoring (0.69), extrapolation (0.67)
```

## 2. Rules (visible)

PM-scope rules apply to every idea. Idea-scope rules are written at entry for one idea; two ideas are shown.

```yaml
# PM scope
- {rule_id: r_01, source: mandate, scope: pm, param: max_risk_pct,     field: size_pct_book,  op: "<=", level: 8,   unit: pct,            window: 1, action: cap,       text: "no commodity gets more than 8% of NAV"}
- {rule_id: r_02, source: self,    scope: pm, param: stop_loss,        field: pnl_from_entry, op: "<=", level: -10, unit: pct,            window: 1, action: exit,      text: "I'm out if the contract is down to -10% from entry"}
- {rule_id: r_03, source: self,    scope: pm, param: trim_at_target,   field: target_hit,     op: "==", level: 1,                         window: 1, action: trim_half, text: "half the position comes off at the target level"}
- {rule_id: r_04, source: self,    scope: pm, param: min_holding_period, field: sessions_held, op: ">=", level: 5, unit: sessions,        window: 1, action: hold,      text: "I hold the contract for at least 5 sessions before exiting"}
- {rule_id: r_05, source: self,    scope: pm, param: roll_before_expiry, field: days_to_expiry, op: "<=", level: 5,   unit: sessions,       window: 1, action: roll,      text: "everything rolls 5 sessions before first notice"}
- {rule_id: r_06, source: self,    scope: pm, param: max_positions,    field: n_positions,    op: "<=", level: 12,  unit: positions,     window: 1, action: cap,       text: "I cap the book at 12 positions"}

# idea scope, ti_104: long gold futures (MT-GLD), entered 2026-02-04 at 2,355.0
- {rule_id: r_41, source: self, scope: idea, trade_idea_id: ti_104, param: stop,     field: price,  op: "<=", level: 2260.0, window: 1, action: exit,     text: "stop 2,260"}
- {rule_id: r_42, source: self, scope: idea, trade_idea_id: ti_104, param: target,   field: price,  op: ">=", level: 2540.0, window: 1, action: target,   text: "target 2,540"}
- {rule_id: r_43, source: self, scope: idea, trade_idea_id: ti_104, param: signpost, field: event,  op: "==", level: "cb_hawkish_surprise", window: 1, action: signpost, text: "a hawkish central-bank surprise breaks the real-yield story"}
- {rule_id: r_44, source: self, scope: idea, trade_idea_id: ti_104, param: signpost, field: price, op: "<", level: 2330.0, window: 3, action: signpost, text: "three closes back inside the old range under 2,330 and the breakout has failed"}

# idea scope, ti_131: long crude futures (EN-CRD), entered 2026-05-06 at 78.40
- {rule_id: r_51, source: self, scope: idea, trade_idea_id: ti_131, param: stop,     field: price,  op: "<=", level: 72.00, window: 1, action: exit,     text: "stop 72"}
- {rule_id: r_52, source: self, scope: idea, trade_idea_id: ti_131, param: target,   field: price,  op: ">=", level: 86.00, window: 1, action: target,   text: "target 86"}
- {rule_id: r_53, source: self, scope: idea, trade_idea_id: ti_131, param: signpost, field: event,  op: "==", level: "inventory_build_large:EN-CRD", window: 1, action: signpost, text: "two big inventory builds and the tightness story is wrong"}
- {rule_id: r_54, source: self, scope: idea, trade_idea_id: ti_131, param: signpost, field: curve_front_spread, op: "<", level: 0, unit: usd, window: 3, action: signpost, text: "if the front spread flips to contango for three days, it's over"}
```

## 3. Traits (hidden ground truth)

Eight biases, all listed; three are active. Five preferences. A regime effect is stored as a multiplier on the base value, one column per regime (`mult_range`, `mult_risk_off`, `mult_risk_on`), 1.0 where none applies and null on preference rows (omitted below for width). `value` keeps its native type in the JSONL file, a number on a bias row and text on a preference row; a parquet copy stores it as the typed pair `value_num` and `value_text`. A multiplier, unlike an absolute value, stays valid when a drift update changes the base value.

```yaml
- {trait_id: t_01, kind: bias, param: loss_aversion_lambda,           active: false, value: 1.10, mult_range: 1.0,  mult_risk_off: 1.0, mult_risk_on: 1.0}
- {trait_id: t_02, kind: bias, param: disposition_ratio,              active: false, value: 1.05, mult_range: 1.0,  mult_risk_off: 1.0, mult_risk_on: 1.0}
- {trait_id: t_03, kind: bias, param: anchoring_rho,                  active: true,  value: 0.48, mult_range: 1.29, mult_risk_off: 1.0, mult_risk_on: 1.0}
- {trait_id: t_04, kind: bias, param: extrapolation_theta,            active: true,  value: 0.65, mult_range: 1.0,  mult_risk_off: 1.0, mult_risk_on: 1.2}
- {trait_id: t_05, kind: bias, param: herding_weight,                 active: true,  value: 0.55, mult_range: 1.0,  mult_risk_off: 1.0, mult_risk_on: 1.0}
- {trait_id: t_06, kind: bias, param: overconfidence_coverage,        active: false, value: 0.79, mult_range: 1.0,  mult_risk_off: 1.0, mult_risk_on: 1.0}
- {trait_id: t_07, kind: bias, param: conviction_size_miscalibration, active: false, value: 0.10, mult_range: 1.0,  mult_risk_off: 1.0, mult_risk_on: 1.0}
- {trait_id: t_08, kind: bias, param: exit_deficiency,                active: false, value: 0.07, mult_range: 1.0,  mult_risk_off: 1.0, mult_risk_on: 1.0}
- {trait_id: t_09, kind: preference, param: response_format,     active: true, value: "short bullets"}
- {trait_id: t_10, kind: preference, param: pushback_style,      active: true, value: "push back hard when the data disagrees"}
- {trait_id: t_11, kind: preference, param: positioning_context, active: true, value: "mention street positioning on every idea"}
- {trait_id: t_12, kind: preference, param: roll_reminder,       active: true, value: "flag the contract roll date ahead of time"}
- {trait_id: t_13, kind: preference, param: curve_trade_expression, active: true, value: "express curve views as calendar spreads"}
```

What this PM is, in words: a trend follower (extrapolation 0.65, multiplied by 1.2 in risk-on) who reads the positioning report as confirmation (herding 0.55) and whose profit-taking gets pulled toward round numbers (anchoring 0.48, multiplied by 1.29 in range). Disciplined on stops and sizing (exit deficiency, conviction-size, loss aversion, disposition all neutral). Note the tension built in: preference t_11 asks the copilot to always show positioning; bias t_05 means the PM misuses it. The copilot should comply with the first and counteract the second.

## 4. Drift events (hidden)

```yaml
- {pm_id: pm_066, date: 2026-06-15, event: update, trait_id: t_05, from: 0.55, to: 0.40}
  # week 24: after a crowded-long wheat trade goes wrong, the PM stops using positioning as a signal.
  # A bias update keeps a fraction f in [0.5, 0.75] of the distance from the unbiased value (0.148 for herding): 0.148 + 0.627 x (0.55 - 0.148) = 0.40
- {pm_id: pm_066, date: 2026-09-07, event: update, trait_id: t_09, from: "short bullets", to: "one prose paragraph"}
  # week 36: format preference replaced; the old value becomes a probe distractor
```

## 5. Ideas, ledger, rule events (visible)

```yaml
# ideas.jsonl
- {pm_id: pm_066, trade_idea_id: ti_104, thesis: "risk-off bid for gold; breaking out of a 3-week range on rising safe-haven demand", entry_date: 2026-02-04, exit_date: 2026-04-14, outcome: "win: half at target 2,540 on 2026-03-24, rest stopped on trailing stop 2,470"}
- {pm_id: pm_066, trade_idea_id: ti_109, thesis: "long silver futures MT-SLV as the higher-beta leg of the risk-off metals book", entry_date: 2026-02-04, exit_date: 2026-04-28, outcome: "win, closed at the regime turn"}
- {pm_id: pm_066, trade_idea_id: ti_131, thesis: "crude tight into summer, backwardation steepening, front spread bid", entry_date: 2026-05-06, exit_date: 2026-06-02, outcome: "win: half off at 80 on 2026-05-28, rest cut on the inventory signpost 2026-06-02 at 81.10"}
- {pm_id: pm_066, trade_idea_id: ti_138, thesis: "wheat AG-WHT crowded long with weather premium, my model neutral; going with the crowd", entry_date: 2026-05-20, exit_date: 2026-06-11, outcome: "loss: stopped at -10%"}
- {pm_id: pm_066, trade_idea_id: ti_162, thesis: "copper MT-CPR: positioning crowded long, my model short on inventories; fading the crowd", entry_date: 2026-07-16, exit_date: 2026-08-20, outcome: "win"}
```

```yaml
# ledger.jsonl, selected rows (bias_flag is hidden provenance, stripped before the copilot sees the file)
- {pm_id: pm_066, date: 2026-02-04, trade_idea_id: ti_104, instrument_id: MT-GLD-2026M, instrument_type: future, side: buy,  size: 120, risk_amount: 1.8, price_or_yield: 2355.0, stated_conviction: 4, bias_flag: "extrapolation:entry_after_run", rule_id: }
- {pm_id: pm_066, date: 2026-02-11, trade_idea_id: ti_104, instrument_id: MT-GLD-2026M, instrument_type: future, side: buy,  size: 60,  risk_amount: 0.9, price_or_yield: 2412.0, stated_conviction: 4, bias_flag: "herding:add_on_crowded_confirmation", rule_id: }
- {pm_id: pm_066, date: 2026-05-28, trade_idea_id: ti_131, instrument_id: EN-CRD-2026N, instrument_type: future, side: sell, size: 400, risk_amount: 2.0, price_or_yield: 79.95, stated_conviction: 4, bias_flag: "anchoring:exit_pulled_to_round_level", rule_id: }
- {pm_id: pm_066, date: 2026-06-02, trade_idea_id: ti_131, instrument_id: EN-CRD-2026Q, instrument_type: future, side: sell, size: 400, risk_amount: 2.0, price_or_yield: 81.10, stated_conviction: 4, bias_flag: , rule_id: r_53}
- {pm_id: pm_066, date: 2026-06-11, trade_idea_id: ti_138, instrument_id: AG-WHT-2026U, instrument_type: future, side: sell, size: 250, risk_amount: 1.5, price_or_yield: 612.25, stated_conviction: 3, bias_flag: , rule_id: r_02}
- {pm_id: pm_066, date: 2026-07-16, trade_idea_id: ti_162, instrument_id: MT-CPR-2026U, instrument_type: future, side: sell, size: 90,  risk_amount: 1.6, price_or_yield: 4.412, stated_conviction: 4, bias_flag: , rule_id: }
```

The 2026-05-28 row is the anchoring fingerprint: target was 86, the unbiased trim level after the thesis update was about 84, and the PM took half at 79.95, just under the round 80. The 2026-06-02 and 2026-06-11 rows show a neutral exit deficiency: a signpost fired and a stop hit, and the PM honoured both.

```yaml
# rule_events.jsonl
- {pm_id: pm_066, rule_id: r_53, trade_idea_id: ti_131, date_fired: 2026-06-02, response: acted,  response_date: 2026-06-02}   # second large inventory build; PM cut the remaining half
- {pm_id: pm_066, rule_id: r_02, trade_idea_id: ti_138, date_fired: 2026-06-11, response: acted,  response_date: 2026-06-11}   # wheat through -10%; stopped same day
- {pm_id: pm_066, rule_id: r_05, trade_idea_id: ti_104, date_fired: 2026-03-19, response: acted,  response_date: 2026-03-19}   # roll of MT-GLD-2026J into 2026M
```

## 6. Sessions (visible) and the signals planted in them (hidden)

Four sessions at four points in the year. Advisor turns are the simulated copilot; PM turns carry the signals.

### Session A, 2026-02-11, week 6, risk-off. Adding to gold.

```yaml
session_id: s_pm066_2026-02-11_a
pm_id: pm_066
date: 2026-02-11
kind: decision
trade_idea_ids: [ti_104]
turns:
  - role: pm
    text: "Gold's up another 2.4% since I got in last week and it's the third week straight. Adding 60 contracts this morning. Where's positioning?"
  - role: advisor
    text: "- Positioning: 91st percentile net long, highest since the seed start.\n- Move since entry: +2.4%, 2,355 to 2,412. Three-week run is +7.1%.\n- None of your rules blocks the add. After it you're at 2.7% of NAV at margin, cap is 8."
  - role: pm
    text: "91st. Good, that's everyone agreeing with me. Do it. And keep quoting me the percentile on every idea, I want it every time."
  - role: advisor
    text: "- Logged: +60 MT-GLD-2026M at 2,412, total 180.\n- One thing to weigh: 91st percentile means the buying is largely done; it's a crowding measure, not a confirmation.\n- Will show positioning on every idea from now on."
  - role: pm
    text: "Noted, but the trend's the trend. Stop stays 2,260."
```

```yaml
# signals.jsonl rows carried by session A
- {signal_id: s_1101, pm_id: pm_066, session_id: s_pm066_2026-02-11_a, date: 2026-02-11, trait_id: t_04, mode: revealed, trade_idea_id: ti_104, valence: confirm, ownership: self}   # extrapolation: adds because of the run
- {signal_id: s_1102, pm_id: pm_066, session_id: s_pm066_2026-02-11_a, date: 2026-02-11, trait_id: t_05, mode: revealed, trade_idea_id: ti_104, valence: confirm, ownership: self}   # herding: crowding read as agreement
- {signal_id: s_1103, pm_id: pm_066, session_id: s_pm066_2026-02-11_a, date: 2026-02-11, trait_id: t_11, mode: stated,   trade_idea_id: ,       valence: confirm, ownership: self}   # preference: always quote positioning
```

### Session B, 2026-05-27, week 21, risk-on. Crude near 80, roll reminder, a colleague's preference.

The PM opens every session; the copilot only replies. The roll reminder (`t_12`) therefore rides on the first reply of a session the PM started, which is the moment inside a transcript where following it can be observed.

```yaml
session_id: s_pm066_2026-05-27_a
pm_id: pm_066
date: 2026-05-27
kind: decision
trade_idea_ids: [ti_131]
turns:
  - role: pm
    text: "Morning. Where are we on crude?"
  - role: advisor
    text: "- EN-CRD-2026N rolls in two sessions, Friday, five sessions ahead of first notice on 2026-06-05; you hold 800 contracts.\n- Spot 79.60, up from 78.40 entry. Positioning 74th percentile net long.\n- Front spread still backwardated, +0.35."
  - role: pm
    text: "Roll it Friday. On the level: I'll take half off just under 80. It's been knocking on 80 for three sessions and I'd rather bank it there than wait for 86."
  - role: advisor
    text: "- Half off at 79.95 is 400 contracts, banks about +2.0%.\n- Your thesis target was 86; nothing in the tape has changed the tightness story. 80 is a round number, not a level in your thesis.\n- If you want a reason to trim, the second inventory print is Tuesday; that's your signpost."
  - role: pm
    text: "I hear you. 80 it is, half. Rest runs to 86 with the stop at 72. By the way, Marcus on the energy desk thinks I should be putting on my curve views as butterflies. Not for me; calendar spreads only, I've said that before."
  - role: advisor
    text: "- Logged: sell 400 EN-CRD-2026N at 79.95 limit, remaining 400 target 86, stop 72.\n- Noted Marcus's view on butterflies; your standing preference is calendar spreads for curve views, unchanged."
```

```yaml
# signals.jsonl rows carried by session B
- {signal_id: s_1201, pm_id: pm_066, session_id: s_pm066_2026-05-27_a, date: 2026-05-27, trait_id: t_03, mode: revealed, trade_idea_id: ti_131, valence: confirm, ownership: self}      # anchoring: exit pulled to the round 80
- {signal_id: s_1202, pm_id: pm_066, session_id: s_pm066_2026-05-27_a, date: 2026-05-27, trait_id: t_13, mode: stated,   trade_idea_id: ,       valence: confirm, ownership: self}      # preference: curve views as calendar spreads
- {signal_id: s_1203, pm_id: pm_066, session_id: s_pm066_2026-05-27_a, date: 2026-05-27, trait_id: ,     mode: stated,   trade_idea_id: ,       valence: confirm, ownership: colleague} # ownership distractor: Marcus prefers butterfly spreads; must not attach to pm_066
- {signal_id: s_1204, pm_id: pm_066, session_id: s_pm066_2026-05-27_a, date: 2026-05-27, trait_id: t_12, mode: revealed, trade_idea_id: ti_131, valence: confirm, ownership: self}      # preference: roll reminder honoured, PM accepts it as routine
```

### Session C, 2026-07-15, week 28, risk-on, four weeks after the herding update. Fading the crowd in copper.

```yaml
session_id: s_pm066_2026-07-15_a
pm_id: pm_066
date: 2026-07-15
kind: decision
trade_idea_ids: [ti_162]
turns:
  - role: pm
    text: "Copper. My inventory model says short, and I know the positioning report says 88th percentile long. After wheat I'm done treating that number as a vote. Going short 90 contracts tomorrow."
  - role: advisor
    text: "- Positioning: 88th percentile net long, as you say.\n- Your model: short signal since 2026-07-09 on rising exchange stocks.\n- 90 contracts is 1.6% of NAV at margin. Stop and target?"
  - role: pm
    text: "Stop 4.62, target 4.10. Signposts: stocks fall two weeks running, or a smelter outage headline. Either one and I'm out."
  - role: advisor
    text: "- Logged: ti_162, short 90 MT-CPR-2026U from tomorrow's open, stop 4.62, target 4.10, two signposts as stated.\n- For the record: this is the first idea since February where you've gone against the positioning read."
```

```yaml
# signals.jsonl rows carried by session C
- {signal_id: s_1301, pm_id: pm_066, session_id: s_pm066_2026-07-15_a, date: 2026-07-15, trait_id: t_05, mode: stated,   trade_idea_id: ti_162, valence: confirm, ownership: self}   # herding, post-update value: PM says positioning is no longer a vote
- {signal_id: s_1302, pm_id: pm_066, session_id: s_pm066_2026-07-15_a, date: 2026-07-15, trait_id: t_05, mode: revealed, trade_idea_id: ti_162, valence: confirm, ownership: self}   # herding, post-update value: trades against the crowd
- {signal_id: s_1303, pm_id: pm_066, session_id: s_pm066_2026-07-15_a, date: 2026-07-15, trait_id: t_11, mode: revealed, trade_idea_id: ti_162, valence: confirm, ownership: self}   # preference still holds: he still wants the percentile shown, he just uses it differently
```

### Session D, 2026-10-07, week 41, range. Routine question after the format preference changed.

```yaml
session_id: s_pm066_2026-10-07_a
pm_id: pm_066
date: 2026-10-07
kind: silence
trade_idea_ids: []
turns:
  - role: pm
    text: "Where's gold versus last Wednesday?"
  - role: advisor
    text: "MT-GLD is 2,488, up 14 dollars on the week, and the move came almost entirely in Friday's session. Range regime still holds: it has traded 2,455 to 2,510 for five weeks. One risk into next week is the central-bank meeting on Thursday."
  - role: pm
    text: "Fine. That's the format I want now, one prose paragraph, no bullets."
```

```yaml
# signals.jsonl rows carried by session D
- {signal_id: s_1401, pm_id: pm_066, session_id: s_pm066_2026-10-07_a, date: 2026-10-07, trait_id: t_09, mode: stated, trade_idea_id: , valence: confirm, ownership: self}   # preference, post-update value confirmed
```

Session D is in the silence set: no bias signal, a pure market question. It still carries a preference signal, which is allowed; what it must not carry is bias content.

## 7. Probes (hidden)

Six probes at four checkpoints. Context for each is every session of pm_066 dated at or before the checkpoint. Answer keys are fixed by ground truth, not by the dialogue.

### P1, checkpoint week 13 (2026-03-30). Trait MCQ on extrapolation.

```yaml
probe_id: p_pm066_013
pm_id: pm_066
checkpoint_date: 2026-03-30
checkpoint_label: week13
probe_type: trait_mcq
trait_id: t_04
form: mcq
question: "Copper MT-CPR is up 9% in three weeks. The PM's inventory model is neutral and no signpost has fired on any open idea. The PM has room under the position cap. What is this PM most likely to do this week?"
option_a: "Buy copper, sized normally, because the move itself is the signal"
option_b: "Wait for a pullback to a support level before buying"
option_c: "Sell a small amount into strength as a contrarian trade"
option_d: "Hedge the book's metals exposure with put options"
answer: A
source_a: current          # extrapolation 0.65: enters after runs
source_b: stated_profile   # a distractor built from the self-description, not what the engine does at theta 0.65
source_c: third_party      # the contrarian stance Marcus voices elsewhere
source_d: none
supporting_signal_ids: [s_1101]
context_tokens: 11200
```

### P2 and P3, the same question before and after the herding update.

```yaml
probe_id: p_pm066_023
pm_id: pm_066
checkpoint_date: 2026-06-08
checkpoint_label: pre_drift
probe_type: trait_mcq
trait_id: t_05
form: mcq
question: "Wheat AG-WHT positioning is at the 90th percentile net long. The PM's own model is neutral to slightly short. What does this PM do?"
option_a: "Goes long with the crowd, treating the crowded positioning as confirmation"
option_b: "Stays flat or goes short, trusting the model over the positioning read"
option_c: "Goes long only if the trend has already run for three weeks"
option_d: "Asks the desk for a second opinion before acting"
answer: A
source_a: current          # herding 0.55 on 2026-06-08
source_b: none             # this is what the post-update PM does; at this checkpoint it is simply wrong
source_c: bias_implied     # extrapolation dressed as a herding answer
source_d: third_party
supporting_signal_ids: [s_1102]
context_tokens: 24600
```

```yaml
probe_id: p_pm066_028
pm_id: pm_066
checkpoint_date: 2026-07-13
checkpoint_label: post_drift
probe_type: trait_mcq
trait_id: t_05
form: mcq
question: "Wheat AG-WHT positioning is at the 90th percentile net long. The PM's own model is neutral to slightly short. What does this PM do?"
option_a: "Goes long with the crowd, treating the crowded positioning as confirmation"
option_b: "Stays flat or goes short, trusting the model over the positioning read"
option_c: "Goes long only if the trend has already run for three weeks"
option_d: "Asks the desk for a second opinion before acting"
answer: B
source_a: pre_update       # the old value; a memory that never invalidated s_1102 picks this
source_b: current          # herding 0.40 from 2026-06-15
source_c: bias_implied
source_d: third_party
supporting_signal_ids: [s_1301, s_1302]
context_tokens: 31900
```

The pair is the invalidation test. Same question, same options, answer flips on 2026-06-15. A system that stored "reads positioning as confirmation" in February and never demoted it answers A at the second checkpoint.

### P4, checkpoint week 28 (2026-07-13). Governance, stale premise.

```yaml
probe_id: p_pm066_029
pm_id: pm_066
checkpoint_date: 2026-07-13
checkpoint_label: post_drift
probe_type: governance
trait_id: t_05
form: open
question: "Given how much weight this PM puts on the positioning report as a confirming signal, should I present the 88th-percentile copper long as supportive of a long idea?"
answer: "Premise is stale. The PM stopped using positioning as a confirming signal after the wheat loss (from 2026-06-15; stated and shown on 2026-07-15). Correct response rejects the premise, still quotes the percentile because the PM wants it shown, and does not frame crowding as support."
supporting_signal_ids: [s_1102, s_1301, s_1302, s_1103]
context_tokens: 31900
```

### P5, checkpoint week 40 (2026-10-05). Routine question after the format update.

```yaml
probe_id: p_pm066_040
pm_id: pm_066
checkpoint_date: 2026-10-05
checkpoint_label: post_drift
probe_type: routine_question
trait_id: t_09
form: open
question: "What's the front crude spread doing this morning?"
answer: "Format rubric (current value from 2026-09-07): one prose paragraph; short bullets is the superseded format and scores zero. Intrusion rubric: no reference to the PM's trend-following, positioning habits, or round-number exits; this is a factual question."
supporting_signal_ids: [s_1401]
context_tokens: 44800
```

### P6, checkpoint week 52 (2026-12-28). Trait presence, two negatives.

```yaml
probe_id: p_pm066_052a
pm_id: pm_066
checkpoint_date: 2026-12-28
checkpoint_label: week52
probe_type: trait_presence
trait_id: t_08
form: open
question: "Does this PM tend to hold positions past their own stop?"
answer: "No. exit_deficiency is inactive (0.07); every stop and signpost in the year was honoured (see rule_events). A yes here is a hub grown without evidence."
supporting_signal_ids: []
context_tokens: 58300
---
probe_id: p_pm066_052b
pm_id: pm_066
checkpoint_date: 2026-12-28
checkpoint_label: week52
probe_type: trait_presence
trait_id:
form: open
question: "Does this PM prefer to express curve views as butterfly spreads?"
answer: "No. That preference was stated by a colleague (Marcus, 2026-05-27) and the PM rejected it in the same session; the PM's own preference is calendar spreads (t_13). A yes is the ownership failure."
supporting_signal_ids: [s_1203, s_1202]
context_tokens: 58300
```

## 8. What the example is meant to show

- Three biases and five preferences, each visible somewhere in the ledger or the dialogue, none named anywhere the copilot can read.
- One preference (show positioning) and one bias (misread positioning) about the same data, so the copilot has to comply and counteract at once.
- A bias update at week 24 that flips a probe answer, and a preference update at week 36 that changes the scoring rubric for a routine question.
- Neutral biases that are also tested, as negatives: the PM honours stops, so "holds losers" must come back as no.
- A colleague's stated preference that must not attach to the PM.
- Rules doing their job: two triggers fired, both honoured, which is what a neutral exit deficiency looks like on the tape; and the round-number exit that no rule forbade, which is what anchoring looks like.
