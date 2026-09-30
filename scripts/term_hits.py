"""Count how often each banned stance word and transcript phrase hits PM turns in dialogue_logs.

Prints, per term, the number of PM turns it matches and a few matches in context, so
the transcript phrase list can be tuned against real dialogue rather than guesses.
Usage: uv run python scripts/term_hits.py [--config PATH] [--data-dir PATH] [--samples N]
"""

import argparse
import re
from pathlib import Path

from pm_traitbench.catalogues.loader import BANNED_STANCE_WORDS, TRANSCRIPT_BANNED_PHRASES
from pm_traitbench.config import load_config
from pm_traitbench.enums import TurnRole
from pm_traitbench.tables.specs import DIALOGUE_LOGS
from pm_traitbench.tables.store import DataStore

_CONTEXT_CHARS = 50


def _contexts(pattern: re.Pattern[str], texts: list[str]) -> list[str]:
    hits = []
    for text in texts:
        for match in pattern.finditer(text):
            start = max(0, match.start() - _CONTEXT_CHARS)
            end = min(len(text), match.end() + _CONTEXT_CHARS)
            hits.append(text[start:end].replace("\n", " "))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    store = DataStore(args.data_dir, load_config(args.config).output)
    texts = [
        turn.text.lower()
        for log in store.read(DIALOGUE_LOGS)
        for turn in log.turns
        if turn.role == TurnRole.PM
    ]
    print(f"{len(texts)} PM turns")

    # Old words as the substring scan the grep used to run; new phrases as it runs now.
    lists = (
        ("banned stance words (substring)", BANNED_STANCE_WORDS, ""),
        ("transcript phrases (word start)", TRANSCRIPT_BANNED_PHRASES, r"\b"),
    )
    for title, terms, lead in lists:
        print(f"\n== {title}")
        for term in terms:
            hits = _contexts(re.compile(lead + re.escape(term)), texts)
            print(f"{term!r}: {len(hits)} hits")
            for context in hits[: args.samples]:
                print(f"    ...{context}...")


if __name__ == "__main__":
    main()
