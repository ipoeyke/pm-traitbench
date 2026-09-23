"""Coverage range for the real-market fixed event date lists.

A leaf module with no internal imports, so both `config.py` and
`real/events.py` can read the same constant without a circular import.
"""

from datetime import date

# The first and last date the FOMC, WASDE and NFP lists in real/events.py
# are complete for.
EVENT_DATES_COVER: tuple[date, date] = (date(2018, 6, 4), date(2019, 5, 31))
