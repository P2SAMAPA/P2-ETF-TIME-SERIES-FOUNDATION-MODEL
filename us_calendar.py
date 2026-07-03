"""
us_calendar.py — P2-ETF-TIME-SERIES-FOUNDATION-MODEL (TSFM)

Determines whether a given date is a US equity market trading day, so the
daily workflow can skip weekends/holidays cleanly.
"""

import datetime as dt

try:
    import pandas_market_calendars as mcal
    _HAS_MCAL = True
except ImportError:
    _HAS_MCAL = False

_FIXED_HOLIDAYS_FALLBACK = {
    # Fallback list used only if pandas_market_calendars is unavailable.
    # Not exhaustive for floating holidays (Thanksgiving, MLK, etc.) —
    # pandas_market_calendars should be the primary path.
}


def is_trading_day(date: dt.date = None) -> bool:
    """Return True if the given date (default: today, UTC) is a NYSE trading day."""
    if date is None:
        date = dt.datetime.utcnow().date()

    if date.weekday() >= 5:
        return False

    if _HAS_MCAL:
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(start_date=date, end_date=date)
        return not schedule.empty

    # Fallback: weekday check only if the calendar library isn't installed.
    return True


def last_n_trading_days(n: int, end_date: dt.date = None) -> list:
    """Return the last n NYSE trading days up to and including end_date."""
    if end_date is None:
        end_date = dt.datetime.utcnow().date()

    if _HAS_MCAL:
        nyse = mcal.get_calendar("NYSE")
        start_date = end_date - dt.timedelta(days=int(n * 2.2) + 10)
        schedule = nyse.schedule(start_date=start_date, end_date=end_date)
        days = [d.date() for d in schedule.index]
        return days[-n:]

    # Fallback: approximate using weekdays only.
    days = []
    current = end_date
    while len(days) < n:
        if current.weekday() < 5:
            days.append(current)
        current -= dt.timedelta(days=1)
    return list(reversed(days))


if __name__ == "__main__":
    today = dt.datetime.utcnow().date()
    print("Today:", today, "Trading day:", is_trading_day(today))
