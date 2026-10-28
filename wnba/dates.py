"""US game dates.

Every game date in the data and the logs is a US (Eastern) date. Computing
"today" from the local clock is wrong for a user in Europe: after local
midnight on a US game night the local date is already the next day. These
helpers use New York time, including daylight-saving changes.
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


def us_today() -> date:
    """Today's date in New York -- the default game date."""
    return datetime.now(ET).date()


def current_season(today: date | None = None) -> int:
    """The WNBA season in progress or most recently finished.

    Seasons run May-October inside one calendar year, so January-April still
    belongs to the previous season.
    """
    today = today or us_today()
    return today.year if today.month >= 5 else today.year - 1
