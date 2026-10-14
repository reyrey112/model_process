from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")

PRESETS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}


def now_eastern_naive() -> datetime:
    """
    Current time as a naive datetime representing Eastern local time
    (handles EST/EDT automatically), matching how the DB's naive
    TIMESTAMP column is implicitly interpreted.
    """
    return datetime.now(EASTERN).replace(tzinfo=None)


def resolve_preset(preset: str) -> tuple[datetime, datetime]:
    if preset not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset}'. Valid options: {list(PRESETS.keys())}"
        )
    end = now_eastern_naive()
    start = end - PRESETS[preset]
    return start, end


def _parse_datetime_local(value: str) -> datetime:
    """
    Parses an HTML datetime-local value (e.g. '2024-01-01T10:30' or
    '2024-01-01T10:30:00'). Raises ValueError with a clear message on
    anything malformed.
    """
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        # datetime-local sometimes omits seconds; retry with ":00" appended
        try:
            return datetime.fromisoformat(f"{value}:00")
        except ValueError:
            raise ValueError(f"Invalid datetime value: '{value}'")


def parse_custom_range(start_str: str, end_str: str) -> tuple[datetime, datetime]:
    start = _parse_datetime_local(start_str)
    end = _parse_datetime_local(end_str)

    if start >= end:
        raise ValueError("Start time must be before end time.")

    now = now_eastern_naive()
    if end > now:
        raise ValueError("End time cannot be in the future.")

    return start, end


def resolve_range(
    preset: str | None = None,
    start_str: str | None = None,
    end_str: str | None = None,
) -> tuple[datetime, datetime]:
    """
    Single entry point the API route calls. Exactly one of (preset) or
    (start_str + end_str) should be provided.
    """
    if preset:
        return resolve_preset(preset)

    if start_str and end_str:
        return parse_custom_range(start_str, end_str)

    raise ValueError(
        "Must provide either a 'preset' or both 'start' and 'end' values."
    )