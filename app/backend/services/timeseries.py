import os
from datetime import datetime
from typing import Optional

from app.backend.models.requests import ProcessDataRow

POSTGRES_TABLE = os.environ.get("POSTGRES_TABLE")
TIME_COLUMN = "Timestamp"

# Derived from the same model used for writes, so read/write
# never drift out of sync if you add/remove a parameter later.
PARAMETER_COLUMNS = [
    col for col in ProcessDataRow.model_fields.keys() if col != TIME_COLUMN
]

# "Nice" bucket widths, in seconds, smallest to largest.
_NICE_BUCKET_SECONDS = [
    1, 5, 10, 30,
    60, 5 * 60, 15 * 60, 30 * 60,
    60 * 60, 6 * 60 * 60, 24 * 60 * 60,
]

MAX_POINTS = 2000


def resolve_bucket_seconds(
    start: datetime, end: datetime, max_points: int = MAX_POINTS
) -> Optional[int]:
    """
    Decide raw vs. bucketed resolution for a given range.
    Returns None for raw data, or the smallest "nice" bucket width
    (seconds) that keeps the point count at or under max_points.
    Assumes the known ~1 row/sec ingestion rate.
    """
    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        return None

    estimated_raw_points = total_seconds
    if estimated_raw_points <= max_points:
        return None

    required_bucket_seconds = total_seconds / max_points
    for bucket in _NICE_BUCKET_SECONDS:
        if bucket >= required_bucket_seconds:
            return bucket

    return _NICE_BUCKET_SECONDS[-1]


def _validate_columns(columns: list[str]) -> list[str]:
    """
    Whitelists requested columns against ProcessDataRow's declared fields.
    This is what makes it safe to interpolate column names directly into
    SQL — asyncpg can't parameterize identifiers, so we never accept
    arbitrary column names from the client.
    """
    if not columns:
        raise ValueError("At least one column must be requested.")
    invalid = [c for c in columns if c not in PARAMETER_COLUMNS]
    if invalid:
        raise ValueError(f"Unknown column(s): {invalid}")
    return columns


def _build_query(columns: list[str], bucket_seconds: Optional[int]) -> str:
    quoted_time = f'"{TIME_COLUMN}"'

    if bucket_seconds is None:
        quoted_cols = ", ".join(f'"{c}"' for c in columns)
        return f"""
            SELECT {quoted_time}, {quoted_cols}
            FROM {POSTGRES_TABLE}
            WHERE {quoted_time} BETWEEN $1 AND $2
            ORDER BY {quoted_time}
        """

    bucket_expr = f"time_bucket('{bucket_seconds} seconds', {quoted_time})"
    agg_cols = ", ".join(f'AVG("{c}") AS "{c}"' for c in columns)
    return f"""
        SELECT {bucket_expr} AS "{TIME_COLUMN}", {agg_cols}
        FROM {POSTGRES_TABLE}
        WHERE {quoted_time} BETWEEN $1 AND $2
        GROUP BY 1
        ORDER BY 1
    """


async def fetch_timeseries(
    pool,
    start: datetime,
    end: datetime,
    columns: list[str],
) -> dict[str, list[dict]]:
    """
    Core read function. No FastAPI/Request dependency — takes a pool and
    plain Python values so it's reusable and easy to test on its own.

    Returns: { "column_name": [{"t": iso_string, "v": float|None}, ...], ... }
    """
    columns = _validate_columns(columns)
    bucket_seconds = resolve_bucket_seconds(start, end)
    query = _build_query(columns, bucket_seconds)

    async with pool.acquire() as connection:
        rows = await connection.fetch(query, start, end)

    series: dict[str, list[dict]] = {c: [] for c in columns}
    for row in rows:
        ts = row[TIME_COLUMN].isoformat()
        for c in columns:
            series[c].append({"t": ts, "v": row[c]})

    return series