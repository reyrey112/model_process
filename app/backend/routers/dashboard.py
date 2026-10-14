from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.backend.dependencies import get_db_pool
from app.backend.services.timerange import resolve_range
from app.backend.services.timeseries import (
    PARAMETER_COLUMNS,
    fetch_timeseries,
    resolve_bucket_seconds,
)

router = APIRouter(prefix="/api", tags=["dashboard"])


@router.get("/parameters")
async def list_parameters():
    """
    Returns the full list of available parameter columns, so the frontend
    can build its checkboxes without hardcoding column names in two places.
    """
    return {"parameters": PARAMETER_COLUMNS}


@router.get("/data")
async def get_data(
    columns: List[str] = Query(
        ..., description="Repeat this param per column, e.g. ?columns=quality&columns=state_reactor_temp"
    ),
    preset: Optional[str] = Query(None, description="One of: 1h, 12h, 1d, 1w"),
    start: Optional[str] = Query(None, description="datetime-local string, required if no preset"),
    end: Optional[str] = Query(None, description="datetime-local string, required if no preset"),
    db_pool=Depends(get_db_pool),
):
    # preset takes priority if both preset and start/end are somehow sent together
    try:
        start_dt, end_dt = resolve_range(preset=preset, start_str=start, end_str=end)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        series = await fetch_timeseries(db_pool, start_dt, end_dt, columns)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    return {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "bucket_seconds": resolve_bucket_seconds(start_dt, end_dt),
        "series": series,
    }