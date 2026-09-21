from fastapi import HTTPException, status, Request
import asyncpg

async def get_db_pool(request) -> asyncpg.Pool:
    """
    Extracts the pool from the request app instance context.
    Works perfectly across multiple sub-routers.
    """
    pool = getattr(request.app.state, "db_pool", None)
    if not pool:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection pool is completely unavailable."
        )
    return pool