

from fastapi import Request

async def get_db_pool(request: Request):
    return request.state.db_pool