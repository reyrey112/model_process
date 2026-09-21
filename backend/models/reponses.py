from pydantic import BaseModel
from typing import Optional
from fastapi import HTTPException

class DBWriteResponse(BaseModel):
    status: HTTPException
    