from pydantic import BaseModel
from typing import Optional


class DBWriteRequest(BaseModel):
    data_tuples: list
    all_columns: list