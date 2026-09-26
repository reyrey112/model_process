import os, sys, yaml
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

from backend.models.requests import DBWriteRequest
from backend.models.reponses import DBWriteResponse
import psycopg
from typing import List
from fastapi import APIRouter, HTTPException, status, Header,Depends, BackgroundTasks
from pydantic import BaseModel, Field
import asyncpg
from dotenv import load_dotenv, find_dotenv
from backend.dependencies import get_db_pool
import secrets

load_dotenv(find_dotenv())
DATABASE_URL = os.environ.get("DATABASE_URL")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET")

def require_admin(x_admin_key: str = Header(...)):
    if not secrets.compare_digest(x_admin_key, ADMIN_SECRET):
        raise HTTPException(status_code=403, detail="Forbidden")



router = APIRouter(prefix="/db", dependencies=[Depends(require_admin)])

db_pool = None


@router.post("/write", response_model=DBWriteResponse)
async def write_to_db(
    request: DBWriteRequest,
    db_pool = Depends(get_db_pool),
):
    if not request.data_tuples:
        raise HTTPException(status_code=400, detail="The data tuples cannot be empty.")

    data_tuples = request.data_tuples
    all_columns = request.all_columns
    value_names_sql = ",".join(f'"{col}"' for col in all_columns)
    value_holders_sql = ",".join(f"${i+1}" for i in range(len(all_columns)))


    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool is unavailable.")

    query = f"""
        INSERT INTO process_data ({value_names_sql}) 
        VALUES ({value_holders_sql});
    """

    try:
        async with db_pool.acquire() as connection:
            await connection.executemany(query, data_tuples)
        return {"status": "success", "inserted_records": len(data_tuples)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database insertion failed: {str(e)}",
        )