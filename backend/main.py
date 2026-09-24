import os, sys
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))

if root_dir not in sys.path:
    sys.path.append(root_dir)

from fastapi import FastAPI
from routers import database
import asyncpg
from contextlib import asynccontextmanager




# 2. Modern Lifespan Handler (Replaces @app.on_event)

DATABASE_URL=os.environ.get("DATABASE_URL")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the pool
    database_url = DATABASE_URL
    db_pool = await asyncpg.create_pool(database_url, min_size=5, max_size=20)

    yield {"db_pool": db_pool}

    # Shutdown: Clean closure
    await db_pool.close()


app = FastAPI(title="Process Sim API", version="0.0.1", lifespan=lifespan)
app.include_router(database.router)

@app.get("/health")
def health():
    return {"status": "ok"}