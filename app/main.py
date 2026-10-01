import os, sys
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
frontend_dir = os.path.abspath(os.path.join(current_dir, "frontend"))

if root_dir not in sys.path:
    sys.path.append(root_dir)

# if frontend_dir not in sys.path:
#     sys.path.append(frontend_dir)    
print(current_dir)
from app.backend.routers import database
import asyncpg
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


# 2. Modern Lifespan Handler (Replaces @app.on_event)

DATABASE_URL=os.environ.get("DATABASE_URL")

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=5, max_size=20)
    yield {"db_pool": db_pool}
    await db_pool.close()


app = FastAPI(title="Process Sim API", version="0.0.1", lifespan=lifespan)
app.include_router(database.router)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

items = ["Buy milk", "Walk dog"]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {"items": items})
