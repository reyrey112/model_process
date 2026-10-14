from fastapi import APIRouter, Request

from app.backend.templating import templates
from app.backend.services.timeseries import PARAMETER_COLUMNS
from app.backend.services.timerange import PRESETS

router = APIRouter(tags=["pages"])

DEFAULT_PARAMETER = "quality"
DEFAULT_PRESET = "1h"


@router.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "parameters": PARAMETER_COLUMNS,
        "presets": list(PRESETS.keys()),
        "default_parameter": DEFAULT_PARAMETER,
        "default_preset": DEFAULT_PRESET,
    })