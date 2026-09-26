import os
import httpx, logging, time
import boto3
from dotenv import load_dotenv

load_dotenv()


if os.environ.get("LOCAL") == "yes":
    API_BASE = "http://localhost:8000"
else:
    API_BASE = os.environ.get("API_BASE")
    
# API_AUDIENCE = os.environ.get("API_AUDIENCE", "http://localhost:8000")
# TIMEOUT = 120
# _cache = {"token": None, "exp": 0.0}

# def _get_identity_token() -> str | None:
#     if not API_AUDIENCE or "localhost" in API_AUDIENCE or "127.0.0.1" in API_AUDIENCE:
#         return None

#     # Reuse the token until 60s before expiry
#     if _cache["token"] and time.time() < _cache["exp"] - 60:
#         return _cache["token"]

#     try:
#         resp = boto3.client("sts").get_web_identity_token(
#             Audience=[API_AUDIENCE],
#             SigningAlgorithm="ES384",
#             DurationSeconds=300,
#         )
#         _cache["token"] = resp["WebIdentityToken"]
#         _cache["exp"] = time.time() + 300
#         return _cache["token"]
#     except Exception:
#         logging.warning("Failed to fetch identity token", exc_info=True)
#         return None

# def _headers() -> dict:
#     token = _get_identity_token()
#     return {"Authorization": f"Bearer {token}"} if token else {}


def db_write(data_tuples: list, all_columns: list, headers) -> dict:
    r = httpx.post(
        f"{API_BASE}/db/write",
        json={"data_tuples": data_tuples, "all_columns": all_columns},
        headers=headers,
        # timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()
