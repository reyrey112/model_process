import hashlib
import inspect
import random
import subprocess

import numpy as np
import torch

import mlflow

def get_source_hash(obj) -> str:
    """Content-based version id for a function/class: identical source ->
    identical hash, any change (even whitespace) -> a new hash. Use this to
    tag runs with exactly which version of a loss/reward function, model
    class, or env step function produced them, independent of git state."""
    try:
        src = inspect.getsource(obj)
        return hashlib.sha256(src.encode()).hexdigest()[:10]
    except (OSError, TypeError):
        return "unavailable"


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_git_dirty() -> bool:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"]).decode()
        return len(status.strip()) > 0
    except Exception:
        return False


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_code_version_tags(loss_function, model_class):
    """Tags + source snapshots so you can always answer 'what code produced
    this run', even for uncommitted, in-flight edits to loss_function or the
    VAE architecture."""
    mlflow.set_tags(
        {
            "git_commit": get_git_commit(),
            "git_dirty": get_git_dirty(),
            "loss_function_hash": get_source_hash(loss_function),
            "vae_class_hash": get_source_hash(model_class),
        }
    )
    mlflow.log_text(inspect.getsource(loss_function), f"code_snapshots/loss_function.py")
    mlflow.log_text(inspect.getsource(model_class), f"code_snapshots/{model_class}_class.py")