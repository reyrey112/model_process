"""
Hot-swappable ONNX model manager for a long-running inference loop.

How it works
------------
1. A background thread polls `get_target_version()` (a Redis key, or the MLflow
   registry alias) to learn which model version should be live.
2. When it changes, the thread downloads that version from the registry,
   builds a new ONNX Runtime session, checks that its inputs/outputs match the
   live model, and runs a smoke-test prediction (which also warms it up).
3. Only if all of that succeeds is the new session handed to the main loop.
   Any failure is logged and the current model keeps serving.
4. The main loop calls `manager.current()` once per iteration. That is the only
   place a swap happens, so each prediction is made by exactly one model and
   nothing is ever in flight while the swap occurs.

Downloading from Databricks needs DATABRICKS_HOST / DATABRICKS_TOKEN in the
environment (use a read-only service principal).
"""

import glob
import logging
import os
import shutil
import threading
import time

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

# For GPU use e.g. ["CUDAExecutionProvider", "CPUExecutionProvider"].
PROVIDERS = ["CPUExecutionProvider"]


# --------------------------------------------------------------------------
# Where do we learn the target version from?
# --------------------------------------------------------------------------
def redis_version_getter(r, key):
    """Version published in a Redis key (client must use decode_responses=True)."""
    return lambda: r.get(key)


def mlflow_alias_getter(model_name, alias="champion"):
    """Version a registry alias points to. Polls Databricks directly, so the
    training machine never needs to reach this server or its Redis."""
    client = None

    def _get():
        nonlocal client
        if client is None:
            import mlflow
            from mlflow import MlflowClient

            mlflow.set_registry_uri("databricks-uc")  # or "databricks"
            client = MlflowClient()
        return client.get_model_version_by_alias(model_name, alias).version

    return _get


# --------------------------------------------------------------------------
# Download / load / validate
# --------------------------------------------------------------------------
def fetch_onnx(model_name, version, model_dir):
    """Download a registered model version (once) and return its .onnx path."""
    final_dir = os.path.join(model_dir, f"v{version}")
    if not os.path.isdir(final_dir):
        import mlflow  # lazy: only needed when we actually download

        mlflow.set_registry_uri("databricks-uc")  # or "databricks"
        tmp_dir = final_dir + ".tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)
        mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}/{version}", dst_path=tmp_dir
        )
        os.rename(tmp_dir, final_dir)  # a version dir is either complete or absent
    paths = glob.glob(os.path.join(final_dir, "**", "*.onnx"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No .onnx file found under {final_dir}")
    return paths[0]


def _new_session(path):
    return ort.InferenceSession(path, providers=PROVIDERS)


def _signature(session):
    """Input/output names, dtypes and fixed dims (symbolic dims -> None)."""

    def norm(shape):
        return tuple(d if isinstance(d, int) else None for d in shape)

    return (
        [(i.name, i.type, norm(i.shape)) for i in session.get_inputs()],
        [(o.name, o.type, norm(o.shape)) for o in session.get_outputs()],
    )


def validate(new, old, sample_feed):
    """Raise if `new` is not a drop-in replacement for `old`."""
    if _signature(new) != _signature(old):
        raise ValueError(
            "Input/output signature differs from the live model: "
            f"{_signature(old)} -> {_signature(new)}"
        )
    outputs = new.run(None, sample_feed)  # also warms the session up
    if not all(np.all(np.isfinite(o)) for o in outputs):
        raise ValueError("New model produced NaN/inf on the sample input")


# --------------------------------------------------------------------------
# The manager
# --------------------------------------------------------------------------
class ModelManager:
    def __init__(
        self,
        model_name,
        model_dir,
        get_target_version,
        sample_feed,
        fallback_path=None,
        poll_interval=10,
    ):
        """
        model_name         registered model name, e.g. "catalog.schema.my_model"
        model_dir          local cache dir; each version lands in <model_dir>/v<N>
        get_target_version callable returning the version that should be live
        sample_feed        {input_name: array} used to smoke-test new models
        fallback_path      local .onnx to use if no published version can be loaded
        poll_interval      seconds between version checks
        """
        self._model_name = model_name
        self._model_dir = model_dir
        self._get_target = get_target_version
        self._sample_feed = sample_feed
        self._lock = threading.Lock()
        self._pending = None  # (version, session) validated and waiting to swap in
        self._bad = set()  # versions that failed validation; never retried
        self._last_prepared = None

        self.session = None
        self.version = None
        try:
            target = get_target_version()
            if target:
                target = str(target)
                path = fetch_onnx(model_name, target, model_dir)
                self.session, self.version = _new_session(path), target
                self._last_prepared = target
        except Exception:
            log.exception("Could not load the published model at startup")

        if self.session is None:
            if not fallback_path:
                raise RuntimeError("No model could be loaded and no fallback_path given")
            log.warning("Starting on fallback model %s", fallback_path)
            self.session, self.version = _new_session(fallback_path), "fallback"

        threading.Thread(target=self._watch, args=(poll_interval,), daemon=True).start()

    def _watch(self, interval):
        while True:
            time.sleep(interval)
            try:
                target = self._get_target()
                if not target:
                    continue
                target = str(target)
                if target == self._last_prepared or target in self._bad:
                    continue

                log.info("New model version %s detected; preparing it", target)
                path = fetch_onnx(self._model_name, target, self._model_dir)  # may retry
                try:
                    session = _new_session(path)
                    validate(session, self.session, self._sample_feed)
                except Exception:
                    self._bad.add(target)  # versions are immutable: don't retry
                    raise

                with self._lock:
                    self._pending = (target, session)
                self._last_prepared = target
            except Exception:
                log.exception("Model update check failed; keeping the current model")

    def current(self):
        """Call once per loop iteration; returns the session for this iteration."""
        with self._lock:
            pending, self._pending = self._pending, None
        if pending:
            self.version, self.session = pending
            log.info("Swapped to model version %s", self.version)
        return self.session