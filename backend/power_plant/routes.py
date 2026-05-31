"""
Boiler Tube Early Leak Detection — FastAPI Backend
Model repo: ZOROD/Boiler_Tube_Early_leak_detection (HuggingFace)

Endpoints
─────────
GET  /health              → service + model-load status
POST /predict             → single-row or batch inference (JSON body)
POST /predict/csv         → batch inference from uploaded CSV
                            ?faults_only=true  returns only alarm=True rows
GET  /models/info         → loaded model metadata
GET  /threshold           → current decision threshold
PUT  /threshold           → override the decision threshold

NEW endpoints (v1.2.0)
─────────────────────
POST /rca                         → Root Cause Analysis + per-sensor blame scores
POST /optimize/sensor             → Sensor optimisation recommendations
POST /energy/analysis             → Energy efficiency analysis
POST /simulate                    → What-if single-point simulation
POST /predict/csv/optimized       → CSV bulk upload + per-alarm optimisation
POST /explain/shap                → SHAP feature importance (XGBoost native)
POST /lead-time                   → Time-to-failure estimation
GET  /dashboard/performance       → Pre-computed model KPI dashboard
POST /sensors/health-check        → Rule-based sensor range checker
POST /trends                      → Rolling trend + correlation analysis
"""

from __future__ import annotations
from fastapi import APIRouter
import asyncio
import io
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import joblib
import numpy as np
import pandas as pd
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile, status , WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from tensorflow import keras

import csv
from pathlib import Path

router = APIRouter()
# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("boiler_api")

# ──────────────────────────────────────────────────────────────────────────────
# Constants — must match the notebook exactly
# ──────────────────────────────────────────────────────────────────────────────
SENSOR_COLS = [
    "tube_skin_temp",
    "steam_drum_pressure",
    "feed_water_flow",
    "steam_flow",
    "flue_gas_temp",
    "feed_water_ph",
    "dissolved_oxygen",
    "attemp_spray_flow",
    "acoustic_emission",
    "boiler_load",
]
SEQ_LEN      = 30
ROLL_WINDOWS = [10, 30, 60, 120]
ROC_COLS     = [
    "tube_skin_temp", "steam_drum_pressure", "acoustic_emission",
    "feed_water_flow", "dissolved_oxygen",
]
EWM_COLS     = ["tube_skin_temp", "acoustic_emission"]

# All artifacts that must be present before the API accepts requests
REQUIRED_MODEL_KEYS = {
    "scaler", "rf", "iforest", "meta_learner", "normalizer",
    "autoencoder", "lstm", "xgb", "feature_cols",
}

# HuggingFace URLs
HF_FILES: Dict[str, str] = {
    "random_forest":    "https://huggingface.co/ZOROD/Boiler_Tube_Early_leak_detection/resolve/main/random_forest.pkl",
    "isolation_forest": "https://huggingface.co/ZOROD/Boiler_Tube_Early_leak_detection/resolve/main/isolation_forest.pkl",
    "lstm_model":       "https://huggingface.co/ZOROD/Boiler_Tube_Early_leak_detection/resolve/main/lstm_best.keras",
    "autoencoder":      "https://huggingface.co/ZOROD/Boiler_Tube_Early_leak_detection/resolve/main/autoencoder_best.keras",
    "meta_learner":     "https://huggingface.co/ZOROD/Boiler_Tube_Early_leak_detection/resolve/main/meta_learner.pkl",
    "normalizer":       "https://huggingface.co/ZOROD/Boiler_Tube_Early_leak_detection/resolve/main/normalizer.pkl",
    "standard_scaler":  "https://huggingface.co/ZOROD/Boiler_Tube_Early_leak_detection/resolve/main/standard_scaler.pkl",
    "feature_columns":  "https://huggingface.co/ZOROD/Boiler_Tube_Early_leak_detection/resolve/main/feature_cols.json",
    "xgboost_model":    "https://huggingface.co/ZOROD/Boiler_Tube_Early_leak_detection/resolve/main/xgboost_model.json",
}

# Local cache — set MODEL_CACHE_DIR env var to a persistent volume in production
MODEL_CACHE = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/boiler_models"))
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Normal operating ranges  (low, high, unit)
# ──────────────────────────────────────────────────────────────────────────────
NORMAL_RANGES: Dict[str, tuple] = {
    "tube_skin_temp":      (415.0, 430.0, "°C"),
    "steam_drum_pressure": (103.0, 108.0, "bar"),
    "feed_water_flow":     (275.0, 290.0, "t/h"),
    "steam_flow":          (270.0, 285.0, "t/h"),
    "flue_gas_temp":       (375.0, 390.0, "°C"),
    "feed_water_ph":       (9.1,   9.4,   "pH"),
    "dissolved_oxygen":    (6.5,   8.5,   "ppb"),
    "attemp_spray_flow":   (16.0,  21.0,  "t/h"),
    "acoustic_emission":   (10.0,  20.0,  "mV"),
    "boiler_load":         (80.0,  90.0,  "%"),
}

# Physical bounds used to clamp sensor values in simulation / optimisation
PHYSICAL_BOUNDS: Dict[str, tuple] = {
    "tube_skin_temp":      (350.0,  650.0),
    "steam_drum_pressure": (80.0,   130.0),
    "feed_water_flow":     (180.0,  380.0),
    "steam_flow":          (180.0,  380.0),
    "flue_gas_temp":       (300.0,  550.0),
    "feed_water_ph":       (7.0,    11.0),
    "dissolved_oxygen":    (1.0,    50.0),
    "attemp_spray_flow":   (5.0,    40.0),
    "acoustic_emission":   (5.0,    300.0),
    "boiler_load":         (40.0,   100.0),
}

# Diagnosis templates used in RCA
SENSOR_DIAGNOSES: Dict[str, Dict[str, str]] = {
    "tube_skin_temp": {
        "above": "Tube skin temperature elevated — possible scale/deposit buildup, reduced water flow, or heat flux crisis.",
        "below": "Tube skin temperature low — possible overcooling or sensor fault.",
    },
    "steam_drum_pressure": {
        "above": "Steam drum pressure high — possible blocked steam outlet or safety valve issue.",
        "below": "Steam drum pressure low — possible steam leak or feed water deficiency.",
    },
    "acoustic_emission": {
        "above": "High acoustic emission — structural stress, microcracking, or active leak signature detected.",
        "below": "Acoustic emission below normal — sensor may be under-range or signal loss.",
    },
    "dissolved_oxygen": {
        "above": "Elevated dissolved oxygen — corrosion risk high. Check deaerator performance and O2 dosing.",
        "below": "Dissolved oxygen low — check over-treatment.",
    },
    "feed_water_ph": {
        "below": "Feed water pH acidic — corrosion acceleration likely. Check chemical dosing.",
        "above": "Feed water pH too alkaline — scaling risk elevated.",
    },
    "feed_water_flow": {
        "below": "Feed water flow deficit — mass imbalance detected. Risk of tube starvation.",
        "above": "Feed water flow elevated — check for recirculation or bypass issues.",
    },
    "steam_flow": {
        "above": "Steam flow elevated — possible steam demand spike or control valve fault.",
        "below": "Steam flow low — possible partial blockage or load reduction.",
    },
    "flue_gas_temp": {
        "above": "Flue gas temperature elevated — incomplete combustion or heat transfer surface fouling.",
        "below": "Flue gas temperature low — possible excess air or burner issue.",
    },
    "attemp_spray_flow": {
        "above": "Attemperator spray flow high — excessive steam superheating requiring overcooling.",
        "below": "Attemperator spray flow low — monitor steam outlet temperature.",
    },
    "boiler_load": {
        "above": "Boiler load above normal — operating at high capacity increases wear and failure probability.",
        "below": "Boiler load below normal — may indicate load shedding or demand reduction.",
    },
}

RECOMMENDED_ACTIONS: Dict[str, str] = {
    "tube_skin_temp":      "Check feed water flow rate. Inspect tube for scale or blockage. Reduce boiler load by 5–10%.",
    "steam_drum_pressure": "Verify steam outlet valve positions. Check safety valve setpoints. Monitor steam demand.",
    "feed_water_flow":     "Check feed water pump operation. Inspect control valves. Verify drum level control.",
    "steam_flow":          "Review steam demand. Check control valve position. Monitor header pressure.",
    "flue_gas_temp":       "Inspect combustion air ratio. Check soot blower operation. Verify heat transfer surfaces.",
    "feed_water_ph":       "Adjust chemical dosing rates. Sample water chemistry. Check dosing pump operation.",
    "dissolved_oxygen":    "Check deaerator operation and temperature. Verify O2 scavenger dosing. Inspect steam vent.",
    "attemp_spray_flow":   "Review superheater outlet temperature. Check spray control valve. Adjust firing pattern.",
    "acoustic_emission":   "Initiate tube inspection protocol. Check for vibration sources. Monitor structural integrity.",
    "boiler_load":         "Evaluate load reduction feasibility. Monitor all parameters closely at current load.",
}

# ──────────────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────────────
models: Dict[str, Any] = {}

_threshold: float = 0.722          # default from notebook (val-set optimised)
_threshold_lock = asyncio.Lock()   # guards concurrent PUT /threshold calls



# ──────────────────────────────────────────────────────────────────────────────
# Model download + load
# ──────────────────────────────────────────────────────────────────────────────
async def _download_file(key: str, url: str, retries: int = 3) -> Path:
    """
    Download a model artifact from HuggingFace into the local cache.
    Skips the download if the file is already cached.
    Retries up to `retries` times with exponential back-off on failure.
    """
    suffix = Path(url).suffix
    local  = MODEL_CACHE / f"{key}{suffix}"

    if local.exists():
        logger.info("Cache hit: %s", local)
        return local

    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(retries):
        try:
            logger.info("Downloading %s (attempt %d/%d) …", url, attempt + 1, retries)
            async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            local.write_bytes(resp.content)
            logger.info("Saved %s (%d bytes)", local, local.stat().st_size)
            return local
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("Download failed (%s). Retrying in %ds …", exc, wait)
                await asyncio.sleep(wait)

    raise RuntimeError(f"Failed to download '{key}' after {retries} attempts: {last_exc}") from last_exc


async def load_all_models() -> None:
    """
    Download (if needed) and load every model artifact.
    joblib / Keras loads are run in a thread-pool so they don't block
    the event loop during startup.
    """
    global _threshold

    download_tasks = {key: _download_file(key, url) for key, url in HF_FILES.items()}
    paths: Dict[str, Path] = {}
    for key, coro in download_tasks.items():
        paths[key] = await coro

    models["scaler"]       = await asyncio.to_thread(joblib.load, paths["standard_scaler"])
    models["rf"]           = await asyncio.to_thread(joblib.load, paths["random_forest"])
    models["iforest"]      = await asyncio.to_thread(joblib.load, paths["isolation_forest"])
    models["meta_learner"] = await asyncio.to_thread(joblib.load, paths["meta_learner"])
    models["normalizer"]   = await asyncio.to_thread(joblib.load, paths["normalizer"])
    _ae_path   = str(paths["autoencoder"])
    _lstm_path = str(paths["lstm_model"])

    models["autoencoder"] = await asyncio.to_thread(
        lambda: keras.models.load_model(_ae_path, compile=False)
    )
    models["lstm"] = await asyncio.to_thread(
        lambda: keras.models.load_model(_lstm_path, compile=False)
    )

    def _load_xgb() -> xgb.XGBClassifier:
        clf = xgb.XGBClassifier()
        clf.load_model(str(paths["xgboost_model"]))
        return clf

    models["xgb"] = await asyncio.to_thread(_load_xgb)

    with open(paths["feature_columns"], "r") as fh:
        models["feature_cols"] = json.load(fh)

    logger.info(
        "✅ All models loaded | features=%d | normalizer keys=%s",
        len(models["feature_cols"]),
        list(models["normalizer"].keys()),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers — exact mirror of notebook
# ──────────────────────────────────────────────────────────────────────────────
def engineer_features(
    df_raw: pd.DataFrame,
    context_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if context_df is not None:
        missing_ctx = [c for c in SENSOR_COLS if c not in context_df.columns]
        if missing_ctx:
            raise ValueError(f"context_df is missing sensor columns: {missing_ctx}")
        full = pd.concat(
            [context_df[SENSOR_COLS], df_raw[SENSOR_COLS]], ignore_index=True
        )
    else:
        full = df_raw[SENSOR_COLS].copy()

    # ── Collect ALL new columns in a dict, then concat once ──────────────────
    # Avoids 100+ individual frame.insert calls that fragment memory and trigger
    # PerformanceWarning. A single pd.concat rebuilds a contiguous frame.
    new_cols: dict = {}

    # Physics-derived features (4)
    new_cols["mass_balance"]        = full["feed_water_flow"] - full["steam_flow"]
    new_cols["temp_pressure_ratio"] = full["tube_skin_temp"]  / (full["steam_drum_pressure"] + 1e-5)
    new_cols["heat_flux_index"]     = full["tube_skin_temp"]  * full["boiler_load"] / 100.0
    new_cols["o2_corrosion_index"]  = full["dissolved_oxygen"] * (11.0 - full["feed_water_ph"])

    # Rolling mean & std (80 features: 10 sensors × 4 windows × 2)
    for w in ROLL_WINDOWS:
        for col in SENSOR_COLS:
            new_cols[f"{col}_rm{w}"] = full[col].rolling(w, min_periods=1).mean()
            new_cols[f"{col}_rs{w}"] = full[col].rolling(w, min_periods=1).std().fillna(0)

    # Rate-of-change lags (15 features: 5 sensors × 3 lags)
    for col in ROC_COLS:
        new_cols[f"{col}_roc1"]  = full[col].diff(1).fillna(0)
        new_cols[f"{col}_roc5"]  = full[col].diff(5).fillna(0)
        new_cols[f"{col}_roc30"] = full[col].diff(30).fillna(0)

    # Exponentially weighted mean (2 features)
    for col in EWM_COLS:
        new_cols[f"{col}_ewm20"] = full[col].ewm(span=20, adjust=False).mean()

    # Single concat — produces a contiguous, non-fragmented frame
    full = pd.concat([full, pd.DataFrame(new_cols, index=full.index)], axis=1)

    if context_df is not None:
        full = full.iloc[len(context_df):].reset_index(drop=True)

    return full


def _ae_anomaly_score(X: np.ndarray) -> np.ndarray:
    recon = models["autoencoder"].predict(X, batch_size=512, verbose=0)
    return np.mean((X - recon) ** 2, axis=1)


def _norm_ae(s: np.ndarray) -> np.ndarray:
    n = models["normalizer"]
    return np.clip((s - n["ae_min"]) / (n["ae_max"] - n["ae_min"] + 1e-8), 0, 1)


def _norm_if(s: np.ndarray) -> np.ndarray:
    n = models["normalizer"]
    return np.clip((s - n["if_min"]) / (n["if_max"] - n["if_min"] + 1e-8), 0, 1)


def _lstm_batch_predict(X_sc: np.ndarray, batch: int = 512, stride: int = 5) -> np.ndarray:
    """
    Score at every `stride` rows; linearly interpolate in between.
    Returns a probability array of length = len(X_sc).
    First SEQ_LEN entries are padded with the first valid prediction.
    stride=5 gives ~5× speed-up with negligible recall loss (V3 fix #3).
    """
    n_rows  = len(X_sc)
    n_base  = len(SENSOR_COLS)
    out     = np.zeros(n_rows, dtype=np.float32)

    indices = list(range(SEQ_LEN, n_rows, stride))
    if not indices or indices[-1] != n_rows - 1:
        indices.append(n_rows - 1)   # always score the last row

    seqs  = np.stack([X_sc[i - SEQ_LEN:i, :n_base] for i in indices])
    preds = models["lstm"].predict(seqs.astype(np.float32), batch_size=batch, verbose=0).ravel()

    # ── Linear interpolation between scored points ────────────────────────
    for j in range(len(indices) - 1):
        s, e = indices[j], indices[j + 1]
        out[s:e + 1] = np.linspace(preds[j], preds[j + 1], e - s + 1)
    out[indices[-1]] = preds[-1]
    out[:SEQ_LEN]    = preds[0]   # pad head with first valid prediction
    return out


def run_inference(
    df_raw: pd.DataFrame,
    context_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    feature_cols: List[str] = models["feature_cols"]

    df_fe = engineer_features(df_raw, context_df)

    missing_feats = [c for c in feature_cols if c not in df_fe.columns]
    if missing_feats:
        raise ValueError(
            f"Feature engineering did not produce expected columns: {missing_feats}. "
            "Check that all SENSOR_COLS are present in the input."
        )

    X_raw = df_fe[feature_cols].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw)
    X_sc  = models["scaler"].transform(X_raw)
    n     = len(X_sc)

    if n < SEQ_LEN + 1:
        raise ValueError(
            f"Need at least {SEQ_LEN + 1} rows after feature engineering, got {n}."
        )

    ae_sc     = _ae_anomaly_score(X_sc)
    if_sc     = -models["iforest"].decision_function(X_sc)
    rf_proba  = models["rf"].predict_proba(X_sc)         # shape (n, 2)
    xgb_proba = models["xgb"].predict_proba(X_sc)        # shape (n, 2)
    rf_p      = rf_proba[:, 1]                            # class-1 for result DF
    xgb_p     = xgb_proba[:, 1]                          # class-1 for result DF
    lstm_p    = _lstm_batch_predict(X_sc)

    ae_n = _norm_ae(ae_sc)
    if_n = _norm_if(if_sc)

    off = SEQ_LEN
    # 7-feature meta-stack matching trained meta_learner.pkl:
    # [ae_n, if_n, rf_p0, rf_p1, xgb_p0, xgb_p1, lstm_p]
    M = np.column_stack([
        ae_n[off:],
        if_n[off:],
        rf_proba[off:, 0],
        rf_proba[off:, 1],
        xgb_proba[off:, 0],
        xgb_proba[off:, 1],
        lstm_p[off:],
    ])
    # Guard: verify shape matches what meta_learner expects
    expected_feats = int(getattr(models["meta_learner"], "n_features_in_", 7))
    if M.shape[1] != expected_feats:
        raise ValueError(
            f"Meta-learner shape mismatch: expected {expected_feats} features, "
            f"got {M.shape[1]}. Check the M column_stack in run_inference."
        )
    risk  = models["meta_learner"].predict_proba(M)[:, 1]
    alarm = risk >= _threshold

    result = pd.DataFrame({
        "risk_score":       risk,
        "alarm":            alarm,
        "autoencoder":      ae_n[off:],
        "isolation_forest": if_n[off:],
        "random_forest":    rf_p[off:],
        "xgboost":          xgb_p[off:],
        "lstm":             lstm_p[off:],
    })

    if "timestamp" in df_raw.columns:
        result.insert(
            0, "timestamp",
            df_raw["timestamp"].values[off: off + len(result)],
        )

    return result


def _summary_stats(result: pd.DataFrame) -> Dict[str, Any]:
    total  = len(result)
    alarms = int(result["alarm"].sum())
    return {
        "total_windows":  total,
        "alarm_count":    alarms,
        "normal_count":   total - alarms,
        "alarm_rate_pct": round(alarms / total * 100, 2) if total else 0.0,
        "risk_score": {
            "mean": round(float(result["risk_score"].mean()), 4),
            "max":  round(float(result["risk_score"].max()),  4),
            "min":  round(float(result["risk_score"].min()),  4),
        },
        "base_model_means": {
            "autoencoder":      round(float(result["autoencoder"].mean()),      4),
            "isolation_forest": round(float(result["isolation_forest"].mean()), 4),
            "random_forest":    round(float(result["random_forest"].mean()),    4),
            "xgboost":          round(float(result["xgboost"].mean()),          4),
            "lstm":             round(float(result["lstm"].mean()),             4),
        },
        "threshold_used": _threshold,
    }


def _result_to_records(
    result: pd.DataFrame,
    df_raw: Optional[pd.DataFrame] = None,
    off: int = 0,
) -> List[Dict[str, Any]]:
    """
    Convert inference result DataFrame to a list of record dicts.
    If df_raw is supplied, raw sensor values are embedded into each record
    (aligned via the SEQ_LEN offset) so the frontend can display them.
    """
    has_ts = "timestamp" in result.columns
    records = result.to_dict("records")
    out = []
    for i, r in enumerate(records):
        rec: Dict[str, Any] = {
            "timestamp":        str(r["timestamp"]) if has_ts else None,
            "risk_score":       round(float(r["risk_score"]),       4),
            "alarm":            bool(r["alarm"]),
            "autoencoder":      round(float(r["autoencoder"]),      4),
            "isolation_forest": round(float(r["isolation_forest"]), 4),
            "random_forest":    round(float(r["random_forest"]),    4),
            "xgboost":          round(float(r["xgboost"]),          4),
            "lstm":             round(float(r["lstm"]),             4),
        }
        # Embed raw sensor values so the frontend table can display them
        if df_raw is not None:
            raw_idx = off + i
            if raw_idx < len(df_raw):
                for s in SENSOR_COLS:
                    rec[s] = round(float(df_raw.iloc[raw_idx][s]), 4)
        out.append(rec)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# NEW helper utilities
# ──────────────────────────────────────────────────────────────────────────────
def _strip_suffix(feat_name: str) -> str:
    """Map an engineered feature name back to its source sensor."""
    for col in SENSOR_COLS:
        if feat_name.startswith(col):
            return col
    return feat_name   # derived: mass_balance, heat_flux_index, etc.


def _make_synthetic_window(values: dict, n_rows: int = 35) -> pd.DataFrame:
    """Repeat a single-point dict with slight noise to form a synthetic time window."""
    rng = np.random.default_rng(42)
    data: Dict[str, Any] = {}
    for k, v in values.items():
        lo, hi = PHYSICAL_BOUNDS.get(k, (0.0, 1e9))
        data[k] = np.clip(
            rng.normal(v, abs(v) * 0.005, n_rows),
            lo, hi,
        )
    data["timestamp"] = pd.date_range("2026-01-01", periods=n_rows, freq="1min")
    return pd.DataFrame(data)


def _compute_energy_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute all energy/efficiency metrics from a raw sensor DataFrame."""
    ef  = df["steam_flow"] / (df["feed_water_flow"] + 1e-5)
    hr  = df["flue_gas_temp"] * df["boiler_load"] / (df["steam_flow"] + 1e-5)
    bd  = (df["feed_water_flow"] - df["steam_flow"]).abs() / (df["feed_water_flow"] + 1e-5)
    sw  = df["attemp_spray_flow"] / (df["steam_flow"] + 1e-5) * 100.0
    cd  = df["dissolved_oxygen"] * (11.0 - df["feed_water_ph"])

    # Penalties scaled 0–25 each
    hr_pen  = float(np.clip((hr.mean()  - 4.10) / (6.00 - 4.10) * 25, 0, 25))
    bd_pen  = float(np.clip((bd.mean()  - 0.01) / (0.10 - 0.01) * 25, 0, 25))
    sw_pen  = float(np.clip((sw.mean()  - 5.00) / (15.0 - 5.00) * 25, 0, 25))
    cd_pen  = float(np.clip((cd.mean()  - 8.00) / (20.0 - 8.00) * 25, 0, 25))
    score   = max(0.0, 100.0 - (hr_pen + bd_pen + sw_pen + cd_pen))

    return {
        "steam_generation_efficiency_pct": round(float(ef.mean() * 100), 4),
        "heat_rate_index":                 round(float(hr.mean()),        4),
        "blowdown_loss_pct":               round(float(bd.mean() * 100),  4),
        "spray_cooling_waste_pct":         round(float(sw.mean()),        4),
        "corrosion_drag_index":            round(float(cd.mean()),        4),
        "overall_efficiency_score":        round(score,                   4),
    }


def _sensor_status(sensor: str, value: float) -> Dict[str, Any]:
    """Return status dict for a single sensor value."""
    lo, hi, unit = NORMAL_RANGES[sensor]
    center = (lo + hi) / 2.0
    half   = (hi - lo) / 2.0
    dev    = value - center
    dev_pct = round(abs(dev) / center * 100, 2) if center else 0.0
    sigma  = abs(dev) / (half + 1e-9)
    if sigma > 2.0:
        status = "CRITICAL" if value > hi else "CRITICAL"
    elif sigma > 1.0:
        status = "WARNING"
    else:
        status = "NORMAL"
    direction = "above" if value > hi else ("below" if value < lo else "within")
    return {
        "sensor":               sensor,
        "value":                round(value, 4),
        "normal_range":         [lo, hi],
        "status":               status,
        "deviation_pct":        dev_pct,
        "direction":            direction,
        "unit":                 unit,
    }


# ──────────────────────────────────────────────────────────────────────────────
# App lifespan
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up — loading models …")
    await load_all_models()
    logger.info("✅ Models ready.")
    yield
    logger.info("🛑 Shutting down.")


app = FastAPI(
    title="Boiler Tube Early Leak Detection API",
    description=(
        "Stacking-ensemble inference "
        "(Autoencoder + IsolationForest + LSTM + RandomForest + XGBoost → Meta-Learner) "
        "for predicting boiler tube failures. "
        "Model repo: ZOROD/Boiler_Tube_Early_leak_detection on HuggingFace."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas — existing (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    """A single 1-minute sensor snapshot."""
    timestamp:           Optional[str]  = Field(None,  example="2026-02-04T17:20:00")
    tube_skin_temp:      float          = Field(...,   example=420.2)
    steam_drum_pressure: float          = Field(...,   example=103.6)
    feed_water_flow:     float          = Field(...,   example=283.7)
    steam_flow:          float          = Field(...,   example=274.6)
    flue_gas_temp:       float          = Field(...,   example=374.5)
    feed_water_ph:       float          = Field(...,   example=9.29)
    dissolved_oxygen:    float          = Field(...,   example=7.44)
    attemp_spray_flow:   float          = Field(...,   example=18.4)
    acoustic_emission:   float          = Field(...,   example=12.8)
    boiler_load:         float          = Field(...,   example=84.0)


class PredictRequest(BaseModel):
    readings: List[SensorReading] = Field(..., min_items=1)
    context:  Optional[List[SensorReading]] = Field(None)

    @validator("readings")
    def enough_rows(cls, v):
        if len(v) < SEQ_LEN + 1:
            raise ValueError(
                f"At least {SEQ_LEN + 1} readings are required "
                f"(LSTM sequence window = {SEQ_LEN})."
            )
        return v


class WindowResult(BaseModel):
    timestamp:        Optional[str]
    risk_score:       float
    alarm:            bool
    autoencoder:      float
    isolation_forest: float
    random_forest:    float
    xgboost:          float
    lstm:             float


class PredictResponse(BaseModel):
    inference_time_ms: float
    summary:           Dict[str, Any]
    predictions:       List[WindowResult]


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas — NEW (v1.2.0)
# ──────────────────────────────────────────────────────────────────────────────
class RCARequest(BaseModel):
    readings: List[SensorReading]
    context:  Optional[List[SensorReading]] = None
    top_n:    int = Field(5, ge=1, le=10)


class SensorOptimizeRequest(BaseModel):
    readings:             List[SensorReading]
    context:              Optional[List[SensorReading]] = None
    target_risk_score:    float = Field(0.3, ge=0.0, le=1.0)
    optimize_last_n_rows: int   = Field(10,  ge=1,   le=50)


class EnergyAnalysisRequest(BaseModel):
    readings: List[SensorReading]
    context:  Optional[List[SensorReading]] = None


class SimulationRequest(BaseModel):
    tube_skin_temp:      float = Field(..., ge=350,  le=650)
    steam_drum_pressure: float = Field(..., ge=80,   le=130)
    feed_water_flow:     float = Field(..., ge=180,  le=380)
    steam_flow:          float = Field(..., ge=180,  le=380)
    flue_gas_temp:       float = Field(..., ge=300,  le=550)
    feed_water_ph:       float = Field(..., ge=7.0,  le=11.0)
    dissolved_oxygen:    float = Field(..., ge=1.0,  le=50.0)
    attemp_spray_flow:   float = Field(..., ge=5.0,  le=40.0)
    acoustic_emission:   float = Field(..., ge=5.0,  le=300.0)
    boiler_load:         float = Field(..., ge=40.0, le=100.0)


class SHAPRequest(BaseModel):
    readings:    List[SensorReading]
    context:     Optional[List[SensorReading]] = None
    plot_type:   str = Field("bar", pattern="^(bar|waterfall|beeswarm)$")
    max_display: int = Field(15, ge=5, le=30)


class LeadTimeRequest(BaseModel):
    readings: List[SensorReading]
    context:  Optional[List[SensorReading]] = None


class SensorHealthRequest(BaseModel):
    readings: List[SensorReading]


class TrendRequest(BaseModel):
    readings:          List[SensorReading]
    context:           Optional[List[SensorReading]] = None
    sensors:           Optional[List[str]] = None
    resample_minutes:  int = Field(5, ge=1, le=60)



@router.get("/", tags=["meta"])
async def root():
    return {"service": "Boiler Leak Detection API", "version": "1.2.0", "docs": "/docs"}
  
# ──────────────────────────────────────────────────────────────────────────────
# Routes — EXISTING (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/health", tags=["meta"])
async def health():
    loaded  = list(models.keys())
    missing = list(REQUIRED_MODEL_KEYS - set(loaded))
    ready   = len(missing) == 0
    return {
        "status":         "ok" if ready else "loading",
        "models_loaded":  loaded,
        "models_missing": missing,
        "threshold":      _threshold,
    }


@router.get("/models/info", tags=["meta"])
async def model_info():
    _require_models()
    n = models["normalizer"]
    return {
        "feature_count":        len(models["feature_cols"]),
        "feature_columns":      models["feature_cols"],
        "scaler_n_features":    int(models["scaler"].n_features_in_),
        "autoencoder_params":   int(models["autoencoder"].count_params()),
        "lstm_params":          int(models["lstm"].count_params()),
        "rf_n_estimators":      int(models["rf"].n_estimators),
        "xgb_boost_rounds":     int(models["xgb"].get_booster().num_boosted_rounds()),
        "iforest_n_estimators": int(models["iforest"].n_estimators),
        "meta_coef_shape":      list(models["meta_learner"].coef_.shape),
        "normalizer": {
            "ae_min": round(float(n["ae_min"]), 6),
            "ae_max": round(float(n["ae_max"]), 6),
            "if_min": round(float(n["if_min"]), 6),
            "if_max": round(float(n["if_max"]), 6),
        },
        "seq_len":   SEQ_LEN,
        "threshold": _threshold,
    }


@router.get("/threshold", tags=["meta"])
async def get_threshold():
    return {"threshold": _threshold}


@router.put("/threshold", tags=["meta"])
async def set_threshold(body: ThresholdRequest):
    global _threshold
    async with _threshold_lock:
        old        = _threshold
        _threshold = body.threshold
    logger.info("Threshold updated: %.4f → %.4f", old, body.threshold)
    return {"old_threshold": old, "new_threshold": body.threshold}


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(body: PredictRequest):
    _require_models()
    df_raw     = _readings_to_df(body.readings)
    context_df = _readings_to_df(body.context) if body.context else None

    t0 = time.perf_counter()
    try:
        result = run_inference(df_raw, context_df)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    predictions = [WindowResult(**row) for row in _result_to_records(result)]
    return PredictResponse(
        inference_time_ms=round(elapsed_ms, 2),
        summary=_summary_stats(result),
        predictions=predictions,
    )


@router.post("/predict/csv", tags=["inference"])
async def predict_csv(
    file: UploadFile = File(...),
    faults_only: bool = Query(False),
):
    _require_models()
    raw_bytes = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    _validate_df_columns(df_raw)
    if len(df_raw) < SEQ_LEN + 1:
        raise HTTPException(
            status_code=422,
            detail=f"CSV must have at least {SEQ_LEN + 1} rows, found {len(df_raw)}.",
        )
    if "timestamp" in df_raw.columns:
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")

    t0 = time.perf_counter()
    try:
        result = run_inference(df_raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    summary    = _summary_stats(result)
    output     = result[result["alarm"]] if faults_only else result
    predictions = _result_to_records(output)

    return {
        "filename":          file.filename,
        "rows_in":           len(df_raw),
        "windows_scored":    len(result),
        "faults_only":       faults_only,
        "inference_time_ms": round(elapsed_ms, 2),
        "summary":           summary,
        "predictions":       predictions,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Routes — NEW (v1.2.0)
# ──────────────────────────────────────────────────────────────────────────────

# ── 1. ROOT CAUSE ANALYSIS ────────────────────────────────────────────────────
@router.post("/rca", tags=["rca"])
async def root_cause_analysis(body: RCARequest):
    """
    Compute per-sensor blame scores for alarmed windows using XGBoost native SHAP
    + Autoencoder reconstruction error. Returns ranked root causes with diagnosis
    strings, physics insights, and model agreement consensus.
    """
    _require_models()
    df_raw     = _readings_to_df(body.readings)
    context_df = _readings_to_df(body.context) if body.context else None

    try:
        result = run_inference(df_raw, context_df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    alarm_mask   = result["alarm"].values
    alarm_count  = int(alarm_mask.sum())
    total_count  = len(result)

    if alarm_count == 0:
        return {
            "alarm_windows":      0,
            "total_windows":      total_count,
            "rca_performed_on":   "alarmed_windows",
            "root_causes":        [],
            "physics_insights":   {},
            "model_agreement":    {},
            "message":            "No alarm windows detected — RCA not required.",
        }

    feature_cols: List[str] = models["feature_cols"]
    df_fe = engineer_features(df_raw, context_df)
    X_raw = df_fe[feature_cols].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw)
    X_sc  = models["scaler"].transform(X_raw)

    off = SEQ_LEN
    X_aligned = X_sc[off:]
    alarm_X   = X_aligned[alarm_mask]

    # ── XGBoost SHAP (native, no shap library) ─────────────────────────────
    dmatrix   = xgb.DMatrix(alarm_X, feature_names=feature_cols)
    shap_raw  = models["xgb"].get_booster().predict(dmatrix, pred_contribs=True)
    shap_vals = shap_raw[:, :-1]   # drop bias column
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)   # (n_features,)

    # Aggregate SHAP per source sensor
    shap_by_sensor: Dict[str, float] = {s: 0.0 for s in SENSOR_COLS}
    for i, feat in enumerate(feature_cols):
        src = _strip_suffix(feat)
        if src in shap_by_sensor:
            shap_by_sensor[src] += float(mean_abs_shap[i])

    # ── Autoencoder per-feature reconstruction error ────────────────────────
    recon      = models["autoencoder"].predict(alarm_X, batch_size=512, verbose=0)
    ae_feat_err = np.mean((alarm_X - recon) ** 2, axis=0)   # (n_features,)

    ae_by_sensor: Dict[str, float] = {s: 0.0 for s in SENSOR_COLS}
    for i, feat in enumerate(feature_cols):
        src = _strip_suffix(feat)
        if src in ae_by_sensor:
            ae_by_sensor[src] += float(ae_feat_err[i])

    # Normalise both contributions 0–1
    def _norm_dict(d: Dict[str, float]) -> Dict[str, float]:
        mx = max(d.values()) or 1e-8
        return {k: v / mx for k, v in d.items()}

    shap_n = _norm_dict(shap_by_sensor)
    ae_n   = _norm_dict(ae_by_sensor)

    # Blame score: 60% SHAP + 40% AE
    blame: Dict[str, float] = {
        s: round(0.6 * shap_n[s] + 0.4 * ae_n[s], 4)
        for s in SENSOR_COLS
    }
    sorted_sensors = sorted(blame, key=lambda x: blame[x], reverse=True)

    # Mean sensor values in alarmed windows (aligned to result index)
    df_alarm_raw = df_raw.iloc[off:].reset_index(drop=True)
    if len(alarm_mask) <= len(df_alarm_raw):
        alarm_raw_subset = df_alarm_raw[alarm_mask[:len(df_alarm_raw)]]
    else:
        alarm_raw_subset = df_alarm_raw

    # ── Build root-causes list ──────────────────────────────────────────────
    root_causes = []
    for rank, sensor in enumerate(sorted_sensors[: body.top_n], start=1):
        lo, hi, unit = NORMAL_RANGES[sensor]
        mean_val = float(alarm_raw_subset[sensor].mean()) if sensor in alarm_raw_subset.columns else (lo + hi) / 2
        normal_center = (lo + hi) / 2.0
        dev_pct = round(abs(mean_val - normal_center) / normal_center * 100, 2) if normal_center else 0.0
        direction = "above" if mean_val > hi else ("below" if mean_val < lo else "within")

        diag_tmpl  = SENSOR_DIAGNOSES.get(sensor, {})
        diagnosis  = diag_tmpl.get(direction, f"{sensor} value {mean_val:.2f}{unit} is {direction} normal range ({lo}–{hi}{unit}).")
        full_diag  = (
            f"{sensor.replace('_', ' ').title()} is {dev_pct}% {direction} normal range "
            f"({mean_val:.2f}{unit} vs {lo}–{hi}{unit}). {diagnosis}"
        )

        root_causes.append({
            "rank":                rank,
            "sensor":              sensor,
            "blame_score":         blame[sensor],
            "mean_value_in_alarm": round(mean_val, 4),
            "normal_range":        [lo, hi],
            "unit":                unit,
            "deviation_pct":       dev_pct,
            "direction":           direction,
            "diagnosis":           full_diag,
            "recommended_action":  RECOMMENDED_ACTIONS.get(sensor, "Consult plant engineer."),
        })

    # ── Physics insights ────────────────────────────────────────────────────
    if all(c in alarm_raw_subset.columns for c in ["feed_water_flow", "steam_flow", "tube_skin_temp",
                                                     "boiler_load", "dissolved_oxygen", "feed_water_ph"]):
        mb    = float((alarm_raw_subset["feed_water_flow"] - alarm_raw_subset["steam_flow"]).mean())
        hfi   = float((alarm_raw_subset["tube_skin_temp"] * alarm_raw_subset["boiler_load"] / 100.0).mean())
        o2ci  = float((alarm_raw_subset["dissolved_oxygen"] * (11.0 - alarm_raw_subset["feed_water_ph"])).mean())
        physics_insights = {
            "mass_balance":         round(mb,   4),
            "mass_balance_status":  "negative — feed water deficiency detected" if mb < 0 else "positive — normal or feed excess",
            "heat_flux_index":      round(hfi,  4),
            "heat_flux_status":     "elevated — possible heat flux crisis" if hfi > 400 else "normal",
            "o2_corrosion_index":   round(o2ci, 4),
            "o2_corrosion_status":  "high — corrosion risk elevated" if o2ci > 12 else "acceptable",
        }
    else:
        physics_insights = {}

    # ── Model agreement ─────────────────────────────────────────────────────
    alarm_result = result[alarm_mask]
    model_scores = {
        "autoencoder":      round(float(alarm_result["autoencoder"].mean()),      4),
        "isolation_forest": round(float(alarm_result["isolation_forest"].mean()), 4),
        "random_forest":    round(float(alarm_result["random_forest"].mean()),    4),
        "xgboost":          round(float(alarm_result["xgboost"].mean()),          4),
        "lstm":             round(float(alarm_result["lstm"].mean()),             4),
    }
    agreeing = sum(1 for v in model_scores.values() if v >= 0.5)
    consensus_label = f"{'HIGH' if agreeing == 5 else ('MEDIUM' if agreeing >= 3 else 'LOW')} — {agreeing}/5 models agree on anomaly"
    model_scores["consensus"] = consensus_label

    return {
        "alarm_windows":    alarm_count,
        "total_windows":    total_count,
        "rca_performed_on": "alarmed_windows",
        "root_causes":      root_causes,
        "physics_insights": physics_insights,
        "model_agreement":  model_scores,
    }


# ── 2. SENSOR OPTIMISATION ────────────────────────────────────────────────────
@router.post("/optimize/sensor", tags=["optimization"])
async def optimize_sensor(body: SensorOptimizeRequest):
    """
    Compute optimal sensor adjustments to bring ensemble risk score below target.
    Uses XGBoost sensitivity analysis (±5% perturbation) for speed.
    """
    _require_models()
    df_raw     = _readings_to_df(body.readings)
    context_df = _readings_to_df(body.context) if body.context else None

    try:
        result = run_inference(df_raw, context_df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    current_risk = float(result["risk_score"].iloc[-1])
    last_n       = min(body.optimize_last_n_rows, len(df_raw))
    last_rows    = df_raw.iloc[-last_n:]

    # Mean current values of last N rows
    current_vals: Dict[str, float] = {s: float(last_rows[s].mean()) for s in SENSOR_COLS}

    feature_cols: List[str] = models["feature_cols"]

    def _quick_risk(values: Dict[str, float]) -> float:
        """XGBoost-only risk estimate for a single synthetic point."""
        syn = _make_synthetic_window(values, n_rows=SEQ_LEN + 5)
        df_fe = engineer_features(syn)
        X_raw = df_fe[feature_cols].values.astype(np.float32)
        X_raw = np.nan_to_num(X_raw)
        X_sc  = models["scaler"].transform(X_raw)
        xgb_p = models["xgb"].predict_proba(X_sc)[:, 1]
        return float(xgb_p[-1])

    # Compute sensitivities via ±5% perturbation
    sensitivities: Dict[str, float] = {}
    for sensor in SENSOR_COLS:
        lo, hi = PHYSICAL_BOUNDS[sensor]
        v_up   = {**current_vals, sensor: min(current_vals[sensor] * 1.05, hi)}
        v_dn   = {**current_vals, sensor: max(current_vals[sensor] * 0.95, lo)}
        r_up   = _quick_risk(v_up)
        r_dn   = _quick_risk(v_dn)
        delta_r = r_up - r_dn
        delta_s = v_up[sensor] - v_dn[sensor]
        sensitivities[sensor] = abs(delta_r / (delta_s + 1e-12))

    # Build recommendations
    risk_gap     = current_risk - body.target_risk_score
    total_sens   = sum(sensitivities.values()) or 1e-8
    recs         = []
    new_vals     = dict(current_vals)

    for sensor in sorted(SENSOR_COLS, key=lambda s: sensitivities[s], reverse=True):
        lo, hi   = PHYSICAL_BOUNDS[sensor]
        sens     = sensitivities[sensor]
        fraction = sens / total_sens
        ideal_lo, ideal_hi, unit = NORMAL_RANGES[sensor]
        ideal_center = (ideal_lo + ideal_hi) / 2.0

        if current_vals[sensor] > ideal_hi:
            direction = "DECREASE"
            recommended = max(ideal_center, lo)
        elif current_vals[sensor] < ideal_lo:
            direction = "INCREASE"
            recommended = min(ideal_center, hi)
        else:
            direction = "MAINTAIN"
            recommended = current_vals[sensor]

        recommended = float(np.clip(recommended, lo, hi))
        delta        = round(recommended - current_vals[sensor], 4)
        delta_pct    = round(delta / (current_vals[sensor] + 1e-8) * 100, 2)
        priority     = "HIGH" if sens > 0.01 else ("MEDIUM" if sens > 0.001 else "LOW")

        new_vals[sensor] = recommended
        recs.append({
            "sensor":           sensor,
            "current_value":    round(current_vals[sensor], 4),
            "recommended_value": round(recommended, 4),
            "delta":            delta,
            "delta_pct":        delta_pct,
            "sensitivity":      round(sens, 6),
            "priority":         priority,
            "action":           direction,
            "physical_bounds":  [lo, hi],
        })

    expected_risk = round(_quick_risk(new_vals), 4)
    top_levers    = [r["sensor"] for r in sorted(recs, key=lambda x: x["sensitivity"], reverse=True)[:3]]
    feasible      = expected_risk <= body.target_risk_score * 1.1

    advisory = (
        f"Reducing {top_levers[0].replace('_', ' ')} is the single highest-impact action. "
        "All recommendations require operator review before implementation."
    ) if top_levers else "No high-sensitivity levers found. Consult plant engineer."

    return {
        "current_risk_score":                round(current_risk,          4),
        "target_risk_score":                 body.target_risk_score,
        "expected_risk_after_optimization":  expected_risk,
        "optimization_feasible":             feasible,
        "sensor_recommendations":            recs,
        "top_levers":                        top_levers,
        "estimated_lead_time_gain_minutes":  int(risk_gap / 0.01) if risk_gap > 0 else 0,
        "advisory":                          advisory,
    }


# ── 3. ENERGY ANALYSIS ───────────────────────────────────────────────────────
@router.post("/energy/analysis", tags=["energy"])
async def energy_analysis(body: EnergyAnalysisRequest):
    """
    Compute steam generation efficiency, heat rate, blowdown loss, spray waste,
    corrosion drag, and overall boiler efficiency score.
    """
    _require_models()
    df_raw     = _readings_to_df(body.readings)
    context_df = _readings_to_df(body.context) if body.context else None

    try:
        result = run_inference(df_raw, context_df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    metrics = _compute_energy_metrics(df_raw)
    optimal = {
        "steam_generation_efficiency_pct": 98.0,
        "heat_rate_index":                 4.10,
        "blowdown_loss_pct":               1.5,
        "spray_cooling_waste_pct":         5.0,
        "corrosion_drag_index":            8.0,
        "overall_efficiency_score":        90.0,
    }
    gap_pct  = round(optimal["overall_efficiency_score"] - metrics["overall_efficiency_score"], 2)
    save_est = round(gap_pct * 0.47, 2)   # empirical mapping from efficiency gap to energy saving

    # Identify top wastes
    waste_keys = ["corrosion_drag_index", "blowdown_loss_pct", "spray_cooling_waste_pct", "heat_rate_index"]
    wastes = []
    for k in waste_keys:
        cur = metrics[k]
        tgt = optimal[k]
        if cur > tgt * 1.05:
            impact = "HIGH" if (cur - tgt) / (tgt + 1e-8) > 0.3 else "MEDIUM"
            if k == "corrosion_drag_index":
                rec = "Increase feed_water_ph to 9.2–9.4 and reduce dissolved_oxygen below 7 ppb. Consider dosing adjustments."
            elif k == "blowdown_loss_pct":
                rec = "Optimise continuous blowdown rate. Verify drum level setpoint."
            elif k == "spray_cooling_waste_pct":
                rec = "Review superheater outlet temperature control. Reduce attemperator spray if steam temperature permits."
            else:
                rec = "Inspect heat transfer surfaces for fouling. Optimise combustion air ratio."
            wastes.append({"parameter": k, "current": round(cur, 4), "target": tgt, "impact": impact, "recommendation": rec})

    # Risk–energy correlation
    normal_eff   = _compute_energy_metrics(df_raw[~result["alarm"].reindex(df_raw.index, fill_value=False)])
    alarm_eff    = _compute_energy_metrics(df_raw[result["alarm"].reindex(df_raw.index, fill_value=False)]) if result["alarm"].any() else metrics
    corr_note    = (
        f"Current high-risk windows have "
        f"{abs(round(normal_eff['overall_efficiency_score'] - alarm_eff['overall_efficiency_score'], 1))}% "
        "lower efficiency than normal windows — failure events waste energy."
    )

    return {
        "window_rows":             len(df_raw),
        "energy_metrics":          metrics,
        "optimal_targets":         optimal,
        "efficiency_gap_pct":      gap_pct,
        "estimated_energy_saving_pct": save_est,
        "top_energy_wastes":       wastes,
        "risk_energy_correlation": corr_note,
    }


# ── 4. SIMULATION (WHAT-IF) ───────────────────────────────────────────────────
@router.post("/simulate", tags=["simulation"])
async def simulate(body: SimulationRequest):
    _require_models()
    input_values = body.model_dump()

    # Run heavy inference off the event loop so WebSocket stays alive
    def _run():
        syn_df = _make_synthetic_window(input_values, n_rows=SEQ_LEN + 5)
        return run_inference(syn_df), syn_df

    try:
        result, syn_df = await asyncio.to_thread(_run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    last = result.iloc[-1]

    # Debug log — check Render logs after submitting to see actual values
    logger.info(
        "simulate → ae=%.4f if=%.4f rf=%.4f xgb=%.4f lstm=%.4f risk=%.4f alarm=%s",
        float(last["autoencoder"]), float(last["isolation_forest"]),
        float(last["random_forest"]), float(last["xgboost"]),
        float(last["lstm"]), float(last["risk_score"]), bool(last["alarm"]),
    )

    prediction = {
        "risk_score":       round(float(last["risk_score"]),       4),
        "alarm":            bool(last["alarm"]),
        "autoencoder":      round(float(last["autoencoder"]),      4),
        "isolation_forest": round(float(last["isolation_forest"]), 4),
        "random_forest":    round(float(last["random_forest"]),    4),
        "xgboost":          round(float(last["xgboost"]),          4),
        "lstm":             round(float(last["lstm"]),             4),
    }

    sensor_status = [_sensor_status(s, input_values[s]) for s in SENSOR_COLS]
    energy = _compute_energy_metrics(syn_df)
    energy_snapshot = {
        "overall_efficiency_score":                 energy["overall_efficiency_score"],
        "steam_generation_efficiency_pct":          energy["steam_generation_efficiency_pct"],
        "estimated_energy_saving_if_optimized_pct": round(max(0, 90 - energy["overall_efficiency_score"]) * 0.47, 2),
    }

    to_normalize = []
    for ss in sensor_status:
        if ss["status"] in ("CRITICAL", "WARNING") and ss["direction"] != "within":
            lo, hi, _ = NORMAL_RANGES[ss["sensor"]]
            ideal = (lo + hi) / 2.0
            to_normalize.append({
                "sensor":    ss["sensor"],
                "change_to": round(ideal, 4),
                "change_by": round(ideal - ss["value"], 4),
            })

    return {
        "simulation_mode":   True,
        "input_values":      {k: round(v, 4) for k, v in input_values.items()},
        "prediction":        prediction,
        "sensor_status":     sensor_status,
        "energy_snapshot":   energy_snapshot,
        "to_normalize_risk": to_normalize,
    }
# ── 5. CSV BULK WITH OPTIMISATION ─────────────────────────────────────────────
@router.post("/predict/csv/optimized", tags=["inference", "optimization"])
async def predict_csv_optimized(
    file: UploadFile = File(...),
    include_energy: bool = Query(True),
    include_rca:    bool = Query(True),
):
    """
    CSV upload → full inference + per-alarm optimised values + optional energy + RCA summary.
    Sensor values are embedded in every prediction record for frontend display.
    Optimisation is batched (single XGBoost predict_proba call) for speed.
    All heavy CPU work runs in a thread pool via asyncio.to_thread so the event
    loop is never blocked and the server stays responsive.
    """
    _require_models()
    raw_bytes = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    _validate_df_columns(df_raw)
    if len(df_raw) < SEQ_LEN + 1:
        raise HTTPException(status_code=422, detail=f"CSV must have ≥ {SEQ_LEN + 1} rows.")
    if "timestamp" in df_raw.columns:
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], errors="coerce")

    # Coerce all sensor columns to float and fill NaN with column median
    for s in SENSOR_COLS:
        df_raw[s] = pd.to_numeric(df_raw[s], errors="coerce")
        if df_raw[s].isna().any():
            df_raw[s] = df_raw[s].fillna(df_raw[s].median())

    filename = file.filename

    # ── Run ALL heavy CPU work off the event loop ──────────────────────────────
    try:
        out = await asyncio.to_thread(
            _run_bulk_inference, df_raw, filename, include_energy, include_rca
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return out


# Maximum alarm rows to run optimisation on (keeps large-file response fast)
MAX_OPT_ALARMS = 50
# Maximum input rows — files larger than this are evenly sampled
MAX_ROWS_BULK  = 20_000


def _run_bulk_inference(
    df_raw: pd.DataFrame,
    filename: str,
    include_energy: bool,
    include_rca: bool,
) -> Dict[str, Any]:
    """
    Sync worker: full inference + optimisation + RCA.
    Runs in a thread pool via asyncio.to_thread so the event loop stays free.
    Large files (>MAX_ROWS_BULK rows) are evenly sampled before inference.
    Optimisation is capped at the first MAX_OPT_ALARMS alarm rows for speed.
    """
    t0 = time.perf_counter()
    original_rows = len(df_raw)

    # ── Cap very large files to avoid OOM / multi-minute hangs ────────────────
    if original_rows > MAX_ROWS_BULK:
        step = original_rows // MAX_ROWS_BULK
        df_raw = df_raw.iloc[::step].reset_index(drop=True)
        logger.warning(
            "Large file (%d rows) sampled to %d rows (every %d-th row)",
            original_rows, len(df_raw), step,
        )

    try:
        result = run_inference(df_raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    feature_cols: List[str] = models["feature_cols"]
    off = SEQ_LEN

    # ── Embed raw sensor values into every record ──────────────────────────────
    records = _result_to_records(result, df_raw=df_raw, off=off)

    # ── Batch optimisation — capped at MAX_OPT_ALARMS for large files ──────────
    all_alarm_indices = [i for i, r in enumerate(records) if r["alarm"]]
    alarm_indices = all_alarm_indices[:MAX_OPT_ALARMS]  # cap for speed

    if alarm_indices:
        # Build opt_vals for each alarm row
        all_opt_vals: List[Dict[str, float]] = []
        for idx in alarm_indices:
            raw_idx = off + idx
            if raw_idx >= len(df_raw):
                all_opt_vals.append({})
                continue
            row_vals = {s: float(df_raw.iloc[raw_idx][s]) for s in SENSOR_COLS}
            opt_vals: Dict[str, float] = {}
            for sensor in SENSOR_COLS:
                lo_r, hi_r, _ = NORMAL_RANGES[sensor]
                lo_p, hi_p    = PHYSICAL_BOUNDS[sensor]
                if row_vals[sensor] > hi_r or row_vals[sensor] < lo_r:
                    opt_vals[sensor] = float(np.clip((lo_r + hi_r) / 2.0, lo_p, hi_p))
                else:
                    opt_vals[sensor] = row_vals[sensor]
            all_opt_vals.append(opt_vals)

        # Build synthetic windows for all alarm rows and stack into one array
        X_opt_list: List[np.ndarray] = []
        valid_alarm_mask: List[bool] = []
        for opt_vals in all_opt_vals:
            if not opt_vals:
                valid_alarm_mask.append(False)
                X_opt_list.append(None)
                continue
            try:
                syn = _make_synthetic_window(opt_vals, n_rows=SEQ_LEN + 5)
                df_fe = engineer_features(syn)
                X_raw_row = df_fe[feature_cols].values.astype(np.float32)
                X_raw_row = np.nan_to_num(X_raw_row)
                X_sc_row  = models["scaler"].transform(X_raw_row)
                X_opt_list.append(X_sc_row[-1:])   # only need last row for XGB
                valid_alarm_mask.append(True)
            except Exception:
                valid_alarm_mask.append(False)
                X_opt_list.append(None)

        # Batch XGBoost predict_proba on all valid last rows
        valid_rows = [x for x in X_opt_list if x is not None]
        if valid_rows:
            X_batch = np.vstack(valid_rows)
            batch_risks = models["xgb"].predict_proba(X_batch)[:, 1]
        else:
            batch_risks = np.array([])

        risk_ptr = 0
        for i, idx in enumerate(alarm_indices):
            raw_idx = off + idx
            if raw_idx >= len(df_raw) or not valid_alarm_mask[i]:
                continue

            row_vals = {s: records[idx].get(s, float(df_raw.iloc[raw_idx][s])) for s in SENSOR_COLS}
            opt_vals = all_opt_vals[i]
            expected_risk = round(float(batch_risks[risk_ptr]), 4)
            risk_ptr += 1

            # Only include sensors that actually changed
            changed = {
                s: round(opt_vals[s], 4)
                for s in SENSOR_COLS
                if abs(opt_vals[s] - row_vals[s]) > 0.01
            }
            changed["expected_risk_after"] = expected_risk
            records[idx]["optimized_values"] = changed

            # Primary cause: store the sensor key directly (not a formatted string)
            out_of_range = [
                s for s in SENSOR_COLS
                if row_vals[s] > NORMAL_RANGES[s][1] or row_vals[s] < NORMAL_RANGES[s][0]
            ]
            if out_of_range:
                oor_s = out_of_range[0]
                center = (NORMAL_RANGES[oor_s][0] + NORMAL_RANGES[oor_s][1]) / 2.0
                dev_pct = round(abs(row_vals[oor_s] - center) / (center + 1e-8) * 100, 1)
                records[idx]["primary_cause"] = f"{oor_s.replace('_', ' ')} elevated by {dev_pct}%"
                records[idx]["primary_cause_sensor"] = oor_s   # raw key for aggregation
            else:
                records[idx]["primary_cause"] = "no single dominant cause"
                records[idx]["primary_cause_sensor"] = None

    elapsed_ms = (time.perf_counter() - t0) * 1000
    summary    = _summary_stats(result)

    out: Dict[str, Any] = {
        "filename":            filename,
        "rows_in":             original_rows,
        "rows_used":           len(df_raw),
        "sampled":             original_rows > MAX_ROWS_BULK,
        "windows_scored":      len(result),
        "alarm_count":         summary["alarm_count"],
        "alarm_rate_pct":      summary["alarm_rate_pct"],
        "opt_alarms_computed": len(alarm_indices),
        "total_alarms":        len(all_alarm_indices),
        "inference_time_ms":   round(elapsed_ms, 2),
        "summary":             summary,
        "predictions":         records,
    }

    if include_energy:
        out["energy_analysis"] = _compute_energy_metrics(df_raw)

    if include_rca and summary["alarm_count"] > 0:
        cause_counter: Dict[str, int] = {}
        for r in records:
            # Use the stable sensor-key field for aggregation
            sensor_key = r.get("primary_cause_sensor")
            if sensor_key:
                cause_counter[sensor_key] = cause_counter.get(sensor_key, 0) + 1
        top_cause = max(cause_counter, key=cause_counter.get) if cause_counter else "unknown"
        out["rca_summary"] = {
            "most_common_root_cause": top_cause,
            "cause_frequency":        cause_counter,
        }

    return out


# ── 6. SHAP EXPLAINABILITY ────────────────────────────────────────────────────
@router.post("/explain/shap", tags=["explainability"])
async def explain_shap(body: SHAPRequest):
    """
    Return XGBoost native SHAP values for feature-level and sensor-level importance.
    Supports bar, waterfall, and beeswarm plot types.
    """
    _require_models()
    df_raw     = _readings_to_df(body.readings)
    context_df = _readings_to_df(body.context) if body.context else None

    feature_cols: List[str] = models["feature_cols"]
    try:
        df_fe = engineer_features(df_raw, context_df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    X_raw = df_fe[feature_cols].values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw)
    X_sc  = models["scaler"].transform(X_raw)
    if len(X_sc) < SEQ_LEN + 1:
        raise HTTPException(status_code=422, detail=f"Need ≥ {SEQ_LEN + 1} rows after feature engineering.")

    dmatrix  = xgb.DMatrix(X_sc, feature_names=feature_cols)
    shap_raw = models["xgb"].get_booster().predict(dmatrix, pred_contribs=True)
    shap_vals = shap_raw[:, :-1]   # drop bias

    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_feat_idx = np.argsort(mean_abs)[::-1][: body.max_display]
    top_features = [
        {
            "feature":       feature_cols[i],
            "mean_abs_shap": round(float(mean_abs[i]), 6),
            "rank":          rank + 1,
        }
        for rank, i in enumerate(top_feat_idx)
    ]

    # Sensor-level aggregation
    sensor_shap: Dict[str, float] = {s: 0.0 for s in SENSOR_COLS}
    for i, feat in enumerate(feature_cols):
        src = _strip_suffix(feat)
        if src in sensor_shap:
            sensor_shap[src] += float(mean_abs[i])
    sorted_sensors = sorted(sensor_shap, key=lambda x: sensor_shap[x], reverse=True)
    ranked_sensors = [
        {"sensor": s, "total_shap": round(sensor_shap[s], 6), "rank": r + 1}
        for r, s in enumerate(sorted_sensors)
    ]

    # Waterfall / beeswarm: use highest-risk row
    xgb_proba  = models["xgb"].predict_proba(X_sc)[:, 1]
    best_row   = int(np.argmax(xgb_proba))
    bias_val   = float(shap_raw[best_row, -1])
    top_contrib_idx = np.argsort(np.abs(shap_vals[best_row]))[::-1][: body.max_display]
    contributions   = [
        {
            "feature":       feature_cols[i],
            "shap_value":    round(float(shap_vals[best_row, i]), 6),
            "feature_value": round(float(X_sc[best_row, i]),      6),
        }
        for i in top_contrib_idx
    ]

    # Meta-learner coefficients (normalised) — 5-feature stack
    coef = models["meta_learner"].coef_[0]
    coef_norm = coef / (np.sum(np.abs(coef)) + 1e-8)
    META_NAMES = [
        "autoencoder", "isolation_forest", "random_forest", "xgboost", "lstm",
    ]
    meta_weights = {k: round(float(coef_norm[i]), 4) for i, k in enumerate(META_NAMES)}

    return {
        "plot_type":         body.plot_type,
        "n_rows_explained":  len(X_sc),
        "feature_level": {"top_features": top_features},
        "sensor_level":  {"ranked_sensors": ranked_sensors},
        "waterfall_data": {
            "base_value":   round(bias_val, 6),
            "row_index":    best_row,
            "risk_score":   round(float(xgb_proba[best_row]), 4),
            "contributions": contributions,
        },
        "meta_learner_weights": meta_weights,
    }


# ── 7. LEAD-TIME PREDICTION ───────────────────────────────────────────────────
@router.post("/lead-time", tags=["monitoring"])
async def lead_time(body: LeadTimeRequest):
    """
    Estimate time remaining before boiler tube failure based on risk score trend.
    Extrapolates linearly from the last 10 alarmed windows.
    """
    _require_models()
    df_raw     = _readings_to_df(body.readings)
    context_df = _readings_to_df(body.context) if body.context else None

    try:
        result = run_inference(df_raw, context_df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    total      = len(result)
    alarm_mask = result["alarm"].values
    alarm_idx  = [i for i, v in enumerate(alarm_mask) if v]
    alarm_count = len(alarm_idx)

    has_ts = "timestamp" in result.columns
    first_alarm_ts = None
    if alarm_idx:
        first_idx = alarm_idx[0]
        if has_ts:
            first_alarm_ts = str(result["timestamp"].iloc[first_idx])

    current_risk = float(result["risk_score"].iloc[-1])

    # Trend estimation
    slope = 0.0
    if len(alarm_idx) >= 2:
        recent_idx    = alarm_idx[-min(10, len(alarm_idx)):]
        risk_vals     = result["risk_score"].values[recent_idx]
        x             = np.arange(len(risk_vals), dtype=float)
        slope, _      = np.polyfit(x, risk_vals, 1)

    time_to_critical = None
    failure_ts       = None
    if slope > 1e-6:
        time_to_critical = int((1.0 - current_risk) / slope)
        if has_ts and first_alarm_ts:
            try:
                last_ts = pd.Timestamp(result["timestamp"].iloc[-1])
                failure_ts = str(last_ts + pd.Timedelta(minutes=time_to_critical))
            except Exception:
                failure_ts = None

    # Urgency classification
    if current_risk > 0.9:
        urgency = "CRITICAL"
        reason  = f"Risk score {current_risk:.3f} exceeds 0.9 — immediate action required."
    elif current_risk >= 0.7 and (time_to_critical or 999) < 60:
        urgency = "HIGH"
        reason  = f"Risk rising at +{slope:.4f}/min. Estimated critical threshold in ~{time_to_critical} min."
    elif current_risk >= 0.5:
        urgency = "MEDIUM"
        reason  = "Risk elevated — monitor closely and prepare for intervention."
    else:
        urgency = "LOW"
        reason  = "Risk within acceptable bounds."

    actions = {
        "CRITICAL": ["Initiate emergency shutdown procedure", "Alert operations supervisor immediately", "Initiate tube inspection protocol"],
        "HIGH":     ["Reduce boiler load by 10% immediately", "Increase feed water flow by 5–10 t/h", "Initiate tube inspection protocol"],
        "MEDIUM":   ["Increase monitoring frequency", "Verify sensor readings", "Prepare contingency plan"],
        "LOW":      ["Continue routine monitoring", "Log current readings for trend analysis"],
    }

    # Risk timeline (sample every 5th row for brevity)
    timeline = []
    for i in range(0, total, max(1, total // 50)):
        row = result.iloc[i]
        timeline.append({
            "index":      i,
            "timestamp":  str(row["timestamp"]) if has_ts else None,
            "risk_score": round(float(row["risk_score"]), 4),
            "alarm":      bool(row["alarm"]),
        })

    return {
        "windows_scored":             total,
        "alarm_windows":              alarm_count,
        "first_alarm_at_index":       alarm_idx[0] if alarm_idx else None,
        "first_alarm_timestamp":      first_alarm_ts,
        "current_risk_score":         round(current_risk, 4),
        "risk_trend_slope":           round(float(slope), 6),
        "estimated_time_to_critical_min": time_to_critical,
        "estimated_failure_timestamp":    failure_ts,
        "urgency":                    urgency,
        "urgency_reason":             reason,
        "lead_time_available_min":    time_to_critical,
        "recommended_actions":        actions[urgency],
        "risk_timeline":              timeline,
    }


# ── 8. DASHBOARD PERFORMANCE ──────────────────────────────────────────────────
@router.get("/dashboard/performance", tags=["dashboard"])
async def dashboard_performance():
    """
    Return pre-computed model performance KPIs + live model metadata for the
    frontend performance dashboard. No inference required.
    """
    # Fallback defaults for 5-feature stack
    meta_weights = {
        "autoencoder": 0.15, "isolation_forest": 0.12,
        "random_forest": 0.20, "xgboost": 0.25, "lstm": 0.28,
    }
    if "meta_learner" in models:
        try:
            coef      = models["meta_learner"].coef_[0]
            coef_norm = coef / (np.sum(np.abs(coef)) + 1e-8)
            META_NAMES = [
                "autoencoder", "isolation_forest", "random_forest", "xgboost", "lstm",
            ]
            meta_weights = {k: round(float(coef_norm[i]), 4) for i, k in enumerate(META_NAMES)}
        except Exception:
            pass

    model_info_live = {
        "autoencoder_params": int(models["autoencoder"].count_params()) if "autoencoder" in models else 0,
        "lstm_params":         int(models["lstm"].count_params())        if "lstm"        in models else 0,
        "rf_n_estimators":     int(models["rf"].n_estimators)            if "rf"          in models else 0,
        "xgb_boost_rounds":    int(models["xgb"].get_booster().num_boosted_rounds()) if "xgb" in models else 0,
        "feature_count":       len(models["feature_cols"])               if "feature_cols" in models else 0,
        "seq_len":             SEQ_LEN,
    }

    return {
        "model_version": "V3",
        "threshold":     _threshold,
        "ensemble_metrics": {
            "roc_auc":       0.987,
            "avg_precision": 0.961,
            "f1_score":      0.879,
            "recall":        0.887,
            "precision":     0.871,
            "specificity":   0.944,
            "brier_score":   0.048,
        },
        "base_model_metrics": [
            {"model": "XGBoost",         "roc_auc": 0.971, "avg_precision": 0.928},
            {"model": "RandomForest",    "roc_auc": 0.963, "avg_precision": 0.914},
            {"model": "LSTM",            "roc_auc": 0.944, "avg_precision": 0.891},
            {"model": "Autoencoder",     "roc_auc": 0.912, "avg_precision": 0.876},
            {"model": "IsolationForest", "roc_auc": 0.889, "avg_precision": 0.851},
        ],
        "meta_learner_coefficients": meta_weights,
        "lead_time_benchmark": {
            "detection_rate_pct":   94.0,
            "median_lead_time_min": 210,
            "mean_lead_time_min":   225,
            "min_lead_time_min":    45,
            "max_lead_time_min":    358,
        },
        "model_info": model_info_live,
    }


# ── 9. SENSOR HEALTH CHECK ────────────────────────────────────────────────────
@router.post("/sensors/health-check", tags=["monitoring"])
async def sensor_health_check(body: SensorHealthRequest):
    """
    Rule-based sensor health card — no ML inference required.
    Checks each sensor against normal operating ranges and returns a health score.
    """
    df = pd.DataFrame([r.model_dump() for r in body.readings])
    sensor_cards = []
    for sensor in SENSOR_COLS:
        lo, hi, unit = NORMAL_RANGES[sensor]
        center       = (lo + hi) / 2.0
        half_range   = (hi - lo) / 2.0
        mean_val     = float(df[sensor].mean())
        dev          = mean_val - center
        sigma        = abs(dev) / (half_range + 1e-9)
        health_score = round(max(0.0, 100.0 - sigma * 33.3), 2)
        if sigma > 2.0:
            card_status = "CRITICAL"
        elif sigma > 1.0:
            card_status = "WARNING"
        else:
            card_status = "OK"
        direction = "above" if mean_val > hi else ("below" if mean_val < lo else "within")
        sensor_cards.append({
            "sensor":               sensor,
            "mean_value":           round(mean_val, 4),
            "normal_center":        round(center,   4),
            "normal_range":         [lo, hi],
            "health_score":         health_score,
            "status":               card_status,
            "deviation_from_center": round(abs(dev), 4),
            "deviation_pct":        round(abs(dev) / center * 100, 2) if center else 0.0,
            "unit":                 unit,
            "direction":            direction,
        })

    overall_score  = round(float(np.mean([c["health_score"] for c in sensor_cards])), 2)
    overall_status = "CRITICAL" if overall_score < 50 else ("WARNING" if overall_score < 75 else "HEALTHY")
    critical_sensors = [c["sensor"] for c in sensor_cards if c["status"] == "CRITICAL"]
    warning_sensors  = [c["sensor"] for c in sensor_cards if c["status"] == "WARNING"]
    healthy_sensors  = [c["sensor"] for c in sensor_cards if c["status"] == "OK"]

    return {
        "rows_checked":     len(body.readings),
        "overall_health_score": overall_score,
        "overall_status":   overall_status,
        "sensor_health":    sensor_cards,
        "critical_sensors": critical_sensors,
        "warning_sensors":  warning_sensors,
        "healthy_sensors":  healthy_sensors,
    }


# ── 10. TREND ANALYSIS ────────────────────────────────────────────────────────
@router.post("/trends", tags=["monitoring"])
async def trend_analysis(body: TrendRequest):
    """
    Compute rolling trend statistics, linear slope, min/max, and risk correlation
    for each sensor. Detect fast-moving anomalies (>2% per minute slope).
    """
    _require_models()
    if len(body.readings) < 60:
        logger.warning("/trends received only %d rows; 60+ recommended.", len(body.readings))

    df_raw     = _readings_to_df(body.readings)
    context_df = _readings_to_df(body.context) if body.context else None

    try:
        result = run_inference(df_raw, context_df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    sensors_to_use = body.sensors if body.sensors else SENSOR_COLS
    sensors_to_use = [s for s in sensors_to_use if s in SENSOR_COLS]

    w = body.resample_minutes
    risk_vals   = result["risk_score"].values
    risk_slope, _ = np.polyfit(np.arange(len(risk_vals)), risk_vals, 1)
    has_ts      = "timestamp" in result.columns

    risk_trend = {
        "values":          [round(float(v), 4) for v in risk_vals[::max(1, w)]],
        "timestamps":      [str(result["timestamp"].iloc[i]) for i in range(0, len(result), max(1, w))] if has_ts else [],
        "slope":           round(float(risk_slope), 6),
        "trend_direction": "rising" if risk_slope > 1e-4 else ("falling" if risk_slope < -1e-4 else "stable"),
    }

    # Align df_raw rows to result (skip SEQ_LEN offset rows)
    off        = SEQ_LEN
    df_aligned = df_raw.iloc[off: off + len(result)].reset_index(drop=True)

    sensor_trends: Dict[str, Any] = {}
    corr_map:      Dict[str, float] = {}
    fast_moving:   List[str] = []

    for sensor in sensors_to_use:
        if sensor not in df_aligned.columns:
            continue
        vals     = df_aligned[sensor].values.astype(float)
        rm       = pd.Series(vals).rolling(w, min_periods=1).mean().values
        rs       = pd.Series(vals).rolling(w, min_periods=1).std().fillna(0).values
        slope_s, _ = np.polyfit(np.arange(len(vals)), vals, 1)
        corr     = float(np.corrcoef(vals, risk_vals[:len(vals)])[0, 1]) if len(vals) > 2 else 0.0
        direction = "rising" if slope_s > 0.005 else ("falling" if slope_s < -0.005 else "stable")
        is_fast  = abs(slope_s / (np.mean(vals) + 1e-8)) > 0.02

        sensor_trends[sensor] = {
            "values":               [round(float(v), 4) for v in vals[::max(1, w)]],
            "rolling_mean":         [round(float(v), 4) for v in rm[::max(1, w)]],
            "rolling_std":          [round(float(v), 4) for v in rs[::max(1, w)]],
            "slope_per_min":        round(float(slope_s), 6),
            "trend_direction":      direction,
            "min":                  round(float(vals.min()), 4),
            "max":                  round(float(vals.max()), 4),
            "correlation_with_risk": round(corr, 4),
            "fast_moving":          is_fast,
        }
        corr_map[sensor] = round(corr, 4)
        if is_fast:
            fast_moving.append(sensor)

    return {
        "windows_scored":           len(result),
        "resample_minutes":         w,
        "risk_trend":               risk_trend,
        "sensor_trends":            sensor_trends,
        "fast_moving_sensors":      fast_moving,
        "sensor_risk_correlations": corr_map,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _require_models() -> None:
    missing = REQUIRED_MODEL_KEYS - set(models.keys())
    if missing:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Models still loading. Missing: {sorted(missing)}. Retry in a moment.",
        )


def _readings_to_df(readings: List[SensorReading]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in readings])
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _validate_df_columns(df: pd.DataFrame) -> None:
    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required sensor columns: {missing}",
        )



# ──────────────────────────────────────────────────────────────────────────────
# Enriched predicition calls on singular fields with RCA rich and optimization result.
# ──────────────────────────────────────────────────────────────────────────────

# Add these models near the other Pydantic schemas (e.g., after PredictResponse)
class SensorStatusItem(BaseModel):
    sensor: str
    value: float
    normal_range: List[float]
    unit: str
    status: str   # NORMAL, WARNING, CRITICAL
    direction: str  # within, above, below
    deviation_pct: Optional[float] = None

class EnrichedPrediction(BaseModel):
    timestamp: Optional[str]
    risk_score: float
    alarm: bool
    autoencoder: float
    isolation_forest: float
    random_forest: float
    xgboost: float
    lstm: float
    sensor_status: Optional[List[SensorStatusItem]] = None
    optimized_values: Optional[Dict[str, float]] = None
    primary_cause: Optional[str] = None
    primary_cause_sensor: Optional[str] = None

class EnrichedPredictResponse(BaseModel):
    inference_time_ms: float
    summary: Dict[str, Any]
    predictions: List[EnrichedPrediction] 
    
class EnrichedFullResponse(EnrichedPredictResponse):
    rca: Optional[Dict[str, Any]] = None
    optimization: Optional[Dict[str, Any]] = None
    energy: Optional[Dict[str, Any]] = None

async def compute_full_analysis(
    readings: List[SensorReading],
    context: Optional[List[SensorReading]] = None
) -> Dict[str, Any]:
    """Compute enriched predictions + RCA + optimization + energy from one inference pass."""
    t0 = time.perf_counter()  
    df_raw = _readings_to_df(readings)
    context_df = _readings_to_df(context) if context else None
    
    

    # Run inference once
    result = await asyncio.to_thread(run_inference, df_raw, context_df)

    off = SEQ_LEN
    df_aligned = df_raw.iloc[off: off + len(result)].reset_index(drop=True) if len(df_raw) > off else df_raw

    # Build enriched predictions (same logic as /predict/enriched)
    enriched_predictions = []
    for i, row in result.iterrows():
        rec = {
            "timestamp": str(row["timestamp"]) if "timestamp" in result.columns else None,
            "risk_score": round(float(row["risk_score"]), 4),
            "alarm": bool(row["alarm"]),
            "autoencoder": round(float(row["autoencoder"]), 4),
            "isolation_forest": round(float(row["isolation_forest"]), 4),
            "random_forest": round(float(row["random_forest"]), 4),
            "xgboost": round(float(row["xgboost"]), 4),
            "lstm": round(float(row["lstm"]), 4),
        }
        raw_row = df_aligned.iloc[i] if i < len(df_aligned) else None
        if raw_row is not None:
            sensor_status = []
            for sensor in SENSOR_COLS:
                val = float(raw_row[sensor])
                lo, hi, unit = NORMAL_RANGES[sensor]
                if val > hi:
                    status = "CRITICAL" if val > hi * 1.1 else "WARNING"
                    direction = "above"
                    dev_pct = (val - hi) / hi * 100
                elif val < lo:
                    status = "CRITICAL" if val < lo * 0.9 else "WARNING"
                    direction = "below"
                    dev_pct = (lo - val) / lo * 100
                else:
                    status = "NORMAL"
                    direction = "within"
                    dev_pct = 0.0
                sensor_status.append({
                    "sensor": sensor,
                    "value": round(val, 4),
                    "normal_range": [lo, hi],
                    "unit": unit,
                    "status": status,
                    "direction": direction,
                    "deviation_pct": round(dev_pct, 2)
                })
            rec["sensor_status"] = sensor_status

            if row["alarm"]:
                opt_vals = {}
                for sensor in SENSOR_COLS:
                    val = float(raw_row[sensor])
                    lo, hi, _ = NORMAL_RANGES[sensor]
                    if val > hi or val < lo:
                        center = (lo + hi) / 2.0
                        opt_vals[sensor] = round(center, 4)
                if opt_vals:
                    rec["optimized_values"] = opt_vals

                max_dev = 0.0
                primary_sensor = None
                for sensor in SENSOR_COLS:
                    val = float(raw_row[sensor])
                    lo, hi, _ = NORMAL_RANGES[sensor]
                    if val > hi:
                        dev = (val - hi) / hi
                    elif val < lo:
                        dev = (lo - val) / lo
                    else:
                        dev = 0.0
                    if dev > max_dev:
                        max_dev = dev
                        primary_sensor = sensor
                if primary_sensor:
                    rec["primary_cause"] = f"{primary_sensor.replace('_', ' ')} out of normal range ({round(max_dev*100,1)}% deviation)"
                    rec["primary_cause_sensor"] = primary_sensor
        enriched_predictions.append(rec)

    # Compute RCA (using the same result, no extra inference)
    rca_result = {}
    if result["alarm"].any():
        alarm_mask = result["alarm"].values
        alarm_count = int(alarm_mask.sum())
        total_count = len(result)

        # Extract the alarmed rows from the raw data and result
        alarm_raw_subset = df_raw.iloc[off:].reset_index(drop=True)[alarm_mask[:len(df_raw)-off]]
        # Reuse the SHAP and AE logic from root_cause_analysis but on the already computed data
        # (Simplified: you can refactor that logic into a separate function)
        feature_cols = models["feature_cols"]
        df_fe = engineer_features(df_raw, context_df)
        X_raw = df_fe[feature_cols].values.astype(np.float32)
        X_raw = np.nan_to_num(X_raw)
        X_sc = models["scaler"].transform(X_raw)
        X_aligned = X_sc[off:]
        alarm_X = X_aligned[alarm_mask]

        # SHAP
        dmatrix = xgb.DMatrix(alarm_X, feature_names=feature_cols)
        shap_raw = models["xgb"].get_booster().predict(dmatrix, pred_contribs=True)
        shap_vals = shap_raw[:, :-1]
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_by_sensor = {s: 0.0 for s in SENSOR_COLS}
        for i, feat in enumerate(feature_cols):
            src = _strip_suffix(feat)
            if src in shap_by_sensor:
                shap_by_sensor[src] += float(mean_abs_shap[i])

        # Autoencoder error
        recon = models["autoencoder"].predict(alarm_X, batch_size=512, verbose=0)
        ae_feat_err = np.mean((alarm_X - recon) ** 2, axis=0)
        ae_by_sensor = {s: 0.0 for s in SENSOR_COLS}
        for i, feat in enumerate(feature_cols):
            src = _strip_suffix(feat)
            if src in ae_by_sensor:
                ae_by_sensor[src] += float(ae_feat_err[i])

        def norm_dict(d):
            mx = max(d.values()) or 1e-8
            return {k: v / mx for k, v in d.items()}

        shap_n = norm_dict(shap_by_sensor)
        ae_n = norm_dict(ae_by_sensor)
        blame = {s: round(0.6 * shap_n[s] + 0.4 * ae_n[s], 4) for s in SENSOR_COLS}
        sorted_sensors = sorted(blame, key=lambda x: blame[x], reverse=True)

        root_causes = []
        for rank, sensor in enumerate(sorted_sensors[:5], start=1):
            lo, hi, unit = NORMAL_RANGES[sensor]
            mean_val = float(alarm_raw_subset[sensor].mean())
            normal_center = (lo + hi) / 2.0
            dev_pct = round(abs(mean_val - normal_center) / normal_center * 100, 2) if normal_center else 0.0
            direction = "above" if mean_val > hi else ("below" if mean_val < lo else "within")
            diag_tmpl = SENSOR_DIAGNOSES.get(sensor, {})
            diagnosis = diag_tmpl.get(direction, f"{sensor} value {mean_val:.2f}{unit} is {direction} normal range ({lo}–{hi}{unit}).")
            full_diag = f"{sensor.replace('_', ' ').title()} is {dev_pct}% {direction} normal range ({mean_val:.2f}{unit} vs {lo}–{hi}{unit}). {diagnosis}"
            root_causes.append({
                "rank": rank,
                "sensor": sensor,
                "blame_score": blame[sensor],
                "mean_value_in_alarm": round(mean_val, 4),
                "normal_range": [lo, hi],
                "unit": unit,
                "deviation_pct": dev_pct,
                "direction": direction,
                "diagnosis": full_diag,
                "recommended_action": RECOMMENDED_ACTIONS.get(sensor, "Consult plant engineer."),
            })

        # Physics insights (reuse from your original rca code)
        physics_insights = {}
        if all(c in alarm_raw_subset.columns for c in ["feed_water_flow", "steam_flow", "tube_skin_temp", "boiler_load", "dissolved_oxygen", "feed_water_ph"]):
            mb = float((alarm_raw_subset["feed_water_flow"] - alarm_raw_subset["steam_flow"]).mean())
            hfi = float((alarm_raw_subset["tube_skin_temp"] * alarm_raw_subset["boiler_load"] / 100.0).mean())
            o2ci = float((alarm_raw_subset["dissolved_oxygen"] * (11.0 - alarm_raw_subset["feed_water_ph"])).mean())
            physics_insights = {
                "mass_balance": round(mb, 4),
                "mass_balance_status": "negative — feed water deficiency detected" if mb < 0 else "positive — normal or feed excess",
                "heat_flux_index": round(hfi, 4),
                "heat_flux_status": "elevated — possible heat flux crisis" if hfi > 400 else "normal",
                "o2_corrosion_index": round(o2ci, 4),
                "o2_corrosion_status": "high — corrosion risk elevated" if o2ci > 12 else "acceptable",
            }

        # Model agreement
        alarm_result = result[alarm_mask]
        model_scores = {
            "autoencoder": round(float(alarm_result["autoencoder"].mean()), 4),
            "isolation_forest": round(float(alarm_result["isolation_forest"].mean()), 4),
            "random_forest": round(float(alarm_result["random_forest"].mean()), 4),
            "xgboost": round(float(alarm_result["xgboost"].mean()), 4),
            "lstm": round(float(alarm_result["lstm"].mean()), 4),
        }
        agreeing = sum(1 for v in model_scores.values() if v >= 0.5)
        consensus_label = f"{'HIGH' if agreeing == 5 else ('MEDIUM' if agreeing >= 3 else 'LOW')} — {agreeing}/5 models agree on anomaly"
        model_scores["consensus"] = consensus_label

        rca_result = {
            "alarm_windows": alarm_count,
            "total_windows": total_count,
            "rca_performed_on": "alarmed_windows",
            "root_causes": root_causes,
            "physics_insights": physics_insights,
            "model_agreement": model_scores,
        }

    # Optimization (run only if there are alarms)
      # ─────────────────────────────────────────────────────────────────
    # Optimization (run only if there are alarms)
    # ─────────────────────────────────────────────────────────────────
    opt_result = {}
    if result["alarm"].any():
        current_risk = float(result["risk_score"].iloc[-1])
        last_n = min(10, len(df_raw))
        last_rows = df_raw.iloc[-last_n:]
        current_vals = {s: float(last_rows[s].mean()) for s in SENSOR_COLS}
        
        # Compute sensitivities using XGBoost only (no full inference)
        sensitivities = {}
        feature_cols = models["feature_cols"]
        
        # Define helper inside this block (or outside once)
        def _quick_risk(vals):
            syn = _make_synthetic_window(vals, n_rows=SEQ_LEN + 5)
            df_fe = engineer_features(syn)
            X_raw = df_fe[feature_cols].values.astype(np.float32)
            X_raw = np.nan_to_num(X_raw)
            X_sc = models["scaler"].transform(X_raw)
            xgb_p = models["xgb"].predict_proba(X_sc)[:, 1]
            return float(xgb_p[-1])
        
        for sensor in SENSOR_COLS:
            lo, hi = PHYSICAL_BOUNDS[sensor]
            v_up = {**current_vals, sensor: min(current_vals[sensor] * 1.05, hi)}
            v_dn = {**current_vals, sensor: max(current_vals[sensor] * 0.95, lo)}
            r_up = _quick_risk(v_up)
            r_dn = _quick_risk(v_dn)
            delta_r = r_up - r_dn
            delta_s = v_up[sensor] - v_dn[sensor]
            sensitivities[sensor] = abs(delta_r / (delta_s + 1e-12))
        
        # Build recommendations
        risk_gap = current_risk - 0.3   # target_risk_score = 0.3
        total_sens = sum(sensitivities.values()) or 1e-8
        recs = []
        new_vals = dict(current_vals)
        
        for sensor in sorted(SENSOR_COLS, key=lambda s: sensitivities[s], reverse=True):
            lo, hi = PHYSICAL_BOUNDS[sensor]
            ideal_lo, ideal_hi, _ = NORMAL_RANGES[sensor]
            ideal_center = (ideal_lo + ideal_hi) / 2.0
            if current_vals[sensor] > ideal_hi:
                recommended = max(ideal_center, lo)
            elif current_vals[sensor] < ideal_lo:
                recommended = min(ideal_center, hi)
            else:
                recommended = current_vals[sensor]
            recommended = float(np.clip(recommended, lo, hi))
            new_vals[sensor] = recommended
            recs.append({
                "sensor": sensor,
                "current_value": round(current_vals[sensor], 4),
                "recommended_value": round(recommended, 4),
                "delta": round(recommended - current_vals[sensor], 4),
                "sensitivity": round(sensitivities[sensor], 6),
                "priority": "HIGH" if sensitivities[sensor] > 0.01 else ("MEDIUM" if sensitivities[sensor] > 0.001 else "LOW"),
            })
        
        expected_risk = _quick_risk(new_vals)
        opt_result = {
            "current_risk_score": round(current_risk, 4),
            "target_risk_score": 0.3,
            "expected_risk_after_optimization": round(expected_risk, 4),
            "optimization_feasible": expected_risk <= 0.33,
            "sensor_recommendations": recs,
            "top_levers": [r["sensor"] for r in recs[:3]],
            "estimated_lead_time_gain_minutes": int(risk_gap / 0.01) if risk_gap > 0 else 0,
        }
    # Energy analysis – uses df_raw directly, no extra inference needed
    energy_result = _compute_energy_metrics(df_raw)
    optimal_targets = {
        "steam_generation_efficiency_pct": 98.0,
        "heat_rate_index": 4.10,
        "blowdown_loss_pct": 1.5,
        "spray_cooling_waste_pct": 5.0,
        "corrosion_drag_index": 8.0,
        "overall_efficiency_score": 90.0,
    }
    # You can compute top wastes similarly – but this is already in your energy_analysis endpoint.

    return {
        "inference_time_ms": (time.perf_counter() - t0) * 1000,
        "summary": _summary_stats(result),
        "predictions": enriched_predictions,
        "rca": rca_result,
        "optimization": opt_result,   # fill with actual computed data
        "energy": {
            "window_rows": len(df_raw),
            "energy_metrics": energy_result,
            "optimal_targets": optimal_targets,
            # ... other fields from your energy_analysis endpoint
        }
    }
    

@router.post("/predict/enriched_full", response_model=EnrichedFullResponse)
async def predict_enriched_full(body: PredictRequest):
    full_data = await compute_full_analysis(body.readings, body.context)
    return EnrichedFullResponse(**full_data)


# ──────────────────────────────────────────────────────────────────────────────
# Web Socket 
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION: Path to your testing CSV
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent  # goes up 3 levels
CSV_PATH = os.getenv("TEST_CSV_PATH", str(PROJECT_ROOT / "Data" / "boiler_testing_data_10k.csv"))
# ──────────────────────────────────────────────────────────────────

@router.websocket("/ws/stream")
# At the top of stream_test_data, after accept()
async def _handle_incoming():
    """Drain any messages from the client (pings etc) without blocking."""
    try:
        async for msg in websocket.iter_text():
            pass  # just discard — we don't need client messages
    except Exception:
        pass

# Don't await it — run alongside the stream
asyncio.create_task(_handle_incoming())

async def stream_test_data(websocket: WebSocket):
    await websocket.accept()
    
    if not Path(CSV_PATH).exists():
        await websocket.send_json({"error": f"CSV file not found at {CSV_PATH}"})
        await websocket.close()
        return
    
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        await websocket.send_json({"error": f"Failed to read CSV: {str(e)}"})
        await websocket.close()
        return
    
    if not rows:
        await websocket.send_json({"error": "CSV file is empty"})
        await websocket.close()
        return
    
    delay_seconds = 2.5
    try:
        for idx, row in enumerate(rows):
            # Check if client is still connected before sending
            try:
                processed_row = {}
                for key, value in row.items():
                    try:
                        processed_row[key] = float(value)
                    except (ValueError, TypeError):
                        processed_row[key] = value
                processed_row["_row_index"] = idx + 1
                
                await websocket.send_json(processed_row)
                await asyncio.sleep(delay_seconds)
            except Exception as e:
                # Client disconnected, exit gracefully
                logger.info(f"WebSocket client disconnected at row {idx+1}: {e}")
                break
        
        # Only send completion if client still connected
        try:
            await websocket.send_json({"message": "Stream complete", "total_rows": len(rows)})
            await asyncio.sleep(1)
            await websocket.close()
        except Exception:
            pass  # Client already gone
    except Exception as e:
        logger.warning(f"WebSocket stream error: {e}")
    finally:
        # Ensure websocket is closed if not already
        try:
            await websocket.close()
        except Exception:
            pass
        
# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main_2:app", host="0.0.0.0", port=port, reload=False)