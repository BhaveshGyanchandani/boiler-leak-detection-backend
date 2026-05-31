"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  STEEL PLANT ENERGY OPTIMISATION — FastAPI Backend  (V10)                   ║
║  Route: EAF → Ladle Furnace → Continuous Caster → Hot Rolling Mill          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ENDPOINTS                                                                   ║
║                                                                              ║
║  ── Core ────────────────────────────────────────────────────────────────── ║
║  GET  /health                  → service + model load status                ║
║  GET  /models/info             → loaded model metadata + weight counts      ║
║  GET  /threshold/anomaly       → current anomaly threshold                  ║
║  PUT  /threshold/anomaly       → override anomaly threshold                 ║
║                                                                              ║
║  ── Prediction ──────────────────────────────────────────────────────────── ║
║  POST /predict                 → KPI prediction for a single window         ║
║  POST /predict/csv             → CSV batch prediction (all windows)         ║
║  POST /predict/csv/faults-only → CSV batch — anomalous windows only        ║
║  POST /predict/enriched        → prediction + RCA + optimization in one     ║
║                                                                              ║
║  ── Optimization ────────────────────────────────────────────────────────── ║
║  POST /optimize                → full TPE→CMA-ES optimization (≤10s)       ║
║  POST /optimize/quick          → fast optimization for real-time UI (≤2s)  ║
║  POST /simulate                → what-if: change setpoints, see KPIs       ║
║  POST /predict/csv/optimized   → CSV batch + per-row optimization          ║
║                                                                              ║
║  ── Analysis & Explainability ───────────────────────────────────────────── ║
║  POST /rca                     → Root Cause Analysis (SHAP + z-score)      ║
║  POST /shap/explain            → SHAP waterfall for a single window        ║
║  POST /anomaly/detect          → LSTM-AE + IsoForest anomaly scores        ║
║  POST /energy/analysis         → detailed energy breakdown per window      ║
║  POST /trends                  → rolling trends + sensor correlations       ║
║  POST /sensors/health-check    → rule-based sensor range checker           ║
║                                                                              ║
║  ── Visualizations (base64 PNG) ─────────────────────────────────────────── ║
║  GET  /charts/backtest         → pre-computed backtest chart                ║
║  GET  /charts/shap             → pre-computed SHAP beeswarm                ║
║  POST /charts/live             → live energy timeline from sensor window   ║
║  POST /charts/pareto           → energy vs yield Pareto front              ║
║  POST /charts/sensor-trends    → multi-panel sensor trend chart            ║
║  GET  /charts/model-performance→ model R² / RMSE bar chart                ║
║                                                                              ║
║  ── Dashboard ───────────────────────────────────────────────────────────── ║
║  GET  /dashboard               → combined KPI + anomaly + model status     ║
║  GET  /dashboard/performance   → surrogate model performance metrics       ║
║                                                                              ║
║  ── Live Streaming ──────────────────────────────────────────────────────── ║
║  WS   /ws/stream               → WebSocket live sensor stream from CSV     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
from fastapi import APIRouter
import asyncio
import base64
import io
import json
import logging
import math
import os
import pickle
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import (
    FastAPI, File, HTTPException, Query, UploadFile, WebSocket,
    WebSocketDisconnect, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings("ignore")

# ── Optional heavy imports (graceful fallback if not installed) ───────────────

# All artifacts that must be present before the API accepts requests
REQUIRED_MODEL_KEYS = {
    "scaler", "rf", "iforest", "meta_learner", "normalizer",
    "autoencoder", "lstm", "xgb", "feature_cols",
}

# ── HuggingFace lazy download helpers ─────────────────────────────────────────
# Files are fetched on first use (inside load_all_models at startup),
# not at import time, so the worker process starts instantly.
# huggingface_hub caches every file under ~/.cache/huggingface by default;
# point HF_HOME / HUGGINGFACE_HUB_CACHE to a persistent volume in production
# so files survive container restarts and are never re-downloaded.
from huggingface_hub import hf_hub_download
router = APIRouter()
HF_REPO_ID   = "ZOROD/Steel-plant-optimization"
HF_REPO_TYPE = "model"
VERSION="V10"
# Mapping from logical key → filename on the Hub
HF_MODEL_FILES: Dict[str, str] = {
    "scaler"         : f"scaler_{VERSION}.pkl",

    "lgb_energy"     : f"lgb_Energy_kWh_per_ton_{VERSION}.pkl",
    "lgb_production" : f"lgb_Production_Rate_tph_{VERSION}.pkl",
    "lgb_yield"      : f"lgb_Steel_Yield_Pct_{VERSION}.pkl",
    "lgb_carbon"     : f"lgb_Tap_Carbon_Pct_{VERSION}.pkl",

    "xgb_energy"     : f"xgb_Energy_kWh_per_ton_{VERSION}.pkl",
    "xgb_production" : f"xgb_Production_Rate_tph_{VERSION}.pkl",
    "xgb_yield"      : f"xgb_Steel_Yield_Pct_{VERSION}.pkl",
    "xgb_carbon"     : f"xgb_Tap_Carbon_Pct_{VERSION}.pkl",

    "cat_energy"     : f"cat_Energy_kWh_per_ton_{VERSION}.cbm",
    "cat_production" : f"cat_Production_Rate_tph_{VERSION}.cbm",
    "cat_yield"      : f"cat_Steel_Yield_Pct_{VERSION}.cbm",
    "cat_carbon"     : f"cat_Tap_Carbon_Pct_{VERSION}.cbm",

    "meta_energy"    : f"meta_Energy_kWh_per_ton_{VERSION}.pkl",
    "meta_production": f"meta_Production_Rate_tph_{VERSION}.pkl",
    "meta_yield"     : f"meta_Steel_Yield_Pct_{VERSION}.pkl",
    "meta_carbon"    : f"meta_Tap_Carbon_Pct_{VERSION}.pkl",

    "iso_forest"     : f"iso_forest_{VERSION}.pkl",
    "conformal"      : f"conformal_{VERSION}.pkl",
    "quantile"       : f"lgb_quantile_{VERSION}.pkl",

    "lstm_seed7"     : f"lstm_ae_seed7_{VERSION}.pt",
    "lstm_seed13"    : f"lstm_ae_seed13_{VERSION}.pt",
    "lstm_seed42"    : f"lstm_ae_seed42_{VERSION}.pt",
}


def _hf_fetch(key: str) -> Path:
    """
    Return the local cached path for a HuggingFace model file.

    • On the first call for a given file the Hub client downloads it and
      stores it in the HF cache (~/.cache/huggingface or $HF_HOME).
    • On every subsequent call the cached copy is returned instantly —
      no network I/O.
    • Call this inside asyncio.to_thread() so the download never blocks
      the event loop.
    """
    filename = HF_MODEL_FILES[key]
    local_path = hf_hub_download(
        repo_id   = HF_REPO_ID,
        filename  = filename,
        repo_type = HF_REPO_TYPE,
        # force_download=False (default) → use cache if present
    )
    logger.info("HF cache hit/download OK: %s → %s", filename, local_path)
    return Path(local_path)


def _hf_path(key: str) -> Path:
    """Synchronous convenience alias — same as _hf_fetch."""
    return _hf_fetch(key)
  
  
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import optuna
    from optuna.samplers import CmaEsSampler, TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False



# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("steel_api")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (mirror V10 notebook exactly)
# ══════════════════════════════════════════════════════════════════════════════
VERSION = "V10"

CONTROLLABLE = [
    "Scrap_Charge_Weight_t", "Electrode_Power_MW", "O2_Blow_Rate_Nm3h",
    "Lance_Height_mm", "Lime_Addition_kg", "DRI_Feed_Rate_tph",
    "LF_Heating_Power_MW", "LF_Argon_Flow_Nlmin",
    "Cast_Speed_mmin", "Tundish_Temp_C", "Mold_Water_Flow_Lmin",
    "Roll_Speed_ms", "Reheat_Furnace_Temp_C",
]

TARGET_COLS = [
    "Energy_kWh_per_ton",
    "Production_Rate_tph",
    "Steel_Yield_Pct",
    "Tap_Carbon_Pct",
]
TARGET_ENERGY    = "Energy_kWh_per_ton"

SENSOR_STATE_COLS = [
    "Scrap_Charge_Weight_t", "Electrode_Power_MW", "O2_Blow_Rate_Nm3h",
    "Lance_Height_mm", "Lime_Addition_kg", "DRI_Feed_Rate_tph",
    "LF_Heating_Power_MW", "LF_Argon_Flow_Nlmin",
    "Cast_Speed_mmin", "Tundish_Temp_C", "Mold_Water_Flow_Lmin",
    "Roll_Speed_ms", "Reheat_Furnace_Temp_C",
    "EAF_Bath_Temp_C", "Bath_Carbon_Pct", "Slag_Basicity",
    "Electrode_Consumption_kgheat", "Bath_Phosphorus_ppm",
    "Tap_Temp_C", "Tundish_Level_mm", "Strand_Surface_Temp_C", "Roll_Force_kN",
]

LSTM_SENSOR_COLS = SENSOR_STATE_COLS  # 22 sensors for AE

# Physical bounds (FIX 3 from V10)
BOUNDS: Dict[str, Tuple[float, float]] = {
    "Scrap_Charge_Weight_t"  : (80,   160),
    "Electrode_Power_MW"     : (40,   100),
    "O2_Blow_Rate_Nm3h"      : (1000, 3500),
    "Lance_Height_mm"        : (1200, 2400),
    "Lime_Addition_kg"       : (1500, 5000),
    "DRI_Feed_Rate_tph"      : (0,    60),
    "LF_Heating_Power_MW"    : (8,    28),
    "LF_Argon_Flow_Nlmin"    : (100,  600),
    "Cast_Speed_mmin"        : (0.6,  1.4),   # V10 FIX 3
    "Tundish_Temp_C"         : (1520, 1600),
    "Mold_Water_Flow_Lmin"   : (2000, 4500),
    "Roll_Speed_ms"          : (2.0,  8.0),
    "Reheat_Furnace_Temp_C"  : (1150, 1280),
}

# Normal operating ranges for health check
NORMAL_RANGES: Dict[str, Tuple[float, float, str]] = {
    "Scrap_Charge_Weight_t"  : (100,  140,   "t"),
    "Electrode_Power_MW"     : (60,   80,    "MW"),
    "O2_Blow_Rate_Nm3h"      : (1800, 2600,  "Nm³/h"),
    "Lance_Height_mm"        : (1400, 2200,  "mm"),
    "Lime_Addition_kg"       : (2500, 4000,  "kg"),
    "DRI_Feed_Rate_tph"      : (20,   45,    "t/h"),
    "LF_Heating_Power_MW"    : (12,   24,    "MW"),
    "LF_Argon_Flow_Nlmin"    : (200,  500,   "Nl/min"),
    "Cast_Speed_mmin"        : (0.7,  1.3,   "m/min"),
    "Tundish_Temp_C"         : (1535, 1590,  "°C"),
    "Mold_Water_Flow_Lmin"   : (2500, 4000,  "L/min"),
    "Roll_Speed_ms"          : (3.0,  7.0,   "m/s"),
    "Reheat_Furnace_Temp_C"  : (1170, 1270,  "°C"),
    "EAF_Bath_Temp_C"        : (1530, 1630,  "°C"),
    "Bath_Carbon_Pct"        : (0.08, 0.55,  "%"),
    "Slag_Basicity"          : (1.8,  3.2,   "ratio"),
    "Tap_Temp_C"             : (1580, 1630,  "°C"),
}

# Operational hard constraints (V10 FIX 3)
CONSTRAINTS = {
    "PRODUCTION_MIN_TPH" : 80.0,
    "PRODUCTION_MAX_TPH" : 141.0,  # corrected from 160
    "STEEL_YIELD_MIN"    : 85.0,
    "TAP_CARBON_MIN"     : 0.04,
    "TAP_CARBON_MAX"     : 0.55,
    "ENERGY_HARD_MAX"    : 480.0,
    "ROC_MAX_PCT"        : 0.10,
}

# RCA diagnosis templates — steel process domain knowledge
RCA_DIAGNOSES: Dict[str, Dict[str, str]] = {
    "Electrode_Power_MW": {
        "above": "Excessive arc power → resistive heating losses, electrode consumption spike, bath superheating.",
        "below": "Insufficient arc power → cold bath, poor scrap melting, process delay.",
    },
    "O2_Blow_Rate_Nm3h": {
        "above": "Over-oxidation → yield loss, excessive iron in slag, CO post-combustion inefficiency.",
        "below": "Under-oxidation → high carbon at tap, incomplete decarburisation, re-blow required.",
    },
    "Scrap_Charge_Weight_t": {
        "above": "Overloaded scrap charge → under-melting risk, longer tap-to-tap time, higher energy/ton.",
        "below": "Under-charged heat → low production, furnace under-utilisation.",
    },
    "Cast_Speed_mmin": {
        "above": "High cast speed → breakout risk, insufficient solidification, quality defects.",
        "below": "Low cast speed → reduced production rate, heat loss, skull formation.",
    },
    "Tundish_Temp_C": {
        "above": "Tundish superheating → refractory erosion, inclusions in product, quality risk.",
        "below": "Tundish underheating → nozzle blockage, casting interruption.",
    },
    "Lime_Addition_kg": {
        "above": "Excess lime → high slag volume, desulphurisation good but basicity too high.",
        "below": "Insufficient lime → acidic slag, poor dephosphorisation, yield loss.",
    },
    "LF_Heating_Power_MW": {
        "above": "LF over-heating → excessive electrode consumption, energy waste.",
        "below": "LF under-heating → cold steel at casting, tundish blockage risk.",
    },
    "Reheat_Furnace_Temp_C": {
        "above": "Reheat furnace over-temperature → scale loss, decarburisation, surface defects.",
        "below": "Reheat furnace under-temperature → poor rolling, mill overload.",
    },
    "EAF_Bath_Temp_C": {
        "above": "Bath superheating → refractory wear, electrode oxidation, energy penalty.",
        "below": "Bath too cold → incomplete melting, high carbon retention.",
    },
    "Bath_Carbon_Pct": {
        "above": "High bath carbon → extended blow required, higher O2 consumption.",
        "below": "Low bath carbon → re-carburisation needed, cost penalty.",
    },
}

RECOMMENDED_ACTIONS: Dict[str, str] = {
    "Electrode_Power_MW"    : "Reduce electrode power by 5–8% if bath temp > 1620°C. Check electrode gap.",
    "O2_Blow_Rate_Nm3h"     : "Adjust O2 flow to target bath carbon 0.08–0.12%. Verify lance height.",
    "Scrap_Charge_Weight_t" : "Optimise scrap mix — reduce heavy scrap fraction. Check DRI ratio.",
    "Cast_Speed_mmin"       : "Adjust cast speed to match tundish temperature. Target 0.9–1.1 m/min.",
    "Tundish_Temp_C"        : "Adjust LF heating time. Check tundish preheating duration.",
    "Lime_Addition_kg"      : "Target slag basicity 2.2–2.8. Adjust lime per heat composition.",
    "LF_Heating_Power_MW"   : "Optimise LF power to reach target tap temperature ±5°C.",
    "Reheat_Furnace_Temp_C" : "Adjust furnace zone temps for target slab exit temp 1220–1240°C.",
    "EAF_Bath_Temp_C"       : "Monitor bath temp trend. Adjust power-off timing.",
    "Bath_Carbon_Pct"       : "Adjust O2 blow pattern. Consider DRI carbon contribution.",
    "DRI_Feed_Rate_tph"     : "Optimise DRI feed rate vs scrap ratio for energy efficiency.",
    "Lance_Height_mm"       : "Adjust lance height for optimal post-combustion efficiency.",
    "LF_Argon_Flow_Nlmin"   : "Adjust argon flow for homogenisation without excessive heat loss.",
    "Roll_Speed_ms"         : "Match roll speed to downstream coiler capacity and gauge target.",
    "Mold_Water_Flow_Lmin"  : "Ensure mold cooling matches cast speed — check oscillation marks.",
}

# ── Paths ─────────────────────────────────────────────────────────────────────
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", f"/outputs/{VERSION}/weights"))
PLOTS_DIR   = Path(os.getenv("PLOTS_DIR",   f"/outputs/{VERSION}/plots"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", f"/outputs/{VERSION}/reports"))
CSV_STREAM_PATH = Path(os.getenv(
    "STREAM_CSV_PATH",
    f"/outputs/{VERSION}/data/steel_plant_synthetic_{VERSION}.csv",
))

# ── Optuna settings ───────────────────────────────────────────────────────────
OPTUNA_TRIALS_TPE_FULL  = 100
OPTUNA_TRIALS_CMA_FULL  = 400
OPTUNA_TRIALS_TPE_QUICK = 30
OPTUNA_TRIALS_CMA_QUICK = 70

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════
models: Dict[str, Any]  = {}
_anomaly_threshold: float = 0.60
_threshold_lock = asyncio.Lock()

# Cached performance metrics (loaded from backtest CSV at startup)
_perf_cache: Dict[str, Any] = {}

# Local cache — set MODEL_CACHE_DIR env var to a persistent volume in production
MODEL_CACHE = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/steel_plant_optimization"))
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# LSTM-AE ARCHITECTURE  (must match V10 notebook exactly)
# ══════════════════════════════════════════════════════════════════════════════
if TORCH_AVAILABLE:
    class LSTMEncoder(nn.Module):
        def __init__(self, n_features, hidden_dim=64, latent_dim=16, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True,
                                dropout=0.1 if num_layers > 1 else 0.0)
            self.fc   = nn.Linear(hidden_dim, latent_dim)
        def forward(self, x):
            _, (h_n, c_n) = self.lstm(x)
            return self.fc(h_n[-1]), h_n, c_n

    class Seq2SeqDecoder(nn.Module):
        def __init__(self, n_features, seq_len, hidden_dim=64, latent_dim=16, num_layers=2):
            super().__init__()
            self.seq_len   = seq_len
            self.num_layers = num_layers
            self.fc_latent  = nn.Linear(latent_dim, hidden_dim)
            self.fc_latent_c= nn.Linear(latent_dim, hidden_dim)
            self.lstm   = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True,
                                  dropout=0.1 if num_layers > 1 else 0.0)
            self.fc_out = nn.Linear(hidden_dim, n_features)
        def forward(self, z, n_features):
            batch = z.size(0)
            h0 = self.fc_latent(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
            c0 = self.fc_latent_c(z).unsqueeze(0).repeat(self.num_layers, 1, 1)
            dec_input = torch.zeros(batch, 1, n_features, device=z.device)
            outputs = []; h, c = h0, c0
            for _ in range(self.seq_len):
                out, (h, c) = self.lstm(dec_input, (h, c))
                step_out = self.fc_out(out); outputs.append(step_out)
                dec_input = step_out
            return torch.cat(outputs, dim=1)

    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features, seq_len=12, hidden_dim=64, latent_dim=16, num_layers=2):
            super().__init__()
            self.n_features = n_features
            self.encoder = LSTMEncoder(n_features, hidden_dim, latent_dim, num_layers)
            self.decoder = Seq2SeqDecoder(n_features, seq_len, hidden_dim, latent_dim, num_layers)
        def forward(self, x):
            z, h_n, c_n = self.encoder(x)
            return self.decoder(z, self.n_features), z

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def _load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _try_load(path: Path, loader=None):
    """Load a file if it exists, return None otherwise."""
    if not path.exists():
        logger.warning("Weight file not found: %s", path)
        return None
    try:
        if loader:
            return loader(path)
        return _load_pkl(path)
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return None


async def load_all_models() -> None:
    """
    Download (if not already cached) and load all V10 weight files into the
    global `models` dict.

    Strategy — low network load:
    • Every hf_hub_download call runs in asyncio.to_thread() so it never
      blocks the event loop.
    • huggingface_hub skips the network entirely when the file is already in
      the local cache (default ~/.cache/huggingface).  Point $HF_HOME to a
      persistent volume in production to survive container restarts.
    • Files are fetched one group at a time (scaler → LGB → XGB → …) so the
      peak concurrent connections to the Hub stay low.
    """
    logger.info("Loading V10 steel plant models from HuggingFace (%s) …", HF_REPO_ID)

    # ── Scaler ────────────────────────────────────────────────────────────────
    try:
        scaler_path = await asyncio.to_thread(_hf_fetch, "scaler")
        scaler_data = await asyncio.to_thread(_try_load, scaler_path)
        if scaler_data:
            models["scaler"]       = scaler_data.get("scaler")
            models["feature_cols"] = scaler_data.get("feature_cols", [])
            logger.info("Scaler loaded. Feature count: %d", len(models.get("feature_cols", [])))
    except Exception as exc:
        logger.error("Scaler download/load failed: %s", exc)

    # ── LightGBM surrogates ───────────────────────────────────────────────────
    models["lgb"] = {}
    _lgb_key_map = {
        "Energy_kWh_per_ton" : "lgb_energy",
        "Production_Rate_tph": "lgb_production",
        "Steel_Yield_Pct"    : "lgb_yield",
        "Tap_Carbon_Pct"     : "lgb_carbon",
    }
    for tgt, hf_key in _lgb_key_map.items():
        try:
            path = await asyncio.to_thread(_hf_fetch, hf_key)
            m    = await asyncio.to_thread(_try_load, path)
            if m: models["lgb"][tgt] = m
        except Exception as exc:
            logger.warning("LGB %s download/load failed: %s", tgt, exc)
    logger.info("LGB loaded: %d / %d targets", len(models["lgb"]), len(TARGET_COLS))

    # ── XGBoost surrogates ────────────────────────────────────────────────────
    models["xgb"] = {}
    _xgb_key_map = {
        "Energy_kWh_per_ton" : "xgb_energy",
        "Production_Rate_tph": "xgb_production",
        "Steel_Yield_Pct"    : "xgb_yield",
        "Tap_Carbon_Pct"     : "xgb_carbon",
    }
    for tgt, hf_key in _xgb_key_map.items():
        try:
            path = await asyncio.to_thread(_hf_fetch, hf_key)
            m    = await asyncio.to_thread(_try_load, path)
            if m: models["xgb"][tgt] = m
        except Exception as exc:
            logger.warning("XGB %s download/load failed: %s", tgt, exc)
    logger.info("XGB loaded: %d / %d targets", len(models["xgb"]), len(TARGET_COLS))

    # ── CatBoost surrogates ───────────────────────────────────────────────────
    models["cat"] = {}
    if CAT_AVAILABLE:
        _cat_key_map = {
            "Energy_kWh_per_ton" : "cat_energy",
            "Production_Rate_tph": "cat_production",
            "Steel_Yield_Pct"    : "cat_yield",
            "Tap_Carbon_Pct"     : "cat_carbon",
        }
        for tgt, hf_key in _cat_key_map.items():
            try:
                path = await asyncio.to_thread(_hf_fetch, hf_key)
                def _load_cat(p=path):
                    m = CatBoostRegressor()
                    m.load_model(str(p))
                    return m
                m = await asyncio.to_thread(_load_cat)
                if m: models["cat"][tgt] = m
            except Exception as exc:
                logger.warning("CatBoost %s download/load failed: %s", tgt, exc)
    logger.info("CatBoost loaded: %d / %d targets", len(models["cat"]), len(TARGET_COLS))

    # ── Ridge meta-learners ───────────────────────────────────────────────────
    models["meta"] = {}
    _meta_key_map = {
        "Energy_kWh_per_ton" : "meta_energy",
        "Production_Rate_tph": "meta_production",
        "Steel_Yield_Pct"    : "meta_yield",
        "Tap_Carbon_Pct"     : "meta_carbon",
    }
    for tgt, hf_key in _meta_key_map.items():
        try:
            path = await asyncio.to_thread(_hf_fetch, hf_key)
            m    = await asyncio.to_thread(_try_load, path)
            if m: models["meta"][tgt] = m
        except Exception as exc:
            logger.warning("Meta %s download/load failed: %s", tgt, exc)
    logger.info("Meta-learners loaded: %d / %d", len(models["meta"]), len(TARGET_COLS))

    # ── OLS physics anchors ───────────────────────────────────────────────────
    # Rebuilt inline from saved LGB scaler context — no separate file needed

    # ── Quantile models ───────────────────────────────────────────────────────
    try:
        qm_path = await asyncio.to_thread(_hf_fetch, "quantile")
        qm      = await asyncio.to_thread(_try_load, qm_path)
        models["quantile"] = qm or {}
    except Exception as exc:
        logger.warning("Quantile download/load failed: %s", exc)
        models["quantile"] = {}
    logger.info("Quantile models loaded: %s", list(models["quantile"].keys()))

    # ── Conformal adjustments ─────────────────────────────────────────────────
    try:
        conf_path = await asyncio.to_thread(_hf_fetch, "conformal")
        conf      = await asyncio.to_thread(_try_load, conf_path)
        models["conformal"] = conf or {}
    except Exception as exc:
        logger.warning("Conformal download/load failed: %s", exc)
        models["conformal"] = {}
    logger.info("Conformal: %s", models["conformal"])

    # ── IsolationForest ───────────────────────────────────────────────────────
    try:
        iso_path = await asyncio.to_thread(_hf_fetch, "iso_forest")
        iso_data = await asyncio.to_thread(_try_load, iso_path)
        if iso_data:
            models["iso_forest"]       = iso_data.get("model")
            models["iso_scaler"]       = iso_data.get("scaler")
            models["iso_feature_cols"] = iso_data.get("feature_cols", [])
    except Exception as exc:
        logger.warning("IsolationForest download/load failed: %s", exc)
    logger.info("IsolationForest: %s", "loaded" if models.get("iso_forest") else "missing")

    # ── LSTM-AE ensemble (3 seeds) ────────────────────────────────────────────
    models["lstm_ae"]      = []
    models["lstm_scalers"] = []
    if TORCH_AVAILABLE:
        for seed in [42, 7, 13]:
            hf_key = f"lstm_seed{seed}"
            try:
                pt_path = await asyncio.to_thread(_hf_fetch, hf_key)
                def _load_lstm(p=pt_path):
                    return torch.load(str(p), map_location=DEVICE)
                ckpt = await asyncio.to_thread(_load_lstm)
                ae   = LSTMAutoencoder(
                    n_features=len(LSTM_SENSOR_COLS), seq_len=12,
                    hidden_dim=64, latent_dim=16, num_layers=2,
                ).to(DEVICE)
                ae.load_state_dict(ckpt["model_state"])
                ae.eval()
                models["lstm_ae"].append(ae)
                models["lstm_scalers"].append(ckpt.get("scaler"))
                logger.info("LSTM-AE seed=%d loaded", seed)
            except Exception as exc:
                logger.warning("LSTM-AE seed=%d download/load failed: %s", seed, exc)
    logger.info("LSTM-AE ensemble: %d / 3 models", len(models["lstm_ae"]))

    # ── Load pre-computed performance metrics from backtest CSV ───────────────
    bt_path = REPORTS_DIR / f"backtest_{VERSION}.csv"
    if bt_path.exists():
        try:
            bt_df = pd.read_csv(bt_path)
            feas  = bt_df[bt_df.get("feasible", pd.Series(True, index=bt_df.index)) == True]
            _perf_cache["backtest"] = {
                "total_windows"      : len(bt_df),
                "feasible_windows"   : len(feas),
                "feasibility_rate_pct": round(len(feas) / len(bt_df) * 100, 1) if len(bt_df) else 0,
                "avg_energy_saving_kWh_t": round(float(feas.get("saving", pd.Series([0])).mean()), 3),
                "avg_model_uncertainty_kWh_t": round(float(bt_df.get("model_std", pd.Series([0])).mean()), 3),
            }
        except Exception as exc:
            logger.warning("Could not load backtest CSV: %s", exc)

    total = (len(models.get("lgb", {})) + len(models.get("xgb", {})) +
             len(models.get("cat", {})) + len(models.get("lstm_ae", [])))
    logger.info("✅ Model loading complete. Artifacts loaded: %d", total)


def _models_ready() -> bool:
    return bool(models.get("lgb") and models.get("scaler") and models.get("feature_cols"))


def _require_models():
    if not _models_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not yet loaded. Retry in a few seconds.",
        )


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (exact mirror of V10 notebook — engineer_features_v10)
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce V10's engineer_features_v10() exactly.
    Operates on a DataFrame that already contains SENSOR_STATE_COLS.
    Must be called on isolated splits only (no cross-boundary leakage).
    """
    df = df.copy().reset_index(drop=True)
    w10, w30, w60 = 2, 6, 12

    roll_base = {
        "EP":    "Electrode_Power_MW",
        "O2":    "O2_Blow_Rate_Nm3h",
        "BathT": "EAF_Bath_Temp_C",
        "CS":    "Cast_Speed_mmin",
        "Prod":  "Production_Rate_tph",
        "Enrg":  "Energy_kWh_per_ton",
    }
    for lbl, col in roll_base.items():
        if col not in df.columns:
            continue
        for win, sfx in [(w10, "10m"), (w30, "30m"), (w60, "60m")]:
            df[f"{lbl}_RM_{sfx}"] = df[col].rolling(win, min_periods=1).mean().round(3)
        df[f"{lbl}_Std_30m"] = df[col].rolling(w30, min_periods=1).std().fillna(0).round(4)

    lag_map = {
        "EP_Lag10m":    ("Electrode_Power_MW",  w10),
        "EP_Lag30m":    ("Electrode_Power_MW",  w30),
        "O2_Lag10m":    ("O2_Blow_Rate_Nm3h",   w10),
        "O2_Lag30m":    ("O2_Blow_Rate_Nm3h",   w30),
        "BathT_Lag10m": ("EAF_Bath_Temp_C",     w10),
        "BathT_Lag30m": ("EAF_Bath_Temp_C",     w30),
        "BathT_Lag60m": ("EAF_Bath_Temp_C",     w60),
        "Enrg_Lag10m":  ("Energy_kWh_per_ton",  w10),
        "Enrg_Lag30m":  ("Energy_kWh_per_ton",  w30),
        "Prod_Lag10m":  ("Production_Rate_tph", w10),
        "Prod_Lag30m":  ("Production_Rate_tph", w30),
        "CS_Lag10m":    ("Cast_Speed_mmin",     w10),
        "Yield_Lag10m": ("Steel_Yield_Pct",     w10),
        "Yield_Lag30m": ("Steel_Yield_Pct",     w30),
    }
    for nc, (src, lag) in lag_map.items():
        if src in df.columns:
            df[nc] = df[src].shift(lag).bfill().round(4)

    roc_pairs = [
        ("dBathT_dt",  "EAF_Bath_Temp_C"),
        ("dEP_dt",     "Electrode_Power_MW"),
        ("dO2_dt",     "O2_Blow_Rate_Nm3h"),
        ("dCS_dt",     "Cast_Speed_mmin"),
        ("dProd_dt",   "Production_Rate_tph"),
        ("dEnrg_dt",   "Energy_kWh_per_ton"),
    ]
    for nc, src in roc_pairs:
        if src in df.columns:
            df[nc] = df[src].diff().fillna(0).round(5)

    if "Bath_Carbon_Pct" in df.columns:
        for win, sfx in [(w10, "10m"), (w30, "30m"), (w60, "60m")]:
            df[f"BathC_RM_{sfx}"] = df["Bath_Carbon_Pct"].rolling(win, min_periods=1).mean().round(4)
        df["BathC_Std_30m"]   = df["Bath_Carbon_Pct"].rolling(w30, min_periods=1).std().fillna(0).round(5)
        df["BathC_Lag10m"]    = df["Bath_Carbon_Pct"].shift(w10).bfill().round(4)
        df["BathC_Lag30m"]    = df["Bath_Carbon_Pct"].shift(w30).bfill().round(4)
        df["BathC_Lag60m"]    = df["Bath_Carbon_Pct"].shift(w60).bfill().round(4)
        df["dBathC_dt"]       = df["Bath_Carbon_Pct"].diff().fillna(0).round(5)
        if "O2_Blow_Rate_Nm3h" in df.columns:
            df["O2_CumSum_1h"]  = df["O2_Blow_Rate_Nm3h"].rolling(w60, min_periods=1).sum().round(1)
            df["BathC_x_O2"]    = (df["Bath_Carbon_Pct"] * df["O2_Blow_Rate_Nm3h"] / 1000).round(5)
        if "Slag_Basicity" in df.columns:
            df["BathC_x_SlagBas"] = (df["Bath_Carbon_Pct"] * df["Slag_Basicity"]).round(5)
        if "Lime_Addition_kg" in df.columns:
            df["O2_Lime_Ratio"] = (df["O2_Blow_Rate_Nm3h"] / (df["Lime_Addition_kg"] + 1e-3)).round(5) if "O2_Blow_Rate_Nm3h" in df.columns else 0
        if "O2_Blow_Rate_Nm3h" in df.columns:
            df["dBathC_x_dO2"]  = (df["dBathC_dt"] * df["O2_Blow_Rate_Nm3h"] / 1000).round(6)

    # Physics interaction terms
    if all(c in df.columns for c in ["Electrode_Power_MW", "Scrap_Charge_Weight_t"]):
        df["Pwr_Scrap_Ratio"]  = (df["Electrode_Power_MW"] / (df["Scrap_Charge_Weight_t"] + 1e-6)).round(5)
        df["O2_Scrap_Ratio"]   = (df["O2_Blow_Rate_Nm3h"]  / (df["Scrap_Charge_Weight_t"] + 1e-6)).round(5) if "O2_Blow_Rate_Nm3h" in df.columns else 0
    if all(c in df.columns for c in ["LF_Heating_Power_MW", "Electrode_Power_MW"]):
        df["LF_EAF_PwrRatio"]  = (df["LF_Heating_Power_MW"] / (df["Electrode_Power_MW"] + 1e-6)).round(5)
    if all(c in df.columns for c in ["Cast_Speed_mmin", "Tundish_Temp_C"]):
        df["CastSpeed_Tundish"]= (df["Cast_Speed_mmin"] * df["Tundish_Temp_C"]).round(2)
    if all(c in df.columns for c in ["Roll_Speed_ms", "Reheat_Furnace_Temp_C"]):
        df["Roll_Reheat_Ratio"]= (df["Roll_Speed_ms"] / (df["Reheat_Furnace_Temp_C"] + 1e-6)).round(8)
    if "EAF_Bath_Temp_C" in df.columns:
        df["BathT_Vol_30m"]    = df["EAF_Bath_Temp_C"].rolling(w30, min_periods=1).std().fillna(0).round(3)
        if "O2_Blow_Rate_Nm3h" in df.columns:
            df["O2_Vol_30m"]   = df["O2_Blow_Rate_Nm3h"].rolling(w30, min_periods=1).std().fillna(0).round(3)
        if "Electrode_Campaign_Age_hrs" in df.columns:
            df["BathT_x_CampAge"] = (df["EAF_Bath_Temp_C"] / (1 + 0.001 * df["Electrode_Campaign_Age_hrs"])).round(2)
        bthT_std = df["EAF_Bath_Temp_C"].rolling(w30, min_periods=1).std().fillna(1)
        bthT_rm  = df["EAF_Bath_Temp_C"].rolling(w30, min_periods=1).mean()
        tmp_z    = np.abs(df["EAF_Bath_Temp_C"] - bthT_rm) / (bthT_std + 1e-6)
        df["Extreme_BathT_Flag"] = (tmp_z > 2.5).astype(int)

    if "Scrap_Charge_Weight_t" in df.columns:
        for win, sfx in [(w10, "10m"), (w30, "30m"), (w60, "60m")]:
            df[f"Scrap_RM_{sfx}"] = df["Scrap_Charge_Weight_t"].rolling(win, min_periods=1).mean().round(3)
        df["Scrap_Lag10m"]  = df["Scrap_Charge_Weight_t"].shift(w10).bfill().round(3)
        df["Scrap_Lag30m"]  = df["Scrap_Charge_Weight_t"].shift(w30).bfill().round(3)
        df["Scrap_Std_30m"] = df["Scrap_Charge_Weight_t"].rolling(w30, min_periods=1).std().fillna(0).round(4)
        if "O2_Blow_Rate_Nm3h" in df.columns:
            df["O2_per_Scrap"]  = (df["O2_Blow_Rate_Nm3h"] / (df["Scrap_Charge_Weight_t"] + 1e-3)).round(4)

    if "Lime_Addition_kg" in df.columns:
        df["Lime_RM_30m"] = df["Lime_Addition_kg"].rolling(w30, min_periods=1).mean().round(2)
        df["Lime_Lag30m"] = df["Lime_Addition_kg"].shift(w30).bfill().round(2)
        if "O2_Blow_Rate_Nm3h" in df.columns:
            df["Lime_x_O2"] = (df["Lime_Addition_kg"] * df["O2_Blow_Rate_Nm3h"] / 1e6).round(6)

    if "Slag_Basicity" in df.columns:
        df["SlagBas_sq"]     = (df["Slag_Basicity"] ** 2).round(4)
        df["SlagBas_RM_30m"] = df["Slag_Basicity"].rolling(w30, min_periods=1).mean().round(4)
        df["SlagBas_Lag30m"] = df["Slag_Basicity"].shift(w30).bfill().round(4)
        if "Cast_Speed_mmin" in df.columns:
            df["CS_x_SlagBas"] = (df["Cast_Speed_mmin"] * df["Slag_Basicity"]).round(4)

    if "Tap_Temp_C" in df.columns:
        df["TapTemp_excess"] = (df["Tap_Temp_C"] - 1620).clip(lower=0).round(2)
        df["TapTemp_RM_30m"] = df["Tap_Temp_C"].rolling(w30, min_periods=1).mean().round(2)

    if "Bath_Carbon_Pct" in df.columns and "Slag_Basicity" in df.columns:
        df["BathC_excess_yield"] = (df["Bath_Carbon_Pct"] - 0.40).clip(lower=0).round(4)

    # Fourier cyclical time features
    if "Hour_of_Day" in df.columns:
        df["Hour_sin"] = np.sin(2 * np.pi * df["Hour_of_Day"] / 24).round(4)
        df["Hour_cos"] = np.cos(2 * np.pi * df["Hour_of_Day"] / 24).round(4)
    if "Day_of_Week" in df.columns:
        df["DoW_sin"] = np.sin(2 * np.pi * df["Day_of_Week"] / 7).round(4)
        df["DoW_cos"] = np.cos(2 * np.pi * df["Day_of_Week"] / 7).round(4)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _build_feature_vector(row: pd.Series) -> np.ndarray:
    """Build a single-row feature vector aligned with FEATURE_COLS."""
    feature_cols = models["feature_cols"]
    vals = []
    for c in feature_cols:
        v = row.get(c, 0.0) if hasattr(row, "get") else (row[c] if c in row.index else 0.0)
        vals.append(float(v) if pd.notna(v) else 0.0)
    return np.array(vals, dtype=np.float32).reshape(1, -1)


def _predict_ensemble(x_raw: np.ndarray, target: str) -> Dict[str, float]:
    """Run the 4-model ensemble for one target. Returns point, model_std."""
    scaler = models["scaler"]
    x_sc   = scaler.transform(x_raw)

    preds = []
    if target in models.get("lgb", {}):
        preds.append(float(models["lgb"][target].predict(x_sc)[0]))
    if target in models.get("xgb", {}):
        preds.append(float(models["xgb"][target].predict(x_sc)[0]))
    if target in models.get("cat", {}):
        preds.append(float(models["cat"][target].predict(x_sc)[0]))

    if not preds:
        return {"point": 0.0, "model_std": 0.0, "lo": 0.0, "hi": 0.0}

    arr_preds = np.array(preds[:3])

    # Meta-learner stacking (expects LGB, XGB, CAT, OLS — pad OLS if missing)
    pad = np.mean(arr_preds)
    S   = np.array([[preds[0] if len(preds) > 0 else pad,
                     preds[1] if len(preds) > 1 else pad,
                     preds[2] if len(preds) > 2 else pad,
                     pad]])
    if target in models.get("meta", {}):
        point = float(models["meta"][target].predict(S)[0])
    else:
        point = float(np.mean(arr_preds))

    model_std = float(arr_preds.std()) if len(arr_preds) > 1 else 0.0

    # Quantile prediction intervals
    lo = hi = point
    if target in models.get("quantile", {}):
        qm = models["quantile"][target]
        if "low" in qm:
            lo = float(qm["low"].predict(x_sc)[0])
        if "high" in qm:
            hi = float(qm["high"].predict(x_sc)[0])

    # Conformal calibration
    if target in models.get("conformal", {}):
        q = models["conformal"][target]
        lo -= q; hi += q

    lo = min(lo, point); hi = max(hi, point)

    return {"point": round(point, 4), "lo": round(lo, 4),
            "hi": round(hi, 4), "model_std": round(model_std, 4)}


def _check_constraints(kpis: Dict[str, Dict]) -> Tuple[bool, Dict[str, float]]:
    e  = kpis.get(TARGET_ENERGY, {}).get("point", 9999)
    pr = kpis.get("Production_Rate_tph", {}).get("point", 0)
    sy = kpis.get("Steel_Yield_Pct", {}).get("point", 0)
    tc = kpis.get("Tap_Carbon_Pct", {}).get("point", 0)

    violations = {
        "Energy_too_high"    : max(0.0, e  - CONSTRAINTS["ENERGY_HARD_MAX"]),
        "Production_too_low" : max(0.0, CONSTRAINTS["PRODUCTION_MIN_TPH"] - pr),
        "Production_too_high": max(0.0, pr - CONSTRAINTS["PRODUCTION_MAX_TPH"]),
        "Yield_too_low"      : max(0.0, CONSTRAINTS["STEEL_YIELD_MIN"] - sy),
        "Carbon_too_low"     : max(0.0, CONSTRAINTS["TAP_CARBON_MIN"]  - tc),
        "Carbon_too_high"    : max(0.0, tc - CONSTRAINTS["TAP_CARBON_MAX"]),
    }
    feasible = all(v == 0.0 for v in violations.values())
    return feasible, violations


def _predict_kpis_for_setpoints(
    setpoints: Dict[str, float],
    context_df: pd.DataFrame,
    use_conformal: bool = True,
) -> Dict[str, Dict]:
    """
    Predict all 4 KPIs for a proposed setpoint dict.
    Uses the V10 2-row mini-series approach (FIX 2).
    """
    if context_df is None or len(context_df) == 0:
        return {}

    ctx_row = context_df.iloc[-1].copy()

    # Build row1 with proposed setpoints
    row1 = dict(ctx_row)
    row1.update(setpoints)

    # Derive process variables from new setpoints (V10 physics approximation)
    ep   = row1.get("Electrode_Power_MW", 70)
    o2   = row1.get("O2_Blow_Rate_Nm3h", 2200)
    lime = row1.get("Lime_Addition_kg", 3200)
    pnorm = (ep - 70) / 30; o2n = (o2 - 2200) / 1000; limn = (lime - 3200) / 1200
    row1["EAF_Bath_Temp_C"] = float(np.clip(1570 + 25 * pnorm, 1500, 1650))
    row1["Bath_Carbon_Pct"] = float(np.clip(0.35 - 0.08 * o2n, 0.04, 0.70))
    row1["Slag_Basicity"]   = float(np.clip(2.4 + 0.5 * limn, 1.6, 3.5))
    row1["Tap_Temp_C"]      = float(np.clip(row1["EAF_Bath_Temp_C"]
                                            + 15 * (row1.get("LF_Heating_Power_MW", 18) - 18) / 10,
                                            1560, 1640))

    ctx_ts = pd.to_datetime(ctx_row.get("Timestamp", pd.Timestamp.now()))
    row1["Timestamp"]    = ctx_ts + pd.Timedelta(minutes=5)
    row1["Hour_of_Day"]  = row1["Timestamp"].hour
    row1["Day_of_Week"]  = row1["Timestamp"].dayofweek

    mini = pd.concat([
        pd.DataFrame([dict(ctx_row)]),
        pd.DataFrame([row1]),
    ], ignore_index=True)
    mini["Timestamp"] = pd.to_datetime(mini["Timestamp"])

    mini_eng = engineer_features(mini)
    feat_row  = mini_eng.iloc[-1]

    # Carry forward LSTM/IsoForest scores from context
    for col in (["LSTM_Anomaly_Score", "IsoForest_Anomaly_Score", "Combined_Anomaly_Score"] +
                [f"LSTM_Latent_{d:02d}" for d in range(16)]):
        if col in ctx_row.index and col in models.get("feature_cols", []):
            feat_row[col] = ctx_row[col]

    x_raw = _build_feature_vector(feat_row)
    return {tgt: _predict_ensemble(x_raw, tgt) for tgt in TARGET_COLS}


def _anomaly_score_for_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Run IsolationForest anomaly detection on a DataFrame."""
    iso_cols = models.get("iso_feature_cols", [])
    iso_sc   = models.get("iso_scaler")
    iso_m    = models.get("iso_forest")

    if not iso_m or not iso_cols or not iso_sc:
        return {"combined_score": 0.0, "flag": False, "iso_score": 0.0, "lstm_score": 0.0}

    available = [c for c in iso_cols if c in df.columns]
    if not available:
        return {"combined_score": 0.0, "flag": False, "iso_score": 0.0, "lstm_score": 0.0}

    X = df[available].ffill().bfill().values
    X_sc  = iso_sc.transform(X)
    raw   = iso_m.decision_function(X_sc)
    norm  = float(1 - (raw[-1] - raw.min()) / (raw.max() - raw.min() + 1e-12))
    flag  = norm > _anomaly_threshold

    return {
        "iso_score"      : round(norm, 4),
        "lstm_score"     : 0.0,           # updated below if AE available
        "combined_score" : round(norm, 4),
        "flag"           : flag,
        "alert_level"    : ("CRITICAL" if norm > 0.8 else
                            "HIGH"     if norm > 0.6 else
                            "ELEVATED" if norm > 0.4 else "NORMAL"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════
def _run_optimisation(
    context_df: pd.DataFrame,
    n_tpe: int = OPTUNA_TRIALS_TPE_FULL,
    n_cma: int = OPTUNA_TRIALS_CMA_FULL,
    top_k: int = 5,
) -> List[Dict]:
    """
    Run TPE warm-up → CMA-ES refinement for energy minimisation.
    Returns list of top-k feasible recommendation dicts.
    """
    if not OPTUNA_AVAILABLE:
        return []

    ctx_row = context_df.iloc[-1].copy()
    current_sp = {k: float(ctx_row.get(k, (lo + hi) / 2))
                  for k, (lo, hi) in BOUNDS.items()}

    roc = CONSTRAINTS["ROC_MAX_PCT"]

    def _build_bounds() -> Dict[str, Tuple[float, float]]:
        out = {}
        for var, (lo, hi) in BOUNDS.items():
            cur = current_sp.get(var, (lo + hi) / 2)
            roc_lo = max(lo, cur * (1 - roc))
            roc_hi = min(hi, cur * (1 + roc))
            if roc_hi - roc_lo < 1e-6:
                m = (hi - lo) * roc
                roc_lo = max(lo, cur - m); roc_hi = min(hi, cur + m)
            if roc_lo >= roc_hi: roc_lo, roc_hi = lo, hi
            if var == "Cast_Speed_mmin": roc_hi = min(roc_hi, 1.4)
            out[var] = (roc_lo, roc_hi)
        return out

    search_bounds = _build_bounds()

    def _objective(trial):
        sp = {var: trial.suggest_float(var, *search_bounds[var]) for var in CONTROLLABLE}
        kpis = _predict_kpis_for_setpoints(sp, context_df)
        _, viols = _check_constraints(kpis)
        trial.set_user_attr("_kpis", kpis)
        trial.set_user_attr("_sp", sp)
        std   = kpis.get(TARGET_ENERGY, {}).get("model_std", 0.0)
        point = kpis.get(TARGET_ENERGY, {}).get("point", 9999)
        return (point + std) + 500.0 * sum(v ** 2 for v in viols.values())

    # Phase 1: TPE
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(n_startup_trials=20, multivariate=True, seed=42),
    )
    study.optimize(_objective, n_trials=n_tpe, show_progress_bar=False)

    # Phase 2: CMA-ES from best TPE point
    if study.best_trial:
        sigma0 = np.mean([(b[1] - b[0]) * 0.10 for b in search_bounds.values()])
        study.sampler = CmaEsSampler(
            x0=study.best_params, sigma0=sigma0, seed=42, restart_strategy="ipop"
        )
        study.optimize(_objective, n_trials=n_cma, show_progress_bar=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible  = sorted(
        [t for t in completed
         if t.user_attrs.get("_kpis") and _check_constraints(t.user_attrs["_kpis"])[0]],
        key=lambda t: t.values[0],
    )

    if not feasible:
        # Return best near-feasible
        near = sorted(completed,
                      key=lambda t: sum(_check_constraints(t.user_attrs["_kpis"])[1].values())
                      if t.user_attrs.get("_kpis") else 1e9)
        if near and near[0].user_attrs.get("_kpis"):
            t = near[0]; kpis = t.user_attrs["_kpis"]; sp = t.user_attrs["_sp"]
            _, viols = _check_constraints(kpis)
            return [_format_recommendation(sp, kpis, ctx_row, False, viols)]
        return []

    return [
        _format_recommendation(
            t.user_attrs["_sp"],
            t.user_attrs["_kpis"],
            ctx_row, True, {},
        )
        for t in feasible[:top_k]
    ]


def _format_recommendation(
    sp: Dict, kpis: Dict, ctx_row: pd.Series,
    feasible: bool, violations: Dict,
) -> Dict:
    actual_e = float(ctx_row.get(TARGET_ENERGY, 0))
    opt_e    = kpis.get(TARGET_ENERGY, {}).get("point", 0)
    saving   = actual_e - opt_e

    setpoint_changes = []
    for var in CONTROLLABLE:
        cur = float(ctx_row.get(var, 0))
        rec = sp.get(var, cur)
        delta = rec - cur
        pct   = delta / (abs(cur) + 1e-9) * 100
        within_roc = abs(pct) <= CONSTRAINTS["ROC_MAX_PCT"] * 100 + 0.1
        setpoint_changes.append({
            "setpoint"          : var,
            "current_value"     : round(cur, 4),
            "recommended_value" : round(rec, 4),
            "delta"             : round(delta, 4),
            "change_pct"        : round(pct, 2),
            "within_roc_cap"    : within_roc,
            "unit"              : BOUNDS.get(var, (0, 0)).__class__.__name__,
        })

    return {
        "feasible"              : feasible,
        "violations"            : violations,
        "kpi_predictions"       : {
            tgt: {
                "point"      : kpis[tgt]["point"],
                "lo"         : kpis[tgt]["lo"],
                "hi"         : kpis[tgt]["hi"],
                "model_std"  : kpis[tgt].get("model_std", 0.0),
            }
            for tgt in TARGET_COLS if tgt in kpis
        },
        "energy_saving_kWh_t"   : round(saving, 3),
        "energy_saving_pct"     : round(saving / (actual_e + 1e-9) * 100, 2),
        "setpoint_changes"      : setpoint_changes,
        "top_3_actions"         : [
            s["setpoint"] for s in sorted(
                setpoint_changes, key=lambda x: abs(x["delta"]), reverse=True
            )[:3]
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ROOT CAUSE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def _run_rca(
    context_df: pd.DataFrame,
    target: str = TARGET_ENERGY,
) -> Dict[str, Any]:
    """
    SHAP-based RCA + z-score sensor analysis.
    Returns driver ranking, diagnosis text, and recommended actions.
    """
    if context_df is None or len(context_df) < 2:
        return {"error": "Need at least 2 rows for RCA."}

    eng = engineer_features(context_df.copy())
    row = eng.iloc[-1]
    x_raw = _build_feature_vector(row)
    scaler = models["scaler"]
    x_sc = scaler.transform(x_raw)
    feature_cols = models["feature_cols"]

    # ── SHAP via LightGBM TreeExplainer ──────────────────────────────────────
    shap_drivers = []
    if SHAP_AVAILABLE and target in models.get("lgb", {}):
        try:
            explainer  = shap.TreeExplainer(models["lgb"][target])
            shap_vals  = explainer.shap_values(x_sc)[0]
            shap_pairs = sorted(
                [(feature_cols[i], float(shap_vals[i])) for i in range(len(feature_cols))],
                key=lambda x: abs(x[1]), reverse=True,
            )
            for feat, sv in shap_pairs[:15]:
                # Map back to controllable setpoint if possible
                source = next((c for c in CONTROLLABLE if feat.startswith(c[:8])), feat)
                shap_drivers.append({
                    "feature"        : feat,
                    "source_setpoint": source,
                    "shap_value"     : round(sv, 5),
                    "direction"      : "increases" if sv > 0 else "decreases",
                    "target_impact"  : f"{'+' if sv > 0 else ''}{round(sv, 3)} {target}",
                })
        except Exception as exc:
            logger.warning("SHAP failed: %s", exc)

    # ── Z-score based sensor deviation analysis ───────────────────────────────
    sensor_deviations = []
    for sensor, (lo, hi, unit) in NORMAL_RANGES.items():
        if sensor not in context_df.columns:
            continue
        vals   = context_df[sensor].dropna()
        if len(vals) == 0:
            continue
        center = (lo + hi) / 2.0
        half   = (hi - lo) / 2.0
        cur    = float(vals.iloc[-1])
        z      = (cur - center) / (half + 1e-9)
        direction = "above" if cur > hi else ("below" if cur < lo else "within")
        status_label = ("CRITICAL" if abs(z) > 2.5 else
                        "WARNING"  if abs(z) > 1.5 else
                        "ELEVATED" if abs(z) > 0.8 else "NORMAL")
        diagnosis = ""
        if direction != "within":
            diag_map = RCA_DIAGNOSES.get(sensor, {})
            diagnosis = diag_map.get(direction, "")

        sensor_deviations.append({
            "sensor"          : sensor,
            "current_value"   : round(cur, 4),
            "normal_range"    : [lo, hi],
            "unit"            : unit,
            "z_score"         : round(z, 3),
            "status"          : status_label,
            "direction"       : direction,
            "deviation_pct"   : round(abs(cur - center) / (center + 1e-9) * 100, 2),
            "diagnosis"       : diagnosis,
            "recommended_action": RECOMMENDED_ACTIONS.get(sensor, "Monitor closely.") if direction != "within" else "",
        })

    sensor_deviations.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    critical = [s for s in sensor_deviations if s["status"] in ("CRITICAL", "WARNING")]

    # ── Failure mode classification ───────────────────────────────────────────
    failure_modes = _classify_failure_mode(sensor_deviations, context_df)

    # ── Current KPI prediction ────────────────────────────────────────────────
    ctx_sp = {k: float(context_df.iloc[-1].get(k, 0)) for k in CONTROLLABLE
              if k in context_df.columns}
    kpis   = {tgt: _predict_ensemble(x_sc, tgt) for tgt in TARGET_COLS}

    return {
        "target_kpi"              : target,
        "current_kpi_prediction"  : kpis,
        "shap_drivers"            : shap_drivers,
        "sensor_deviations"       : sensor_deviations,
        "critical_sensors"        : critical,
        "failure_modes"           : failure_modes,
        "top_5_root_causes"       : [s["sensor"] for s in sensor_deviations[:5] if s["status"] != "NORMAL"],
        "immediate_actions"       : [s["recommended_action"] for s in critical[:3] if s["recommended_action"]],
        "overall_health_score"    : _compute_health_score(sensor_deviations),
    }


def _classify_failure_mode(
    deviations: List[Dict], df: pd.DataFrame
) -> List[Dict]:
    """Rule-based failure mode classification from sensor patterns."""
    modes = []
    dev_map = {d["sensor"]: d for d in deviations}

    # High energy mode
    ep  = dev_map.get("Electrode_Power_MW", {})
    o2  = dev_map.get("O2_Blow_Rate_Nm3h", {})
    if ep.get("z_score", 0) > 1.5:
        modes.append({
            "mode"       : "Arc Power Excess",
            "severity"   : "HIGH",
            "probability": min(1.0, abs(ep["z_score"]) / 3),
            "description": "Electrode power significantly above normal — resistive heating losses dominating.",
            "kpi_impact" : "Energy +8–15 kWh/t",
        })

    # Yield loss mode
    bath_c = dev_map.get("Bath_Carbon_Pct", {})
    if o2.get("z_score", 0) > 1.5 or bath_c.get("z_score", 0) < -1.5:
        modes.append({
            "mode"       : "Over-oxidation / Yield Loss",
            "severity"   : "MEDIUM",
            "probability": min(1.0, max(abs(o2.get("z_score", 0)), abs(bath_c.get("z_score", 0))) / 3),
            "description": "Excess oxygen blowing → iron oxidation into slag → steel yield loss.",
            "kpi_impact" : "Yield -2–5%",
        })

    # Casting risk
    cs  = dev_map.get("Cast_Speed_mmin", {})
    tun = dev_map.get("Tundish_Temp_C", {})
    if cs.get("z_score", 0) > 2.0 or tun.get("z_score", 0) < -1.5:
        modes.append({
            "mode"       : "Casting Instability Risk",
            "severity"   : "CRITICAL",
            "probability": min(1.0, max(abs(cs.get("z_score", 0)), abs(tun.get("z_score", 0))) / 3),
            "description": "High cast speed or low tundish temp → breakout risk or nozzle blockage.",
            "kpi_impact" : "Production -10–20 t/h, quality defect risk",
        })

    # Carbon out of spec
    tap = dev_map.get("Tap_Temp_C", {})
    if bath_c.get("z_score", 0) > 1.8:
        modes.append({
            "mode"       : "High Tap Carbon",
            "severity"   : "MEDIUM",
            "probability": min(1.0, abs(bath_c.get("z_score", 0)) / 3),
            "description": "Bath carbon too high at tap → re-blow required → energy penalty.",
            "kpi_impact" : "Tap Carbon >0.45%, re-blow adds 10–20 kWh/t",
        })

    return sorted(modes, key=lambda x: {"CRITICAL": 3, "HIGH": 2, "MEDIUM": 1}.get(x["severity"], 0), reverse=True)


def _compute_health_score(deviations: List[Dict]) -> float:
    """Compute overall plant health score 0–100."""
    if not deviations:
        return 100.0
    penalties = {"CRITICAL": 20, "WARNING": 10, "ELEVATED": 4, "NORMAL": 0}
    total_pen = sum(penalties.get(d["status"], 0) for d in deviations)
    return max(0.0, round(100.0 - total_pen, 1))


# ══════════════════════════════════════════════════════════════════════════════
# CHART GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def _fig_to_b64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _chart_energy_timeline(df: pd.DataFrame, predictions: List[Dict]) -> Optional[str]:
    if not MPL_AVAILABLE or not predictions:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        for spine in ax.spines.values(): spine.set_edgecolor("#444")

    ts   = [p.get("timestamp", i) for i, p in enumerate(predictions)]
    pts  = [p.get(TARGET_ENERGY, {}).get("point", 0) for p in predictions]
    lo_v = [p.get(TARGET_ENERGY, {}).get("lo", 0)    for p in predictions]
    hi_v = [p.get(TARGET_ENERGY, {}).get("hi", 0)    for p in predictions]

    axes[0].plot(range(len(pts)), pts, color="#00d4aa", lw=2, label="Predicted Energy")
    axes[0].fill_between(range(len(pts)), lo_v, hi_v, alpha=0.25, color="#00d4aa", label="P10–P90 CI")
    axes[0].axhline(CONSTRAINTS["ENERGY_HARD_MAX"], ls="--", color="#ff4444", lw=1, label="Hard cap")
    axes[0].set_ylabel("kWh / ton", color="white"); axes[0].legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")
    axes[0].set_title("Predicted Energy Consumption with Confidence Interval", color="white", fontsize=11)

    prod = [p.get("Production_Rate_tph", {}).get("point", 0) for p in predictions]
    axes[1].plot(range(len(prod)), prod, color="#7b61ff", lw=2, label="Production")
    axes[1].axhline(CONSTRAINTS["PRODUCTION_MIN_TPH"], ls="--", color="orange", lw=1, label="Min bound")
    axes[1].axhline(CONSTRAINTS["PRODUCTION_MAX_TPH"], ls="--", color="red",    lw=1, label="Max bound")
    axes[1].set_ylabel("t / h", color="white"); axes[1].set_xlabel("Window index", color="white")
    axes[1].legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")
    axes[1].set_title("Predicted Production Rate", color="white", fontsize=11)

    fig.suptitle(f"Steel Plant Live Monitor — {VERSION}", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_sensor_trends(df: pd.DataFrame, sensors: List[str] = None) -> Optional[str]:
    if not MPL_AVAILABLE:
        return None
    sensors = sensors or CONTROLLABLE[:6]
    cols    = [s for s in sensors if s in df.columns]
    if not cols:
        return None

    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    if n == 1: axes = [axes]
    fig.patch.set_facecolor("#0f1117")

    for ax, col in zip(axes, cols):
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white"); ax.yaxis.label.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        ax.plot(df[col].values, color="#00d4aa", lw=1.5)
        if col in NORMAL_RANGES:
            lo, hi, unit = NORMAL_RANGES[col]
            ax.axhline(lo, ls="--", color="orange", lw=0.8, alpha=0.7)
            ax.axhline(hi, ls="--", color="orange", lw=0.8, alpha=0.7)
            ax.fill_between(range(len(df)), lo, hi, alpha=0.07, color="#00d4aa")
            ax.set_ylabel(f"{unit}", color="white", fontsize=8)
        ax.set_title(col.replace("_", " "), color="white", fontsize=9, pad=3)

    axes[-1].set_xlabel("Time step", color="white")
    fig.suptitle("Sensor Trend Monitor", color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_rca_waterfall(rca_result: Dict) -> Optional[str]:
    if not MPL_AVAILABLE:
        return None
    drivers = rca_result.get("shap_drivers", [])
    if not drivers:
        return None

    top = drivers[:10]
    labels = [d["feature"][:25] for d in top]
    vals   = [d["shap_value"] for d in top]
    colors = ["#ff4444" if v > 0 else "#00d4aa" for v in vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax.barh(labels[::-1], vals[::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(0, color="white", lw=0.8)
    ax.set_xlabel("SHAP value (impact on energy prediction)", color="white")
    ax.set_title(f"Root Cause Analysis — SHAP Drivers\n({rca_result.get('target_kpi', '')})",
                 color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_pareto(optimization_results: List[Dict]) -> Optional[str]:
    if not MPL_AVAILABLE or not optimization_results:
        return None
    energies = [r["kpi_predictions"].get(TARGET_ENERGY, {}).get("point", 0) for r in optimization_results]
    yields   = [r["kpi_predictions"].get("Steel_Yield_Pct", {}).get("point", 0) for r in optimization_results]
    prods    = [r["kpi_predictions"].get("Production_Rate_tph", {}).get("point", 0) for r in optimization_results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    sc1 = axes[0].scatter(energies, yields, c=prods, cmap="plasma", s=60, alpha=0.8)
    plt.colorbar(sc1, ax=axes[0], label="Production (t/h)")
    axes[0].set_xlabel("Energy (kWh/t)", color="white"); axes[0].set_ylabel("Yield (%)", color="white")
    axes[0].set_title("Energy vs Yield\n(colour = Production)", color="white")
    if energies: axes[0].scatter([energies[0]], [yields[0]], c="red", s=100, zorder=5, label="Best")
    axes[0].legend(fontsize=8, facecolor="#1a1d27", labelcolor="white")

    sc2 = axes[1].scatter(energies, prods, c=yields, cmap="viridis", s=60, alpha=0.8)
    plt.colorbar(sc2, ax=axes[1], label="Yield (%)")
    axes[1].set_xlabel("Energy (kWh/t)", color="white"); axes[1].set_ylabel("Production (t/h)", color="white")
    axes[1].set_title("Energy vs Production\n(colour = Yield)", color="white")

    fig.suptitle("Multi-Objective Pareto Front", color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _fig_to_b64(fig)


def _chart_model_performance() -> Optional[str]:
    """Load backtest CSV and draw model performance bar chart."""
    if not MPL_AVAILABLE:
        return None
    bt_path = REPORTS_DIR / f"backtest_{VERSION}.csv"
    if not bt_path.exists():
        return None
    try:
        bt = pd.read_csv(bt_path)
    except Exception:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes:
        ax.set_facecolor("#1a1d27"); ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    if "saving" in bt.columns:
        feas = bt[bt.get("feasible", True) == True] if "feasible" in bt else bt
        axes[0].hist(feas["saving"].clip(-20, 40), bins=20, color="#00d4aa", alpha=0.8, edgecolor="#444")
        axes[0].axvline(0, color="white", lw=0.8)
        axes[0].set_xlabel("Energy saving (kWh/t)", color="white")
        axes[0].set_title("Distribution of Energy Savings", color="white")

    if "model_std" in bt.columns:
        axes[1].plot(bt["model_std"].values, color="#7b61ff", lw=1.5)
        axes[1].set_xlabel("Window index", color="white")
        axes[1].set_ylabel("Ensemble std (kWh/t)", color="white")
        axes[1].set_title("Model Uncertainty Over Backtest", color="white")

    fig.suptitle(f"Surrogate Model Performance — {VERSION}", color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return _fig_to_b64(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════════════════
class SetpointReading(BaseModel):
    """Single 5-minute window of sensor + setpoint values."""
    timestamp              : Optional[str]   = None
    # 13 controllable setpoints
    Scrap_Charge_Weight_t  : float = Field(120.0,  ge=80,   le=160)
    Electrode_Power_MW     : float = Field(70.0,   ge=40,   le=100)
    O2_Blow_Rate_Nm3h      : float = Field(2200.0, ge=1000, le=3500)
    Lance_Height_mm        : float = Field(1800.0, ge=1200, le=2400)
    Lime_Addition_kg       : float = Field(3200.0, ge=1500, le=5000)
    DRI_Feed_Rate_tph      : float = Field(30.0,   ge=0,    le=60)
    LF_Heating_Power_MW    : float = Field(18.0,   ge=8,    le=28)
    LF_Argon_Flow_Nlmin    : float = Field(350.0,  ge=100,  le=600)
    Cast_Speed_mmin        : float = Field(0.9,    ge=0.6,  le=1.4)
    Tundish_Temp_C         : float = Field(1560.0, ge=1520, le=1600)
    Mold_Water_Flow_Lmin   : float = Field(3200.0, ge=2000, le=4500)
    Roll_Speed_ms          : float = Field(4.5,    ge=2.0,  le=8.0)
    Reheat_Furnace_Temp_C  : float = Field(1220.0, ge=1150, le=1280)
    # Optional process state sensors (improves prediction accuracy)
    EAF_Bath_Temp_C        : Optional[float] = None
    Bath_Carbon_Pct        : Optional[float] = None
    Slag_Basicity          : Optional[float] = None
    Electrode_Consumption_kgheat: Optional[float] = None
    Bath_Phosphorus_ppm    : Optional[float] = None
    Tap_Temp_C             : Optional[float] = None
    Tundish_Level_mm       : Optional[float] = None
    Strand_Surface_Temp_C  : Optional[float] = None
    Roll_Force_kN          : Optional[float] = None
    # Metadata
    Electrode_Campaign_Age_hrs: Optional[float] = None
    Shift_ID               : Optional[int]   = None
    Steel_Grade_ID         : Optional[int]   = None
    Hour_of_Day            : Optional[int]   = None
    Day_of_Week            : Optional[int]   = None


class PredictRequest(BaseModel):
    readings : List[SetpointReading] = Field(..., min_length=2)
    context  : Optional[List[SetpointReading]] = None

    @field_validator("readings")
    @classmethod
    def validate_readings(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 readings are required for feature engineering.")
        return v


class SimulateRequest(BaseModel):
    """Change one or more setpoints and see the KPI impact."""
    current_readings    : List[SetpointReading] = Field(..., min_length=2)
    setpoint_overrides  : Dict[str, float]      = Field(...,
        description="Dict of setpoint_name → new_value. Only specify what changes.")

    @field_validator("setpoint_overrides")
    @classmethod
    def validate_overrides(cls, v):
        invalid = [k for k in v if k not in CONTROLLABLE]
        if invalid:
            raise ValueError(f"Unknown setpoints: {invalid}. Valid: {CONTROLLABLE}")
        return v


class RCARequest(BaseModel):
    readings : List[SetpointReading] = Field(..., min_length=2)
    target   : str = Field(TARGET_ENERGY,
                           description=f"KPI to analyse. One of: {TARGET_COLS}")
    context  : Optional[List[SetpointReading]] = None


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)


class OptimizeRequest(BaseModel):
    readings     : List[SetpointReading] = Field(..., min_length=2)
    context      : Optional[List[SetpointReading]] = None
    top_k        : int = Field(5, ge=1, le=10)


class ChartRequest(BaseModel):
    readings : List[SetpointReading] = Field(..., min_length=2)
    sensors  : Optional[List[str]] = None


def _readings_to_df(readings: List[SetpointReading]) -> pd.DataFrame:
    """Convert a list of SetpointReading Pydantic objects to a DataFrame."""
    rows = []
    for r in readings:
        d = r.model_dump()
        ts = d.pop("timestamp", None)
        if ts:
            d["Timestamp"] = pd.to_datetime(ts)
        rows.append(d)
    df = pd.DataFrame(rows)
    if "Timestamp" not in df.columns:
        df["Timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="5min")

    # Fill derived state columns from physics if not provided
    if "EAF_Bath_Temp_C" not in df.columns or df["EAF_Bath_Temp_C"].isna().all():
        ep = df["Electrode_Power_MW"]
        df["EAF_Bath_Temp_C"] = np.clip(1570 + 25 * (ep - 70) / 30, 1500, 1650)
    if "Bath_Carbon_Pct" not in df.columns or df["Bath_Carbon_Pct"].isna().all():
        o2 = df["O2_Blow_Rate_Nm3h"]
        df["Bath_Carbon_Pct"] = np.clip(0.35 - 0.08 * (o2 - 2200) / 1000, 0.04, 0.70)
    if "Slag_Basicity" not in df.columns or df["Slag_Basicity"].isna().all():
        lime = df["Lime_Addition_kg"]
        df["Slag_Basicity"] = np.clip(2.4 + 0.5 * (lime - 3200) / 1200, 1.6, 3.5)
    if "Tap_Temp_C" not in df.columns or df["Tap_Temp_C"].isna().all():
        df["Tap_Temp_C"] = np.clip(df["EAF_Bath_Temp_C"] + 15 * (df["LF_Heating_Power_MW"] - 18) / 10, 1560, 1640)
    if "Hour_of_Day" not in df.columns:
        df["Hour_of_Day"] = df["Timestamp"].dt.hour
    if "Day_of_Week" not in df.columns:
        df["Day_of_Week"] = df["Timestamp"].dt.dayofweek
    if "Shift_ID" not in df.columns:
        df["Shift_ID"] = (df["Hour_of_Day"] // 8).astype(int)
    if "Steel_Grade_ID" not in df.columns:
        df["Steel_Grade_ID"] = 1
    if "Electrode_Campaign_Age_hrs" not in df.columns:
        df["Electrode_Campaign_Age_hrs"] = 0.0

    # Stub out remaining state cols at physically reasonable defaults
    defaults = {
        "Electrode_Consumption_kgheat": 4.0,
        "Bath_Phosphorus_ppm"         : 120.0,
        "Tundish_Level_mm"            : 750.0,
        "Strand_Surface_Temp_C"       : 1080.0,
        "Roll_Force_kN"               : 18000.0,
    }
    for col, val in defaults.items():
        if col not in df.columns or df[col].isna().all():
            df[col] = val

    # Stub out target columns if absent (needed by engineer_features)
    for tgt in TARGET_COLS:
        if tgt not in df.columns:
            df[tgt] = 400.0  # placeholder — not used in prediction, only for feature lags

    return df


def _df_predict_all(df: pd.DataFrame) -> List[Dict]:
    """Predict KPIs for every row in an engineered DataFrame."""
    eng    = engineer_features(df)
    scaler = models["scaler"]
    feat   = models["feature_cols"]
    results = []
    for i in range(len(eng)):
        row = eng.iloc[i]
        x_raw = np.array(
            [float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0 for c in feat],
            dtype=np.float32
        ).reshape(1, -1)
        x_sc = scaler.transform(x_raw)
        pred  = {tgt: _predict_ensemble(x_sc, tgt) for tgt in TARGET_COLS}
        feasible, violations = _check_constraints(pred)
        anomaly = _anomaly_score_for_df(df.iloc[:i+1])
        results.append({
            "row_index"      : i,
            "timestamp"      : str(df.iloc[i].get("Timestamp", "")),
            "kpi_predictions": pred,
            "feasible"       : feasible,
            "violations"     : violations,
            "anomaly"        : anomaly,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Steel Plant API starting up — loading V10 models …")
    await load_all_models()
    logger.info("✅ Ready to serve requests.")
    yield
    logger.info("👋 Shutting down.")


app = FastAPI(
    title="Steel Plant Energy Optimisation API",
    description=(
        "End-to-end AI advisory system for the EAF→LF→CC→HRM route.\n\n"
        "Surrogate: LGB + XGB + CatBoost + Ridge ensemble\n"
        "Anomaly: LSTM-AE (3-seed) + IsolationForest\n"
        "Optimiser: Optuna TPE → CMA-ES\n"
        "Explainability: SHAP TreeExplainer\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# ── CORE ENDPOINTS ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/health", tags=["core"])
async def health():
    """Service status + model load summary."""
    return {
        "status"         : "ok" if _models_ready() else "loading",
        "version"        : VERSION,
        "models_loaded"  : {
            "lgb"        : list(models.get("lgb", {}).keys()),
            "xgb"        : list(models.get("xgb", {}).keys()),
            "cat"        : list(models.get("cat", {}).keys()),
            "meta"       : list(models.get("meta", {}).keys()),
            "quantile"   : list(models.get("quantile", {}).keys()),
            "lstm_ae"    : len(models.get("lstm_ae", [])),
            "iso_forest" : models.get("iso_forest") is not None,
            "scaler"     : models.get("scaler")     is not None,
            "conformal"  : bool(models.get("conformal")),
        },
        "feature_count"  : len(models.get("feature_cols", [])),
        "anomaly_threshold": _anomaly_threshold,
        "constraints"    : CONSTRAINTS,
        "torch_device"   : str(DEVICE) if DEVICE else "unavailable",
        "weights_dir"    : str(WEIGHTS_DIR),
    }


@router.get("/models/info", tags=["core"])
async def models_info():
    """Detailed metadata for every loaded model artifact."""
    _require_models()
    info = {
        "version"        : VERSION,
        "feature_columns": models.get("feature_cols", []),
        "feature_count"  : len(models.get("feature_cols", [])),
        "target_cols"    : TARGET_COLS,
        "controllable_setpoints": CONTROLLABLE,
        "constraints"    : CONSTRAINTS,
        "bounds"         : {k: list(v) for k, v in BOUNDS.items()},
        "conformal_adjustments": models.get("conformal", {}),
        "lstm_ae_ensemble_size": len(models.get("lstm_ae", [])),
    }
    lgb_info = {}
    for tgt, m in models.get("lgb", {}).items():
        lgb_info[tgt] = {
            "n_estimators": getattr(m, "n_estimators_", None),
            "best_iteration": getattr(m, "best_iteration_", None),
        }
    info["lgb"] = lgb_info
    for name in ["xgb", "meta"]:
        info[name] = {tgt: "loaded" for tgt in models.get(name, {})}
    info["backtest_performance"] = _perf_cache.get("backtest", {})
    return info


@router.get("/threshold/anomaly", tags=["core"])
async def get_threshold():
    return {"threshold": _anomaly_threshold, "description": "Combined anomaly score threshold (0–1)"}


@router.put("/threshold/anomaly", tags=["core"])
async def set_threshold(body: ThresholdUpdate):
    global _anomaly_threshold
    async with _threshold_lock:
        old = _anomaly_threshold
        _anomaly_threshold = body.threshold
    return {"old_threshold": old, "new_threshold": _anomaly_threshold}


# ══════════════════════════════════════════════════════════════════════════════
# ── PREDICTION ENDPOINTS ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/predict", tags=["prediction"])
async def predict(body: PredictRequest):
    """
    Predict all 4 KPIs for every row in `readings`.
    Optionally supply `context` (earlier rows) for rolling-feature warm-up.
    """
    _require_models()
    t0 = time.perf_counter()

    ctx_df = _readings_to_df(body.context) if body.context else None
    df     = _readings_to_df(body.readings)

    if ctx_df is not None:
        combined = pd.concat([ctx_df, df], ignore_index=True)
        eng = engineer_features(combined).iloc[len(ctx_df):].reset_index(drop=True)
    else:
        eng = engineer_features(df)

    scaler = models["scaler"]
    feat   = models["feature_cols"]
    predictions = []
    for i in range(len(eng)):
        row   = eng.iloc[i]
        x_raw = np.array(
            [float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0 for c in feat],
            dtype=np.float32
        ).reshape(1, -1)
        x_sc  = scaler.transform(x_raw)
        pred  = {tgt: _predict_ensemble(x_sc, tgt) for tgt in TARGET_COLS}
        feasible, violations = _check_constraints(pred)
        anom  = _anomaly_score_for_df(df.iloc[:i+1])
        predictions.append({
            "row_index"       : i,
            "timestamp"       : str(df.iloc[i].get("Timestamp", "")),
            "kpi_predictions" : pred,
            "feasible"        : feasible,
            "violations"      : {k: round(v, 4) for k, v in violations.items() if v > 0},
            "anomaly"         : anom,
        })

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "version"          : VERSION,
        "inference_time_ms": round(elapsed, 2),
        "windows_scored"   : len(predictions),
        "summary"          : {
            "mean_energy"       : round(float(np.mean([p["kpi_predictions"][TARGET_ENERGY]["point"] for p in predictions])), 2),
            "mean_production"   : round(float(np.mean([p["kpi_predictions"]["Production_Rate_tph"]["point"] for p in predictions])), 2),
            "mean_yield"        : round(float(np.mean([p["kpi_predictions"]["Steel_Yield_Pct"]["point"] for p in predictions])), 2),
            "feasible_windows"  : sum(1 for p in predictions if p["feasible"]),
            "anomaly_windows"   : sum(1 for p in predictions if p["anomaly"]["flag"]),
        },
        "predictions"      : predictions,
    }


@router.post("/predict/csv", tags=["prediction"])
async def predict_csv(
    file: UploadFile = File(...),
    faults_only: bool = Query(False, description="If true, return only anomalous windows"),
):
    """Batch prediction from a CSV file upload."""
    _require_models()
    t0 = time.perf_counter()

    raw = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    if "Timestamp" in df_raw.columns:
        df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"], errors="coerce")

    missing = [c for c in CONTROLLABLE if c not in df_raw.columns]
    if missing:
        raise HTTPException(422, f"Missing required columns: {missing}")

    if len(df_raw) < 2:
        raise HTTPException(422, "CSV must have at least 2 rows.")

    # Fill derived columns
    for tgt in TARGET_COLS:
        if tgt not in df_raw.columns:
            df_raw[tgt] = 400.0
    if "EAF_Bath_Temp_C" not in df_raw.columns:
        ep = df_raw["Electrode_Power_MW"]
        df_raw["EAF_Bath_Temp_C"] = np.clip(1570 + 25 * (ep - 70) / 30, 1500, 1650)
    if "Bath_Carbon_Pct" not in df_raw.columns:
        df_raw["Bath_Carbon_Pct"] = np.clip(0.35 - 0.08 * (df_raw["O2_Blow_Rate_Nm3h"] - 2200) / 1000, 0.04, 0.70)
    if "Slag_Basicity" not in df_raw.columns:
        df_raw["Slag_Basicity"] = np.clip(2.4 + 0.5 * (df_raw["Lime_Addition_kg"] - 3200) / 1200, 1.6, 3.5)
    if "Tap_Temp_C" not in df_raw.columns:
        df_raw["Tap_Temp_C"] = np.clip(df_raw["EAF_Bath_Temp_C"] + 15 * (df_raw["LF_Heating_Power_MW"] - 18) / 10, 1560, 1640)
    if "Hour_of_Day" not in df_raw.columns:
        if "Timestamp" in df_raw.columns:
            df_raw["Hour_of_Day"] = df_raw["Timestamp"].dt.hour
        else:
            df_raw["Hour_of_Day"] = 12
    if "Day_of_Week" not in df_raw.columns:
        df_raw["Day_of_Week"] = 0
    if "Shift_ID" not in df_raw.columns:
        df_raw["Shift_ID"] = (df_raw["Hour_of_Day"] // 8).astype(int)
    if "Steel_Grade_ID" not in df_raw.columns:
        df_raw["Steel_Grade_ID"] = 1
    if "Electrode_Campaign_Age_hrs" not in df_raw.columns:
        df_raw["Electrode_Campaign_Age_hrs"] = 0.0

    eng    = engineer_features(df_raw)
    scaler = models["scaler"]
    feat   = models["feature_cols"]

    predictions = []
    for i in range(len(eng)):
        row   = eng.iloc[i]
        x_raw = np.array(
            [float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0 for c in feat],
            dtype=np.float32
        ).reshape(1, -1)
        x_sc  = scaler.transform(x_raw)
        pred  = {tgt: _predict_ensemble(x_sc, tgt) for tgt in TARGET_COLS}
        feasible, violations = _check_constraints(pred)
        anom  = _anomaly_score_for_df(df_raw.iloc[:i+1])

        if faults_only and not anom["flag"]:
            continue

        predictions.append({
            "row_index"       : i,
            "timestamp"       : str(df_raw.iloc[i].get("Timestamp", i)),
            "kpi_predictions" : pred,
            "feasible"        : feasible,
            "violations"      : {k: round(v, 4) for k, v in violations.items() if v > 0},
            "anomaly"         : anom,
        })

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "filename"         : file.filename,
        "rows_in"          : len(df_raw),
        "windows_returned" : len(predictions),
        "faults_only"      : faults_only,
        "inference_time_ms": round(elapsed, 2),
        "summary"          : {
            "total_rows"        : len(df_raw),
            "anomaly_rows"      : sum(1 for p in predictions if p["anomaly"]["flag"]),
            "infeasible_rows"   : sum(1 for p in predictions if not p["feasible"]),
            "mean_energy_kWh_t" : round(float(np.mean([p["kpi_predictions"][TARGET_ENERGY]["point"] for p in predictions])), 2) if predictions else 0,
        },
        "predictions"      : predictions,
    }


@router.post("/predict/enriched", tags=["prediction"])
async def predict_enriched(body: PredictRequest):
    """
    All-in-one endpoint: prediction + RCA + quick optimization for every window.
    Ideal for the main demo dashboard.
    """
    _require_models()
    t0 = time.perf_counter()

    ctx_df = _readings_to_df(body.context) if body.context else None
    df     = _readings_to_df(body.readings)

    # Prediction
    pred_resp = await predict(body)

    # RCA on the last window
    rca_df = pd.concat([ctx_df, df], ignore_index=True) if ctx_df is not None else df
    rca_result = await asyncio.to_thread(_run_rca, rca_df, TARGET_ENERGY)

    # Quick optimization
    opt_df     = rca_df
    opt_result = await asyncio.to_thread(
        _run_optimisation, opt_df,
        OPTUNA_TRIALS_TPE_QUICK, OPTUNA_TRIALS_CMA_QUICK, 3,
    ) if OPTUNA_AVAILABLE else []

    # Anomaly on last row
    anomaly = _anomaly_score_for_df(df)

    # Charts
    charts = {}
    if MPL_AVAILABLE:
        charts["energy_timeline"] = _chart_energy_timeline(df, pred_resp["predictions"])
        charts["rca_waterfall"]   = _chart_rca_waterfall(rca_result)

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "version"            : VERSION,
        "inference_time_ms"  : round(elapsed, 2),
        "predictions"        : pred_resp["predictions"],
        "summary"            : pred_resp["summary"],
        "rca"                : rca_result,
        "optimization"       : opt_result,
        "anomaly"            : anomaly,
        "charts_base64"      : charts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── OPTIMIZATION ENDPOINTS ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/optimize", tags=["optimization"])
async def optimize(body: OptimizeRequest):
    """
    Full TPE(100) → CMA-ES(400) optimization.
    Returns top-k hard-feasible setpoint bundles with energy saving %.
    Typical runtime: 5–10 seconds.
    """
    _require_models()
    if not OPTUNA_AVAILABLE:
        raise HTTPException(503, "Optuna not installed.")
    t0  = time.perf_counter()
    ctx = _readings_to_df(body.context) if body.context else None
    df  = _readings_to_df(body.readings)
    opt_df = pd.concat([ctx, df], ignore_index=True) if ctx is not None else df
    recs   = await asyncio.to_thread(
        _run_optimisation, opt_df,
        OPTUNA_TRIALS_TPE_FULL, OPTUNA_TRIALS_CMA_FULL, body.top_k,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "version"               : VERSION,
        "optimization_time_ms"  : round(elapsed, 2),
        "n_recommendations"     : len(recs),
        "trials_run"            : OPTUNA_TRIALS_TPE_FULL + OPTUNA_TRIALS_CMA_FULL,
        "recommendations"       : recs,
        "roc_cap_pct"           : CONSTRAINTS["ROC_MAX_PCT"] * 100,
    }


@router.post("/optimize/quick", tags=["optimization"])
async def optimize_quick(body: OptimizeRequest):
    """
    Fast TPE(30) → CMA(70) optimization for real-time UI updates.
    Typical runtime: 1–2 seconds.
    """
    _require_models()
    if not OPTUNA_AVAILABLE:
        raise HTTPException(503, "Optuna not installed.")
    t0  = time.perf_counter()
    ctx = _readings_to_df(body.context) if body.context else None
    df  = _readings_to_df(body.readings)
    opt_df = pd.concat([ctx, df], ignore_index=True) if ctx is not None else df
    recs   = await asyncio.to_thread(
        _run_optimisation, opt_df,
        OPTUNA_TRIALS_TPE_QUICK, OPTUNA_TRIALS_CMA_QUICK, min(body.top_k, 3),
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "version"            : VERSION,
        "optimization_time_ms": round(elapsed, 2),
        "mode"               : "quick",
        "n_recommendations"  : len(recs),
        "recommendations"    : recs,
    }


@router.post("/simulate", tags=["optimization"])
async def simulate(body: SimulateRequest):
    """
    What-if simulation: change specific setpoints and see the predicted KPI change.
    Returns baseline KPIs + simulated KPIs + delta for each target.
    """
    _require_models()
    t0  = time.perf_counter()
    df  = _readings_to_df(body.current_readings)
    eng = engineer_features(df)

    # Validate overrides against physical bounds + RoC cap
    ctx_row     = df.iloc[-1]
    warnings_   = []
    clean_sp    = {}
    for sp, val in body.setpoint_overrides.items():
        lo, hi = BOUNDS.get(sp, (0, 1e9))
        cur    = float(ctx_row.get(sp, (lo + hi) / 2))
        if val < lo:
            warnings_.append(f"{sp}: {val} below physical min {lo} — clamped.")
            val = lo
        elif val > hi:
            warnings_.append(f"{sp}: {val} above physical max {hi} — clamped.")
            val = hi
        pct = abs(val - cur) / (abs(cur) + 1e-9) * 100
        if pct > CONSTRAINTS["ROC_MAX_PCT"] * 100:
            warnings_.append(f"{sp}: {pct:.1f}% change exceeds ±{CONSTRAINTS['ROC_MAX_PCT']*100:.0f}% RoC cap.")
        clean_sp[sp] = val

    # Baseline KPIs
    scaler   = models["scaler"]
    feat     = models["feature_cols"]
    row_eng  = eng.iloc[-1]
    x_base   = np.array([float(row_eng.get(c, 0.0)) if pd.notna(row_eng.get(c, 0.0)) else 0.0 for c in feat], dtype=np.float32).reshape(1, -1)
    x_base_sc = scaler.transform(x_base)
    baseline_kpis = {tgt: _predict_ensemble(x_base_sc, tgt) for tgt in TARGET_COLS}

    # Simulated KPIs with new setpoints
    sim_kpis = _predict_kpis_for_setpoints(clean_sp, df)

    # Deltas
    deltas = {}
    for tgt in TARGET_COLS:
        b = baseline_kpis[tgt]["point"]
        s = sim_kpis.get(tgt, {}).get("point", b)
        deltas[tgt] = {
            "baseline" : round(b, 4),
            "simulated": round(s, 4),
            "delta"    : round(s - b, 4),
            "delta_pct": round((s - b) / (abs(b) + 1e-9) * 100, 2),
            "direction": "improved" if (tgt == TARGET_ENERGY and s < b) or (tgt != TARGET_ENERGY and s > b) else "worsened",
        }

    feasible_base, viols_base = _check_constraints(baseline_kpis)
    feasible_sim,  viols_sim  = _check_constraints(sim_kpis)
    elapsed = (time.perf_counter() - t0) * 1000

    return {
        "version"            : VERSION,
        "simulation_time_ms" : round(elapsed, 2),
        "setpoint_overrides" : clean_sp,
        "guardrail_warnings" : warnings_,
        "baseline"           : {"kpis": baseline_kpis, "feasible": feasible_base},
        "simulated"          : {"kpis": sim_kpis,      "feasible": feasible_sim, "violations": viols_sim},
        "kpi_deltas"         : deltas,
        "energy_saving_kWh_t": round(baseline_kpis[TARGET_ENERGY]["point"] - sim_kpis.get(TARGET_ENERGY, baseline_kpis[TARGET_ENERGY])["point"], 3),
    }


@router.post("/predict/csv/optimized", tags=["optimization"])
async def predict_csv_optimized(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=5),
):
    """
    CSV bulk upload: for every row, run quick optimization and return
    both the current KPIs and the recommended setpoint adjustments.
    """
    _require_models()
    if not OPTUNA_AVAILABLE:
        raise HTTPException(503, "Optuna not installed.")
    t0 = time.perf_counter()

    raw = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Cannot parse CSV: {exc}")

    if len(df_raw) < 2:
        raise HTTPException(422, "CSV must have at least 2 rows.")

    # Fill required columns
    for tgt in TARGET_COLS:
        if tgt not in df_raw.columns: df_raw[tgt] = 400.0
    if "EAF_Bath_Temp_C" not in df_raw.columns:
        df_raw["EAF_Bath_Temp_C"] = np.clip(1570 + 25*(df_raw["Electrode_Power_MW"]-70)/30, 1500, 1650)
    if "Bath_Carbon_Pct" not in df_raw.columns:
        df_raw["Bath_Carbon_Pct"] = np.clip(0.35-0.08*(df_raw["O2_Blow_Rate_Nm3h"]-2200)/1000, 0.04, 0.70)
    if "Slag_Basicity" not in df_raw.columns:
        df_raw["Slag_Basicity"] = np.clip(2.4+0.5*(df_raw["Lime_Addition_kg"]-3200)/1200, 1.6, 3.5)
    if "Tap_Temp_C" not in df_raw.columns:
        df_raw["Tap_Temp_C"] = np.clip(df_raw["EAF_Bath_Temp_C"]+15*(df_raw["LF_Heating_Power_MW"]-18)/10, 1560, 1640)
    for col, val in [("Hour_of_Day",12),("Day_of_Week",0),("Shift_ID",1),("Steel_Grade_ID",1),("Electrode_Campaign_Age_hrs",0.0)]:
        if col not in df_raw.columns: df_raw[col] = val

    STRIDE = max(1, len(df_raw) // 20)  # sample at most 20 windows to avoid timeout
    results = []

    for i in range(0, len(df_raw), STRIDE):
        window = df_raw.iloc[max(0, i-5):i+1].reset_index(drop=True)
        recs   = await asyncio.to_thread(
            _run_optimisation, window,
            OPTUNA_TRIALS_TPE_QUICK, OPTUNA_TRIALS_CMA_QUICK, top_k,
        )
        eng   = engineer_features(window)
        row   = eng.iloc[-1]
        scaler = models["scaler"]; feat = models["feature_cols"]
        x_raw = np.array([float(row.get(c,0)) if pd.notna(row.get(c,0)) else 0 for c in feat], dtype=np.float32).reshape(1,-1)
        x_sc  = scaler.transform(x_raw)
        curr_kpis = {tgt: _predict_ensemble(x_sc, tgt) for tgt in TARGET_COLS}

        results.append({
            "row_index"      : i,
            "timestamp"      : str(df_raw.iloc[i].get("Timestamp", i)),
            "current_kpis"   : curr_kpis,
            "recommendations": recs,
            "best_saving_kWh_t": recs[0]["energy_saving_kWh_t"] if recs else 0,
        })

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "filename"          : file.filename,
        "rows_in"           : len(df_raw),
        "windows_optimized" : len(results),
        "optimization_time_ms": round(elapsed, 2),
        "results"           : results,
        "summary"           : {
            "avg_best_saving_kWh_t": round(float(np.mean([r["best_saving_kWh_t"] for r in results])), 3),
            "windows_with_saving"  : sum(1 for r in results if r["best_saving_kWh_t"] > 0),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── ANALYSIS ENDPOINTS ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/rca", tags=["analysis"])
async def rca(body: RCARequest):
    """
    Root Cause Analysis for the last window in `readings`.
    Returns: SHAP drivers, z-score deviations, failure mode classification,
    top-5 root causes, diagnosis text, and immediate action recommendations.
    """
    _require_models()
    if body.target not in TARGET_COLS:
        raise HTTPException(422, f"target must be one of {TARGET_COLS}")
    t0   = time.perf_counter()
    ctx  = _readings_to_df(body.context) if body.context else None
    df   = _readings_to_df(body.readings)
    full = pd.concat([ctx, df], ignore_index=True) if ctx is not None else df
    result = await asyncio.to_thread(_run_rca, full, body.target)

    # Add RCA waterfall chart
    if MPL_AVAILABLE:
        result["chart_base64"] = _chart_rca_waterfall(result)

    result["rca_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


@router.post("/shap/explain", tags=["analysis"])
async def shap_explain(body: PredictRequest):
    """
    SHAP waterfall explanation for the last window in `readings`.
    Returns top-20 feature contributions with values and directions.
    """
    _require_models()
    if not SHAP_AVAILABLE:
        raise HTTPException(503, "SHAP not installed.")
    t0  = time.perf_counter()
    df  = _readings_to_df(body.readings)
    eng = engineer_features(df)
    row = eng.iloc[-1]
    feat = models["feature_cols"]
    x_raw = np.array([float(row.get(c, 0)) if pd.notna(row.get(c, 0)) else 0 for c in feat], dtype=np.float32).reshape(1, -1)
    x_sc  = models["scaler"].transform(x_raw)

    shap_results = {}
    for tgt in TARGET_COLS:
        if tgt not in models.get("lgb", {}):
            continue
        try:
            exp  = shap.TreeExplainer(models["lgb"][tgt])
            sv   = exp.shap_values(x_sc)[0]
            pairs= sorted(
                [(feat[i], float(sv[i]), float(x_sc[0, i])) for i in range(len(feat))],
                key=lambda x: abs(x[1]), reverse=True,
            )[:20]
            shap_results[tgt] = [
                {"feature": f, "shap_value": round(s, 5),
                 "feature_value": round(v, 4),
                 "direction": "↑ increases" if s > 0 else "↓ decreases",
                 "source": next((c for c in CONTROLLABLE if f.startswith(c[:8])), "process_state")}
                for f, s, v in pairs
            ]
        except Exception as exc:
            shap_results[tgt] = {"error": str(exc)}

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "shap_explanations": shap_results,
        "shap_time_ms"     : round(elapsed, 2),
        "feature_count"    : len(feat),
    }


@router.post("/anomaly/detect", tags=["analysis"])
async def anomaly_detect(body: PredictRequest):
    """
    Run IsolationForest anomaly detection on all provided readings.
    Returns per-row anomaly score, flag, and alert level.
    """
    _require_models()
    t0  = time.perf_counter()
    df  = _readings_to_df(body.readings)
    iso_cols = models.get("iso_feature_cols", [])
    iso_sc   = models.get("iso_scaler")
    iso_m    = models.get("iso_forest")

    if not iso_m or not iso_cols or not iso_sc:
        return {"error": "IsolationForest not loaded.", "anomaly_results": []}

    available = [c for c in iso_cols if c in df.columns]
    X    = df[available].ffill().bfill().values
    X_sc = iso_sc.transform(X)
    raw  = iso_m.decision_function(X_sc)
    mn   = raw.min(); mx = raw.max()
    norm = 1 - (raw - mn) / (mx - mn + 1e-12)

    results = []
    for i in range(len(df)):
        sc   = float(norm[i])
        flag = sc > _anomaly_threshold
        results.append({
            "row_index"  : i,
            "timestamp"  : str(df.iloc[i].get("Timestamp", i)),
            "iso_score"  : round(sc, 4),
            "flag"       : flag,
            "alert_level": ("CRITICAL" if sc > 0.8 else "HIGH" if sc > 0.6 else "ELEVATED" if sc > 0.4 else "NORMAL"),
        })

    elapsed = (time.perf_counter() - t0) * 1000
    anomalous = [r for r in results if r["flag"]]
    return {
        "threshold"     : _anomaly_threshold,
        "total_windows" : len(results),
        "anomaly_count" : len(anomalous),
        "anomaly_rate"  : round(len(anomalous) / len(results) * 100, 2) if results else 0,
        "anomaly_results": results,
        "anomalous_windows": anomalous,
        "inference_time_ms": round(elapsed, 2),
    }


@router.post("/energy/analysis", tags=["analysis"])
async def energy_analysis(body: PredictRequest):
    """
    Detailed energy breakdown analysis for the provided sensor window.
    Returns efficiency metrics, loss components, and optimisation potential.
    """
    _require_models()
    t0  = time.perf_counter()
    df  = _readings_to_df(body.readings)

    # Predict current energy
    eng  = engineer_features(df)
    row  = eng.iloc[-1]
    feat = models["feature_cols"]
    x_raw = np.array([float(row.get(c, 0)) if pd.notna(row.get(c, 0)) else 0 for c in feat], dtype=np.float32).reshape(1, -1)
    x_sc  = models["scaler"].transform(x_raw)
    e_pred = _predict_ensemble(x_sc, TARGET_ENERGY)

    # Energy component breakdown (physics-based)
    ep   = float(df["Electrode_Power_MW"].mean())
    o2   = float(df["O2_Blow_Rate_Nm3h"].mean())
    scrap= float(df["Scrap_Charge_Weight_t"].mean())
    cs   = float(df["Cast_Speed_mmin"].mean())
    lfheat = float(df["LF_Heating_Power_MW"].mean())
    reheat = float(df["Reheat_Furnace_Temp_C"].mean())

    arc_energy       = round(ep * 0.55, 2)                           # ~55% of total
    o2_blowing_loss  = round(o2 / 1000 * 3.5, 2)                    # O2 pumping + post-comb
    scrap_handling   = round(scrap * 0.12, 2)                        # handling & preheating proxy
    lf_energy        = round(lfheat * 0.8, 2)                        # LF contribution
    reheat_energy    = round((reheat - 1150) * 0.08, 2)              # above baseline
    casting_aux      = round((1.4 - cs) * 15, 2)                     # slow cast uses more heat
    total_est        = round(arc_energy + o2_blowing_loss + scrap_handling + lf_energy + reheat_energy + casting_aux, 2)

    opt_potential    = max(0, e_pred["point"] - 370)   # 370 = theoretical minimum

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "version"              : VERSION,
        "predicted_energy"     : e_pred,
        "energy_components_kWh_t": {
            "arc_power"            : arc_energy,
            "o2_blowing_losses"    : o2_blowing_loss,
            "scrap_handling"       : scrap_handling,
            "ladle_furnace"        : lf_energy,
            "reheat_furnace"       : reheat_energy,
            "casting_auxiliary"    : casting_aux,
            "total_estimated"      : total_est,
        },
        "efficiency_metrics"   : {
            "arc_efficiency_pct"   : round(min(100, 95 - max(0, ep - 75) * 0.5), 1),
            "yield_estimated_pct"  : round(min(98, 94 - max(0, o2 - 2200) / 100), 1),
            "heat_utilisation_pct" : round(min(95, 88 + (cs - 0.8) * 10), 1),
        },
        "optimisation_potential_kWh_t": round(opt_potential, 2),
        "estimated_cost_saving_per_heat_usd": round(opt_potential * scrap * 0.035, 2),
        "analysis_time_ms"     : round(elapsed, 2),
    }


@router.post("/trends", tags=["analysis"])
async def trends(body: PredictRequest):
    """
    Rolling trend and sensor correlation analysis for the provided window.
    """
    _require_models()
    t0  = time.perf_counter()
    df  = _readings_to_df(body.readings)

    trend_results = {}
    for col in CONTROLLABLE:
        if col not in df.columns: continue
        s = df[col]
        trend_results[col] = {
            "mean"      : round(float(s.mean()), 4),
            "std"       : round(float(s.std()), 4),
            "min"       : round(float(s.min()), 4),
            "max"       : round(float(s.max()), 4),
            "trend"     : "up"   if float(s.iloc[-1]) > float(s.mean()) else "down",
            "trend_pct" : round((float(s.iloc[-1]) - float(s.mean())) / (abs(float(s.mean())) + 1e-9) * 100, 2),
        }

    # Pearson correlations between controllable setpoints and energy target proxy
    corr_with_energy = {}
    proxy = df["Electrode_Power_MW"] + 0.01 * df["O2_Blow_Rate_Nm3h"]  # energy proxy
    for col in CONTROLLABLE:
        if col in df.columns and len(df) > 2:
            c = float(df[col].corr(proxy))
            corr_with_energy[col] = round(c, 4) if not math.isnan(c) else 0.0

    elapsed = (time.perf_counter() - t0) * 1000
    return {
        "sensor_trends"         : trend_results,
        "correlation_with_energy": corr_with_energy,
        "top_correlated"        : sorted(corr_with_energy.items(), key=lambda x: abs(x[1]), reverse=True)[:5],
        "window_size"           : len(df),
        "analysis_time_ms"      : round(elapsed, 2),
    }


@router.post("/sensors/health-check", tags=["analysis"])
async def sensors_health_check(body: PredictRequest):
    """Rule-based sensor range checker for all 13 controllable setpoints."""
    _require_models()
    df  = _readings_to_df(body.readings)
    row = df.iloc[-1]

    sensor_status = []
    for sensor, (lo, hi, unit) in NORMAL_RANGES.items():
        if sensor not in df.columns:
            continue
        val  = float(row.get(sensor, (lo + hi) / 2))
        ctr  = (lo + hi) / 2
        half = (hi - lo) / 2
        z    = (val - ctr) / (half + 1e-9)
        dir_ = "above" if val > hi else ("below" if val < lo else "within")
        st   = "CRITICAL" if abs(z) > 2.5 else "WARNING" if abs(z) > 1.5 else "ELEVATED" if abs(z) > 0.8 else "NORMAL"
        sensor_status.append({
            "sensor"         : sensor,
            "value"          : round(val, 4),
            "unit"           : unit,
            "normal_range"   : [lo, hi],
            "status"         : st,
            "z_score"        : round(z, 3),
            "direction"      : dir_,
            "diagnosis"      : RCA_DIAGNOSES.get(sensor, {}).get(dir_, "Within normal range.") if dir_ != "within" else "Normal.",
            "action"         : RECOMMENDED_ACTIONS.get(sensor, "") if dir_ != "within" else "",
        })

    critical  = [s for s in sensor_status if s["status"] == "CRITICAL"]
    warning   = [s for s in sensor_status if s["status"] == "WARNING"]
    health_sc = _compute_health_score(sensor_status)

    return {
        "overall_health_score": health_sc,
        "health_label"        : "CRITICAL" if health_sc < 40 else "POOR" if health_sc < 65 else "FAIR" if health_sc < 80 else "GOOD",
        "total_sensors"       : len(sensor_status),
        "critical_count"      : len(critical),
        "warning_count"       : len(warning),
        "sensor_status"       : sensor_status,
        "critical_sensors"    : critical,
        "priority_actions"    : [s["action"] for s in (critical + warning)[:3] if s["action"]],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── VISUALIZATION ENDPOINTS ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/charts/backtest", tags=["charts"])
async def chart_backtest():
    """Return the pre-computed backtest PNG (base64) from the V10 plots dir."""
    path = PLOTS_DIR / f"01_backtest_{VERSION}.png"
    if path.exists():
        data = path.read_bytes()
        return {"chart_base64": base64.b64encode(data).decode("utf-8"),
                "source": "pre-computed", "file": str(path)}
    if MPL_AVAILABLE:
        chart = _chart_model_performance()
        return {"chart_base64": chart, "source": "generated"}
    return {"chart_base64": None, "error": "No backtest chart available."}


@router.get("/charts/shap", tags=["charts"])
async def chart_shap():
    """Return the pre-computed SHAP beeswarm PNG (base64) from the V10 plots dir."""
    path = PLOTS_DIR / f"02_shap_beeswarm_{VERSION}.png"
    if path.exists():
        data = path.read_bytes()
        return {"chart_base64": base64.b64encode(data).decode("utf-8"),
                "source": "pre-computed"}
    return {"chart_base64": None, "error": "SHAP chart not found. Run the V10 notebook first."}


@router.get("/charts/model-performance", tags=["charts"])
async def chart_model_performance():
    """Generate model performance chart from backtest CSV."""
    chart = _chart_model_performance() if MPL_AVAILABLE else None
    return {"chart_base64": chart, "source": "generated"}


@router.post("/charts/live", tags=["charts"])
async def chart_live(body: ChartRequest):
    """Generate a live energy timeline chart from current readings."""
    _require_models()
    df = _readings_to_df(body.readings)
    recs = await predict(PredictRequest(readings=body.readings))
    chart = _chart_energy_timeline(df, recs["predictions"]) if MPL_AVAILABLE else None
    return {"chart_base64": chart, "windows": len(recs["predictions"])}


@router.post("/charts/sensor-trends", tags=["charts"])
async def chart_sensor_trends(body: ChartRequest):
    """Generate a multi-panel sensor trend chart."""
    df = _readings_to_df(body.readings)
    sensors = body.sensors or CONTROLLABLE[:6]
    chart   = _chart_sensor_trends(df, sensors) if MPL_AVAILABLE else None
    return {"chart_base64": chart, "sensors_plotted": sensors}


@router.post("/charts/pareto", tags=["charts"])
async def chart_pareto(body: OptimizeRequest):
    """Run optimization and plot the energy vs yield vs production Pareto front."""
    _require_models()
    if not OPTUNA_AVAILABLE:
        raise HTTPException(503, "Optuna not installed.")
    df     = _readings_to_df(body.readings)
    opt_df = pd.concat([_readings_to_df(body.context), df], ignore_index=True) if body.context else df
    recs   = await asyncio.to_thread(_run_optimisation, opt_df, 50, 100, body.top_k)
    chart  = _chart_pareto(recs) if MPL_AVAILABLE else None
    return {
        "chart_base64"     : chart,
        "n_recommendations": len(recs),
        "recommendations"  : recs,
    }


@router.post("/charts/rca", tags=["charts"])
async def chart_rca(body: RCARequest):
    """RCA waterfall chart for the current window."""
    _require_models()
    ctx  = _readings_to_df(body.context) if body.context else None
    df   = _readings_to_df(body.readings)
    full = pd.concat([ctx, df], ignore_index=True) if ctx is not None else df
    rca_result = await asyncio.to_thread(_run_rca, full, body.target)
    chart = _chart_rca_waterfall(rca_result) if MPL_AVAILABLE else None
    return {"chart_base64": chart, "rca": rca_result}


# ══════════════════════════════════════════════════════════════════════════════
# ── DASHBOARD ENDPOINTS ───────────────────────────────────────────════════════
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/dashboard", tags=["dashboard"])
async def dashboard():
    """
    Combined dashboard payload for the client frontend.
    Returns: model status, backtest performance, constraint specs, normal ranges.
    """
    return {
        "version"         : VERSION,
        "api_status"      : "ready" if _models_ready() else "loading",
        "models"          : {
            "surrogate_targets"  : TARGET_COLS,
            "controllable_inputs": CONTROLLABLE,
            "ensemble_components": ["LightGBM", "XGBoost", "CatBoost", "Ridge-meta"],
            "anomaly_models"     : ["LSTM-AE (3-seed ensemble)", "IsolationForest"],
            "optimizer"          : "Optuna TPE → CMA-ES",
            "explainability"     : "SHAP TreeExplainer",
        },
        "backtest"        : _perf_cache.get("backtest", {}),
        "constraints"     : CONSTRAINTS,
        "normal_ranges"   : {k: {"lo": v[0], "hi": v[1], "unit": v[2]} for k, v in NORMAL_RANGES.items()},
        "bounds"          : {k: {"lo": v[0], "hi": v[1]} for k, v in BOUNDS.items()},
        "conformal_adjustments": models.get("conformal", {}),
        "anomaly_threshold": _anomaly_threshold,
        "features_loaded" : len(models.get("feature_cols", [])),
    }


@router.get("/dashboard/performance", tags=["dashboard"])
async def dashboard_performance():
    """Model performance KPI card data for the frontend."""
    bt = _perf_cache.get("backtest", {})
    return {
        "surrogate_performance" : {
            "note": "Run V10 notebook to populate. Metrics loaded from backtest CSV.",
            **bt,
        },
        "model_components" : {
            "lgb_targets_loaded" : len(models.get("lgb", {})),
            "xgb_targets_loaded" : len(models.get("xgb", {})),
            "cat_targets_loaded" : len(models.get("cat", {})),
            "meta_targets_loaded": len(models.get("meta", {})),
            "lstm_ae_count"      : len(models.get("lstm_ae", [])),
            "iso_forest"         : models.get("iso_forest") is not None,
            "quantile_models"    : list(models.get("quantile", {}).keys()),
            "conformal"          : bool(models.get("conformal")),
        },
        "version": VERSION,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── WEBSOCKET — LIVE SENSOR STREAM ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@router.websocket("/ws/stream")
async def stream_live_data(websocket: WebSocket):
    """
    Streams rows from the test CSV one at a time, simulating a live sensor feed.
    For each row, runs prediction + anomaly detection and pushes the full result.

    Message types sent to client:
      {type: "status",     ...}   — connection info
      {type: "row",        ...}   — per-row prediction result
      {type: "alert",      ...}   — when anomaly detected
      {type: "complete",   ...}   — stream finished
      {type: "error",      ...}   — on error
    """
    await websocket.accept()
    logger.info("WebSocket client connected.")

    if not _models_ready():
        await websocket.send_json({"type": "error", "message": "Models not loaded."})
        await websocket.close()
        return

    if not CSV_STREAM_PATH.exists():
        await websocket.send_json({"type": "error",
                                   "message": f"Stream CSV not found: {CSV_STREAM_PATH}"})
        await websocket.close()
        return

    try:
        df_stream = pd.read_csv(str(CSV_STREAM_PATH))
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"CSV read failed: {exc}"})
        await websocket.close()
        return

    total = len(df_stream)
    await websocket.send_json({
        "type"         : "status",
        "message"      : f"Streaming {total} rows from {CSV_STREAM_PATH.name}",
        "total_rows"   : total,
        "delay_seconds": 2.0,
    })

    # Fill required columns
    for tgt in TARGET_COLS:
        if tgt not in df_stream.columns: df_stream[tgt] = 400.0
    if "EAF_Bath_Temp_C" not in df_stream.columns:
        df_stream["EAF_Bath_Temp_C"] = np.clip(1570+25*(df_stream["Electrode_Power_MW"]-70)/30, 1500, 1650)
    if "Bath_Carbon_Pct" not in df_stream.columns:
        df_stream["Bath_Carbon_Pct"] = np.clip(0.35-0.08*(df_stream["O2_Blow_Rate_Nm3h"]-2200)/1000, 0.04, 0.70)
    if "Slag_Basicity" not in df_stream.columns:
        df_stream["Slag_Basicity"] = np.clip(2.4+0.5*(df_stream["Lime_Addition_kg"]-3200)/1200, 1.6, 3.5)
    if "Tap_Temp_C" not in df_stream.columns:
        df_stream["Tap_Temp_C"] = np.clip(df_stream["EAF_Bath_Temp_C"]+15*(df_stream["LF_Heating_Power_MW"]-18)/10, 1560, 1640)
    for col, val in [("Hour_of_Day",12),("Day_of_Week",0),("Shift_ID",1),("Steel_Grade_ID",1),("Electrode_Campaign_Age_hrs",0.0)]:
        if col not in df_stream.columns: df_stream[col] = val

    scaler = models["scaler"]
    feat   = models["feature_cols"]

    WINDOW = 10  # rolling context window for feature engineering

    try:
        for i in range(len(df_stream)):
            win = df_stream.iloc[max(0, i - WINDOW + 1): i + 1].reset_index(drop=True)
            try:
                eng  = engineer_features(win)
                row  = eng.iloc[-1]
                x_raw = np.array(
                    [float(row.get(c, 0)) if pd.notna(row.get(c, 0)) else 0 for c in feat],
                    dtype=np.float32
                ).reshape(1, -1)
                x_sc = scaler.transform(x_raw)
                pred = {tgt: _predict_ensemble(x_sc, tgt) for tgt in TARGET_COLS}
                feasible, viols = _check_constraints(pred)
                anom = _anomaly_score_for_df(win)

                # Setpoint snapshot for frontend display
                setpoints_now = {c: round(float(df_stream.iloc[i].get(c, 0)), 3) for c in CONTROLLABLE if c in df_stream.columns}

                payload = {
                    "type"           : "row",
                    "row_index"      : i,
                    "timestamp"      : str(df_stream.iloc[i].get("Timestamp", i)),
                    "progress_pct"   : round(i / total * 100, 1),
                    "setpoints"      : setpoints_now,
                    "kpi_predictions": pred,
                    "feasible"       : feasible,
                    "violations"     : {k: round(v, 3) for k, v in viols.items() if v > 0},
                    "anomaly"        : anom,
                }
                await websocket.send_json(payload)

                # Send alert if anomaly detected
                if anom["flag"]:
                    alert_payload = {
                        "type"       : "alert",
                        "row_index"  : i,
                        "timestamp"  : str(df_stream.iloc[i].get("Timestamp", i)),
                        "alert_level": anom["alert_level"],
                        "score"      : anom["combined_score"],
                        "message"    : f"Anomaly detected at row {i} — score={anom['combined_score']:.3f}",
                        "top_deviations": [],
                    }
                    await websocket.send_json(alert_payload)

            except Exception as exc:
                logger.warning("Row %d inference error: %s", i, exc)

            await asyncio.sleep(2.0)  # 2-second interval simulates 5-min real plant cycle

        await websocket.send_json({
            "type"      : "complete",
            "message"   : "Stream finished.",
            "total_rows": total,
        })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(
        "main_2:app",
        host    = "0.0.0.0",
        port    = port,
        reload  = False,
        workers = 1,         # single worker — models are shared global state
    )