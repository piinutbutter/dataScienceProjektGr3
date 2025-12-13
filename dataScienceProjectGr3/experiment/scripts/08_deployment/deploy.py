"""
Lightweight deployment script für GRXEUR Trend Prediction.

- Lädt die letzten 5 Handelstage von 1-Minuten-Bars von yfinance (für Z-Normalisierung),
  dann beschränkt Entscheidungen auf die letzten 2 Tage der regulären Handelszeiten (RTH) 
  mit dem Alpaca Marktkalender.

- Berechnet TA-Features mit der gleichen generate_features() wie im Preprocessing.

- Erstellt 64-D Embeddings mit dem trainierten MLP (best_*_model_h15m.pt).

- Lädt den trainierten Decision Tree (oder Random Forest) und nutzt ihn für Klassifikation.
  Wenn die Vorhersage Klasse 1 (aufwärts) ist, wird ein Buy-Signal generiert.

- Für jedes Symbol: Bewertet das letzte verfügbare Embedding. Wenn der Baum eine 
  Vorhersage von 1 (aufwärts) macht, wird eine einfache Market Order (ohne SL/TP) 
  über Alpaca platziert.

- Abschließend: Prüft alle offenen Positionen; wenn ein Buy >= 30 Minuten her ist, 
  wird eine Market Sell Order zum Schließen der Position abgegeben.

Hinweise:
- Benötigt Packages: yfinance, requests, torch, sklearn, pandas, PyYAML.
- Konfiguration unter ../../conf/keys.yaml und ../../conf/params.yaml.
- Ticker: Standardmäßig GRXEUR, kann über Umgebungsvariable TICKERS geändert werden.
- Läuft einmal (one-shot). Kann extern periodisch geplant werden.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yaml
import pytz
import requests
import importlib.util
import pickle
import yfinance as yf
import torch
from torch import nn

# Lade Feature-Generator aus Preprocessing-Skripten über expliziten Pfad
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Projekt-Root: 3 Ebenen nach oben (von 08_deployment -> scripts -> experiment -> dataScienceProjectGr3)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
EXP_DIR = os.path.join(PROJECT_ROOT, "experiment")
CONF_DIR = os.path.join(EXP_DIR, "conf")
MODELS_DIR = os.path.join(EXP_DIR, "models")
DATA_DIR = os.path.join(EXP_DIR, "data")
FEATURES_PY_PATH = os.path.join(EXP_DIR, "scripts", "03_pre_split_prep", "features.py")

spec = importlib.util.spec_from_file_location("features_module", FEATURES_PY_PATH)
features_module = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader:
    spec.loader.exec_module(features_module)  # type: ignore[attr-defined]
else:
    raise RuntimeError(f"Konnte features.py nicht laden von {FEATURES_PY_PATH}")

generate_features = getattr(features_module, "generate_features")

# -----------------------------
# Konfiguration und Pfade
# -----------------------------
with open(os.path.join(CONF_DIR, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)

keys_path = os.path.join(CONF_DIR, "keys.yaml")
if os.path.exists(keys_path):
    with open(keys_path, "r") as f:
        keys = yaml.safe_load(f)
else:
    keys = {"KEYS": {}}

# Data Prep Params
ema_periods = params["DATA_PREP"]["EMA_PERIODS"]
slope_periods = params["DATA_PREP"]["SLOPE_PERIODS"]
z_norm_window = params["DATA_PREP"]["Z_NORM_WINDOW"]
feature_path = os.path.join(DATA_DIR, "processed", "features.txt")

# Modeling Params
hidden1 = params["MODELING"].get("HIDDEN1", 128)
hidden2 = params["MODELING"].get("HIDDEN2", 64)
dropout_p = params["MODELING"].get("DROPOUT", 0.1)
model_path_cfg = params["MODELING"].get("MODEL_PATH", "experiment/models")
# Konstruiere Pfad relativ zum Projekt-Root (wie in anderen Skripten)
model_path = os.path.abspath(os.path.join(PROJECT_ROOT, os.path.normpath(model_path_cfg)))

# Horizont für Modell (Standard: 15 Minuten)
HORIZON = int(os.getenv("HORIZON", "15"))

# Trading-Modus: 'live' (Alpaca), 'simulation' (nur Signale, kein Trading), oder 'backtest'
TRADING_MODE = os.getenv("TRADING_MODE", "simulation").lower()

# Alpaca Keys (Paper by default). Kann durch ALPACA_KEY_ID / ALPACA_SECRET env vars überschrieben werden.
# Nur relevant wenn TRADING_MODE='live'
ALPACA_KEY_ID = os.getenv("ALPACA_KEY_ID", keys["KEYS"].get("APCA-API-KEY-ID-Paper_v3") or keys["KEYS"].get("APCA-API-KEY-ID-Paper"))
ALPACA_SECRET = os.getenv("ALPACA_SECRET", keys["KEYS"].get("APCA-API-SECRET-KEY-Paper_v3") or keys["KEYS"].get("APCA-API-SECRET-KEY-Paper"))
ALPACA_BASE = os.getenv("ALPACA_BASE", "https://paper-api.alpaca.markets")

# Modell-Typ: 'decision_tree' oder 'random_forest'
MODEL_TYPE = os.getenv("MODEL_TYPE", "decision_tree")

# Ticker-Universum: env var TICKERS="GRXEUR" oder Standard
# PROXY_TICKER: Falls GRXEUR verwendet wird, welcher Ticker bei Alpaca gehandelt werden soll
# WICHTIG: Alpaca Paper Trading unterstützt hauptsächlich US-Märkte
# Empfohlene Proxy-Ticker: EWG (iShares MSCI Germany ETF), oder US-Aktien wie AAPL, MSFT
TICKERS_ENV = os.getenv("TICKERS")
PROXY_TICKER = os.getenv("PROXY_TICKER", "EWG")  # iShares MSCI Germany ETF (US-Notiert, DAX-korreliert)

if TICKERS_ENV:
    TICKERS = [t.strip().upper() for t in TICKERS_ENV.split(",") if t.strip()]
else:
    # Standard: GRXEUR mit lokalem Datensatz
    TICKERS = ["GRXEUR"]
    print(f"[init] Standard-Ticker: GRXEUR (lokale historische Daten)")
    print(f"[init] Proxy-Ticker für Trading: {PROXY_TICKER} (wird bei Alpaca gehandelt)")

# Bestimme ob lokale Daten verwendet werden sollen (muss nach TICKERS definiert werden)
USE_LOCAL_DATA = any(t.upper() == "GRXEUR" for t in TICKERS)

# Device-Auswahl für Torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Modell-Definitionen und Laden
# -----------------------------
class MLP(nn.Module):
    """MLP-Architektur passend zum Training-Skript (Sequential unter `net`).
    
    net.0: Linear(in_dim -> h1)
    net.1: ReLU
    net.2: Dropout
    net.3: Linear(h1 -> h2)
    net.4: Dropout
    net.5: Linear(h2 -> 1)
    
    Die embed() Methode gibt die Repräsentation nach net.4 zurück (Ausgabe der zweiten Hidden-Layer), Shape (N, h2).
    """
    
    def __init__(self, in_dim: int, h1: int, h2: int, dropout_p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h1, h2),
            nn.Dropout(dropout_p),
            nn.Linear(h2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        x = self.net[3](x)
        x = self.net[4](x)
        return x

# Lade Feature-Liste
FEATURES: List[str] = []
if feature_path and os.path.exists(feature_path):
    with open(feature_path, "r") as f:
        for line in f:
            feat = line.strip()
            if feat:
                FEATURES.append(feat)
else:
    raise RuntimeError(f"FEATURE_PATH nicht gefunden: {feature_path}")

IN_DIM = len(FEATURES)

# Lade MLP Checkpoint
ckpt_candidates = [
    os.path.join(model_path, f"best_acc_model_h{HORIZON}m.pt"),
    os.path.join(model_path, f"best_loss_model_h{HORIZON}m.pt"),
]

_ckpt = None
for c in ckpt_candidates:
    if os.path.exists(c):
        _ckpt = c
        break

if _ckpt is None:
    raise FileNotFoundError(f"Kein Model-Checkpoint gefunden in {model_path} (best_*_model_h{HORIZON}m.pt)")

print(f"[init] Lade MLP-Checkpoint: {_ckpt}")
# Versuche kompatibles Laden: PyTorch 2.6+ verwendet weights_only=True per Default.
# Wenn das Checkpoint aus einer vertrauenswürdigen Quelle stammt, explizit weights_only=False nutzen.
try:
    ckpt = torch.load(_ckpt, map_location=DEVICE, weights_only=False)
except TypeError:
    # Ältere Torch-Versionen kennen das Argument nicht
    ckpt = torch.load(_ckpt, map_location=DEVICE)
except Exception:
    # Fallback: allowlist für spezifische numpy-Objekte (nur wenn Checkpoint vertraut)
    try:
        if hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals([np._core.multiarray.scalar]):
                ckpt = torch.load(_ckpt, map_location=DEVICE)
        elif hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([np._core.multiarray.scalar])
            ckpt = torch.load(_ckpt, map_location=DEVICE)
        else:
            raise
    except Exception:
        raise

model = MLP(IN_DIM, hidden1, hidden2, dropout_p).to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Lade Decision Tree oder Random Forest
if MODEL_TYPE == "decision_tree":
    tree_path = os.path.join(model_path, f"decision_tree_h{HORIZON}m.pkl")
elif MODEL_TYPE == "random_forest":
    tree_path = os.path.join(model_path, f"random_forest_h{HORIZON}m.pkl")
else:
    raise ValueError(f"Unbekannter MODEL_TYPE: {MODEL_TYPE}. Muss 'decision_tree' oder 'random_forest' sein.")

if not os.path.exists(tree_path):
    raise FileNotFoundError(f"Modell-Datei nicht gefunden: {tree_path}")

print(f"[init] Lade {MODEL_TYPE}: {tree_path}")
with open(tree_path, "rb") as f:
    bundle = pickle.load(f)

clf = bundle["model"]
tree_feature_cols = bundle.get("feature_cols", FEATURES)

# Stelle sicher, dass Feature-Reihenfolge übereinstimmt
if list(tree_feature_cols) != list(FEATURES):
    print(f"[WARN] Feature-Reihenfolge stimmt nicht überein. Verwende Reihenfolge aus Modell: {len(tree_feature_cols)} Features")

print(f"[init] Model-Typ: {MODEL_TYPE}")
print(f"[init] Features: {IN_DIM}")
print(f"[init] Horizon: {HORIZON}m")
print(f"[init] Ticker: {TICKERS}")
print(f"[init] Trading-Modus: {TRADING_MODE}")
if TRADING_MODE == "simulation":
    print(f"[init] Simulation-Modus: Signale werden generiert, aber keine Orders platziert")
elif TRADING_MODE == "live":
    if not ALPACA_KEY_ID or not ALPACA_SECRET:
        print(f"[WARN] Trading-Modus ist 'live', aber Alpaca-Keys fehlen!")

# -----------------------------
# Helfer: Alpaca RTH Kalender und Trading über REST
# -----------------------------
EASTERN = pytz.timezone("US/Eastern")

def alpaca_headers() -> Dict[str, str]:
    if not ALPACA_KEY_ID or not ALPACA_SECRET:
        raise RuntimeError("Alpaca API-Keys fehlen. Setze env ALPACA_KEY_ID/ALPACA_SECRET oder fülle keys.yaml.")
    return {
        "APCA-API-KEY-ID": ALPACA_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def build_calendar_map(start_dt: datetime, end_dt: datetime) -> Dict[datetime.date, Tuple[datetime, datetime]]:
    """Baut eine Karte von Handelsdatum -> (Open, Close) RTH Zeiten."""
    # Alpaca calendar v2: /v2/calendar?start=YYYY-MM-DD&end=YYYY-MM-DD
    url = f"{ALPACA_BASE}/v2/calendar"
    params_q = {
        "start": start_dt.strftime("%Y-%m-%d"),
        "end": end_dt.strftime("%Y-%m-%d"),
    }
    r = requests.get(url, headers=alpaca_headers(), params=params_q, timeout=30)
    r.raise_for_status()
    days = r.json()
    
    cal_map: Dict[datetime.date, Tuple[datetime, datetime]] = {}
    for d in days:
        # d: {'date': '2025-09-24', 'open': '09:30', 'close': '16:00', ...}
        date_str = d.get("date")
        open_str = d.get("open")
        close_str = d.get("close")
        if not date_str or not open_str or not close_str:
            continue
        y, m, dd = map(int, date_str.split("-"))
        oh, om = map(int, open_str.split(":"))
        ch, cm = map(int, close_str.split(":"))
        open_dt = EASTERN.localize(datetime(y, m, dd, oh, om))
        close_dt = EASTERN.localize(datetime(y, m, dd, ch, cm))
        cal_map[open_dt.date()] = (open_dt, close_dt)
    
    return cal_map

def is_rth(ts: pd.Timestamp, cal_map: Dict[datetime.date, Tuple[datetime, datetime]]) -> bool:
    """Prüft ob Timestamp in Regular Trading Hours liegt."""
    if ts.tzinfo is None:
        # Nehme UTC an wenn tz-naive
        ts_eastern = ts.tz_localize("UTC").astimezone(EASTERN)
    else:
        try:
            ts_eastern = ts.tz_convert(EASTERN)  # type: ignore[attr-defined]
        except Exception:
            ts_eastern = ts.tz_localize("UTC").astimezone(EASTERN)
    
    d = ts_eastern.date()
    if d not in cal_map:
        return False
    
    open_dt, close_dt = cal_map[d]
    return open_dt <= ts_eastern < close_dt

def get_positions() -> List[dict]:
    """Holt alle offenen Positionen."""
    url = f"{ALPACA_BASE}/v2/positions"
    r = requests.get(url, headers=alpaca_headers(), timeout=30)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return r.json()

def get_filled_orders_for_symbol(symbol: str, limit: int = 50) -> List[dict]:
    """Holt gefüllte Orders für ein Symbol."""
    # /v2/orders?status=closed enthält gefüllte Orders
    url = f"{ALPACA_BASE}/v2/orders"
    params_q = {
        "status": "closed",
        "limit": str(limit),
        "nested": "false",
        "direction": "desc",
    }
    r = requests.get(url, headers=alpaca_headers(), params=params_q, timeout=30)
    r.raise_for_status()
    orders = r.json()
    
    # Filtere gefüllte und nach Symbol
    out = []
    for o in orders:
        if str(o.get("status", "")).lower() != "filled":
            continue
        if str(o.get("symbol", "")).upper() != symbol.upper():
            continue
        out.append(o)
    
    return out

def submit_market_order(symbol: str, side: str, qty: int = 1) -> dict | None:
    """Gibt eine Market Order ab (oder simuliert im Simulation-Modus)."""
    # Im Simulation-Modus keine echten Orders
    if TRADING_MODE == "simulation":
        order_id = f"SIM-{int(time.time())}"
        print(f"[sim] {side.upper()} {qty} {symbol}: SIMULIERT (Order ID: {order_id})")
        return {"id": order_id, "status": "simulated", "symbol": symbol, "side": side, "qty": qty}
    
    # Live Trading mit Alpaca
    url = f"{ALPACA_BASE}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side.lower(),
        "type": "market",
        "time_in_force": "day",
    }
    try:
        r = requests.post(url, headers=alpaca_headers(), json=payload, timeout=30)
        r.raise_for_status()
        od = r.json()
        print(f"[order] {side.upper()} {qty} {symbol}: submitted id={od.get('id')}")
        return od
    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        # Versuche mehr Details aus der Antwort zu bekommen
        try:
            if hasattr(e.response, 'json'):
                error_details = e.response.json()
                error_msg = f"{error_msg}: {error_details}"
        except Exception:
            pass
        print(f"[order] {side.upper()} {symbol} fehlgeschlagen: {error_msg}")
        if "422" in error_msg or "Unprocessable Entity" in error_msg:
            print(f"[order] Hinweis: Symbol '{symbol}' ist möglicherweise bei Alpaca nicht handelbar.")
            print(f"[order] Alpaca Paper Trading unterstützt hauptsächlich US-Märkte.")
            print(f"[order] Versuche einen US-Proxy-Ticker: export PROXY_TICKER='EWG' (Germany ETF)")
            print(f"[order] Oder US-Aktien: export PROXY_TICKER='AAPL' / 'MSFT' / 'GOOGL'")
        return None
    except Exception as e:
        print(f"[order] {side.upper()} {symbol} fehlgeschlagen: {e}")
        return None

def close_positions_older_than_30m():
    """Prüft offene Positionen und verkauft die, deren letzter BUY-Fill >= 30 Minuten her ist."""
    if TRADING_MODE == "simulation":
        print("[sim] Position-Management übersprungen (Simulation-Modus)")
        return
    
    try:
        positions = get_positions()
    except Exception as e:
        print(f"[pos] Kann Positionen nicht abrufen: {e}")
        return
    
    if not positions:
        print("[pos] Keine offenen Positionen.")
        return
    
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(minutes=30)
    
    for p in positions:
        symbol = p.get("symbol") or p.get("asset_symbol")
        if not symbol:
            continue
        
        try:
            qty_str = p.get("qty") or p.get("quantity")
            qty = int(float(qty_str)) if qty_str is not None else None
        except Exception:
            qty = None
        
        if not qty:
            continue
        
        # Finde letzte gefüllte BUY-Order für dieses Symbol
        try:
            orders = get_filled_orders_for_symbol(symbol, limit=50)
        except Exception as e:
            print(f"[pos] get_orders fehlgeschlagen für {symbol}: {e}")
            continue
        
        last_buy_fill = None
        for o in orders:
            side = str(o.get("side", "")).lower()
            if side != "buy":
                continue
            
            filled_at = o.get("filled_at")
            if not filled_at:
                continue
            
            try:
                # ISO8601 mit Z
                dt = datetime.fromisoformat(str(filled_at).replace("Z", "+00:00"))
            except Exception:
                continue
            
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            
            if (last_buy_fill is None) or (dt > last_buy_fill):
                last_buy_fill = dt
        
        if last_buy_fill and last_buy_fill <= cutoff:
            print(f"[pos] Schließe {symbol}: letzter BUY-Fill bei {last_buy_fill.isoformat()} (<= {cutoff.isoformat()})")
            submit_market_order(symbol, side="sell", qty=qty)
        else:
            if last_buy_fill:
                print(f"[pos] Behalte {symbol}: letzter BUY-Fill bei {last_buy_fill}")

# -----------------------------
# Daten-Akquisition: Lokale GRXEUR-Daten oder yfinance
# -----------------------------
def load_local_grxeur_data(days_hist: int = 5) -> pd.DataFrame | None:
    """Lädt lokale GRXEUR-Daten aus Parquet-Dateien.
    
    Lädt die letzten `days_hist` Tage aus dem historischen Datensatz (2010-2018).
    Da es historische Daten sind, werden die letzten verfügbaren Tage geladen.
    """
    grxeur_parquet = os.path.join(DATA_DIR, "raw", "Bars_1m_GRXEUR", "GRXEUR_M1_2010_2018.parquet")
    
    if not os.path.exists(grxeur_parquet):
        print(f"[local] GRXEUR-Datei nicht gefunden: {grxeur_parquet}")
        return None
    
    try:
        print(f"[local] Lade lokale GRXEUR-Daten aus {grxeur_parquet}...")
        df = pd.read_parquet(grxeur_parquet)
        
        # Stelle sicher, dass Index DatetimeIndex ist
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime").sort_index()
        
        # Hole die letzten verfügbaren Tage
        cutoff_date = df.index[-1] - pd.Timedelta(days=days_hist)
        df_recent = df[df.index >= cutoff_date].copy()
        
        # Stelle sicher, dass Index tz-aware UTC ist (für Kompatibilität)
        if df_recent.index.tz is None:
            df_recent.index = df_recent.index.tz_localize("UTC")
        
        # Normalisiere Spaltennamen zu lowercase
        df_recent.columns = df_recent.columns.str.lower()
        
        print(f"[local] GRXEUR: {len(df_recent)} Zeilen geladen (von {df_recent.index[0]} bis {df_recent.index[-1]})")
        return df_recent
        
    except Exception as e:
        print(f"[local] Fehler beim Laden von GRXEUR-Daten: {e}")
        return None

def download_minute_data(tickers: List[str], days_hist: int = 5) -> Dict[str, pd.DataFrame]:
    """Lädt 1m Bars für die letzten `days_hist` Tage für die gegebenen Ticker.
    
    Gibt ein Dict ticker -> DataFrame mit UTC tz-aware DatetimeIndex und Spalten:
    Open, High, Low, Close, Adj Close, Volume
    
    Hinweis: GRXEUR ist ein historischer Datensatz (2010-2018) und nicht bei yfinance verfügbar.
    Für Live-Trading sollte ein ähnlicher Index verwendet werden (z.B. ^GDAXI für DAX).
    """
    print(f"[yf] Lade {days_hist}d von 1m Daten für {len(tickers)} Ticker...")
    data: Dict[str, pd.DataFrame] = {}
    
    # yfinance unterstützt Multi-Ticker-Download, aber zur Vermeidung von Rate-Limits iterieren wir
    for t in tickers:
        # Prüfe ob lokale Daten verwendet werden sollen
        if t.upper() == "GRXEUR" and USE_LOCAL_DATA:
            df = load_local_grxeur_data(days_hist=days_hist)
            if df is not None and not df.empty:
                data[t] = df
                continue
            else:
                print(f"[yf] {t}: Lokale Daten nicht verfügbar, versuche yfinance...")
        
        try:
            # Versuche yfinance Download
            df = yf.download(t, period=f"{days_hist}d", interval="1m", auto_adjust=True, prepost=True, progress=False)
            
            if df is None or df.empty:
                print(f"[yf] {t}: Keine Daten verfügbar bei yfinance")
                continue
            
            # Stelle sicher, dass Index tz-aware UTC ist
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            
            # Normalisiere Spaltennamen (yfinance gibt manchmal MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Stelle sicher, dass wir die notwendigen Spalten haben
            required_cols = ["Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_cols):
                print(f"[yf] {t}: Fehlende Spalten. Gefunden: {list(df.columns)}")
                continue
            
            # Normalisiere zu lowercase für Kompatibilität
            df.columns = df.columns.str.lower()
            
            data[t] = df
            print(f"[yf] {t}: Erfolgreich geladen, {len(df)} Zeilen")
            time.sleep(0.2)  # kleiner Backoff
        except Exception as e:
            print(f"[yf] {t}: Unerwarteter Fehler {e}")
    
    return data

# -----------------------------
# Feature- und Embedding-Berechnung
# -----------------------------
def compute_latest_embedding(df_raw: pd.DataFrame) -> Tuple[pd.Timestamp | None, np.ndarray | None]:
    """Gegeben ein rohes 1m OHLCV DataFrame (UTC Index),
    - erstellt VWAP-Approximation
    - berechnet Features mit Trainings-Parametern
    - aligniert zu FEATURES, füllt fehlende mit 0
    - gibt (last_ts, embedding_vector) zurück, wobei last_ts der Timestamp der letzten verwendeten Zeile ist
    """
    if df_raw is None or df_raw.empty:
        return None, None
    
    df = df_raw.copy()
    
    # VWAP-Approximation über typical price
    # Für GRXEUR nutzen wir einfach Close, da wir kein Volume haben
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    else:
        df["vwap"] = df["close"]
    
    # Stelle sicher, dass wir die notwendigen Spalten haben
    df_features = df[["close", "open", "high", "low"]].copy()
    df_features["vwap"] = df["vwap"]
    
    # Berechne Features
    try:
        df_feat, _ = generate_features(
            df=df_features,
            ema_periods=ema_periods,
            slope_periods=slope_periods,
            z_norm_window=z_norm_window,
            price_col="close",
            volume_col=None
        )
    except Exception as e:
        print(f"[feat] Feature-Generierung fehlgeschlagen: {e}")
        return None, None
    
    # Baue Feature-Frame und aligniere zu Trainings-FEATURES-Liste
    # Einige Features können in Live-Daten fehlen aufgrund von Rolling-Windows; fülle mit 0
    X = pd.DataFrame(index=df_feat.index)
    for col in FEATURES:
        if col in df_feat.columns:
            X[col] = df_feat[col]
        else:
            X[col] = 0.0
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Nimm die letzte Timestamp-Zeile
    if X.empty:
        return None, None
    
    last_ts = X.index[-1]
    
    # Stelle sicher, dass Features in der richtigen Reihenfolge sind
    if list(tree_feature_cols) != list(FEATURES):
        x = X[tree_feature_cols].iloc[[-1]].astype(np.float32).values
    else:
        x = X.iloc[[-1]].astype(np.float32).values
    
    with torch.no_grad():
        emb = model.embed(torch.tensor(x, dtype=torch.float32, device=DEVICE)).cpu().numpy()[0]
    
    return last_ts, emb

# -----------------------------
# Entry-Regel über Decision Tree / Random Forest
# -----------------------------
def predict_from_features(x_features: np.ndarray) -> bool:
    """Nutzt den Decision Tree / Random Forest für Vorhersage basierend auf Features."""
    # Vorhersage: 1 = aufwärts (Buy), 0 = abwärts/flach (No Buy)
    pred = clf.predict(x_features)
    # pred ist ein Array, nimm den ersten Wert
    if isinstance(pred, np.ndarray):
        pred_value = pred[0]
    else:
        pred_value = pred
    
    # Klasse 1 bedeutet aufwärts (Buy-Signal)
    return int(pred_value) == 1

# -----------------------------
# Hauptfluss
# -----------------------------
def main():
    # Baue Kalender-Map für RTH-Filterung, abdeckend die letzten ~10 Kalendertage
    end_dt = datetime.now(tz=EASTERN)
    start_dt = end_dt - timedelta(days=10)
    
    try:
        cal_map = build_calendar_map(start_dt=start_dt, end_dt=end_dt)
    except Exception as e:
        print(f"[WARN] Kalender konnte nicht geladen werden: {e}. Nutze keine RTH-Filterung.")
        cal_map = {}
    
    # Lade 5 Tage von 1m Bars für genug Kontext; wir werden nur auf die letzten 2 Tage handeln
    data = download_minute_data(TICKERS, days_hist=5)
    
    if not data:
        print("[ERROR] Keine Daten geladen. Beende.")
        print("[ERROR] Hinweise:")
        print("[ERROR] - GRXEUR ist ein historischer Datensatz (2010-2018) und nicht bei yfinance verfügbar")
        print("[ERROR] - Für Live-Trading verwende einen ähnlichen Index:")
        print("[ERROR]   export TICKERS='^GDAXI'  # DAX Index")
        print("[ERROR]   export TICKERS='EXS1.DE'  # DAX ETF")
        print("[ERROR]   python experiment/scripts/08_deployment/deploy.py")
        return
    
    # Beschränke Entscheidungen auf die letzten 2 Tage
    decision_cutoff_utc = datetime.now(timezone.utc) - timedelta(days=2)
    
    actions = []
    
    for sym, df in data.items():
        # Filtere auf RTH mit Alpaca-Kalender
        if df is None or df.empty:
            continue
        
        # Prüfe ob es historische Daten sind (älter als 1 Jahr)
        is_historical = sym.upper() == "GRXEUR" or (df.index[-1] < datetime.now(timezone.utc) - timedelta(days=365))
        
        # Behalte nur Zeilen innerhalb RTH (nur für Live-Daten)
        if cal_map and not is_historical:
            mask_rth = df.index.to_series().map(lambda ts: is_rth(ts, cal_map))
            df_rth = df.loc[mask_rth]
            if df_rth.empty:
                print(f"[eval] {sym}: keine RTH-Zeilen")
                continue
        else:
            # Für historische Daten: Nutze alle Daten oder simuliere RTH (9:30-16:00 ET)
            if is_historical:
                print(f"[eval] {sym}: Historische Daten erkannt, verwende alle verfügbaren Zeilen")
                # Optional: Filtere auf typische Handelszeiten (9:30-16:00 ET) auch für historische Daten
                try:
                    # Konvertiere zu Eastern Time für RTH-Filterung
                    df_eastern = df.index.tz_convert(EASTERN) if df.index.tz else df.index.tz_localize("UTC").tz_convert(EASTERN)
                    # Filtere auf 9:30-16:00 ET (ungefähr)
                    hour_mask = (df_eastern.hour >= 9) & ((df_eastern.hour < 16) | ((df_eastern.hour == 16) & (df_eastern.minute == 0)))
                    df_rth = df.loc[hour_mask]
                    if df_rth.empty:
                        print(f"[eval] {sym}: Keine Zeilen in typischen Handelszeiten (9:30-16:00 ET)")
                        continue
                except Exception as e:
                    print(f"[eval] {sym}: Fehler bei RTH-Filterung für historische Daten: {e}, verwende alle Daten")
                    df_rth = df.copy()
            else:
                df_rth = df.copy()
        
        if df_rth.empty:
            print(f"[eval] {sym}: keine Daten nach Filterung")
            continue
        
        # Wende 2-Tage-Entscheidungsfenster an (nur für Live-Daten)
        if not is_historical:
            df_rth = df_rth[df_rth.index >= decision_cutoff_utc]
            if df_rth.empty:
                print(f"[eval] {sym}: keine Zeilen innerhalb der letzten 2 Tage")
                continue
        else:
            # Für historische Daten: Nutze die letzten verfügbaren Tage
            hist_cutoff = df_rth.index[-1] - timedelta(days=2)
            df_rth = df_rth[df_rth.index >= hist_cutoff]
            if df_rth.empty:
                print(f"[eval] {sym}: keine Zeilen in den letzten 2 verfügbaren Tagen")
                continue
            print(f"[eval] {sym}: Verwende historische Daten von {df_rth.index[0]} bis {df_rth.index[-1]}")
        
        # Berechne Features (nicht Embedding direkt)
        if df_rth is None or df_rth.empty:
            continue
        
        df_features_input = df_rth[["close", "open", "high", "low"]].copy()
        if "high" in df_rth.columns and "low" in df_rth.columns and "close" in df_rth.columns:
            df_features_input["vwap"] = (df_rth["high"] + df_rth["low"] + df_rth["close"]) / 3.0
        else:
            df_features_input["vwap"] = df_rth["close"]
        
        try:
            df_feat, _ = generate_features(
                df=df_features_input,
                ema_periods=ema_periods,
                slope_periods=slope_periods,
                z_norm_window=z_norm_window,
                price_col="close",
                volume_col=None
            )
        except Exception as e:
            print(f"[eval] {sym}: Feature-Generierung fehlgeschlagen: {e}")
            continue
        
        # Baue Feature-Frame und aligniere zu Trainings-FEATURES-Liste
        X = pd.DataFrame(index=df_feat.index)
        for col in FEATURES:
            if col in df_feat.columns:
                X[col] = df_feat[col]
            else:
                X[col] = 0.0
        
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        if X.empty:
            print(f"[eval] {sym}: konnte Features nicht berechnen")
            continue
        
        # Hole Timestamp: generate_features() setzt Index zurück und speichert Timestamp in Spalte
        if "timestamp" in df_feat.columns:
            last_ts = df_feat["timestamp"].iloc[-1]
        elif isinstance(df_feat.index, pd.DatetimeIndex):
            last_ts = df_feat.index[-1]
        else:
            # Fallback: Verwende Original-Daten Timestamp
            last_ts = df_rth.index[-1]
        
        # Stelle sicher, dass Features in der richtigen Reihenfolge sind
        if list(tree_feature_cols) != list(FEATURES):
            x_features = X[tree_feature_cols].iloc[[-1]].astype(np.float32).values
        else:
            x_features = X.iloc[[-1]].astype(np.float32).values
        
        # Vorhersage mit Decision Tree / Random Forest
        try:
            should_buy_signal = predict_from_features(x_features)
        except Exception as e:
            print(f"[eval] {sym}: Vorhersage fehlgeschlagen: {e}")
            continue
        
        # Formatiere Timestamp für lesbare Ausgabe
        if isinstance(last_ts, pd.Timestamp):
            ts_str = last_ts.strftime("%Y-%m-%d %H:%M:%S %Z")
        elif hasattr(last_ts, 'strftime'):
            ts_str = last_ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            ts_str = str(last_ts)
        
        if should_buy_signal:
            print(f"[signal] {sym} @ {ts_str}: Vorhersage = 1 (aufwärts) -> BUY")
            
            # Im Simulation-Modus: Kein Proxy-Ticker nötig, da kein echtes Trading
            if TRADING_MODE == "simulation":
                trading_symbol = sym  # Verwende Original-Symbol
                print(f"[signal] Simulation: BUY-Signal für {sym} (kein echtes Trading)")
            else:
                # Verwende Proxy-Ticker für Trading, wenn GRXEUR verwendet wird
                trading_symbol = PROXY_TICKER if sym.upper() == "GRXEUR" else sym
                if trading_symbol != sym:
                    print(f"[signal] Verwende Proxy-Ticker {trading_symbol} für Trading (GRXEUR nicht handelbar bei Alpaca)")
            
            order = submit_market_order(trading_symbol, side="buy", qty=1)
            actions.append((sym, ts_str, "BUY", order.get("id") if isinstance(order, dict) else None, trading_symbol))
        else:
            print(f"[signal] {sym} @ {ts_str}: Vorhersage = 0 (kein Signal)")
    
    # Post-Trade-Management: Schließe Positionen >= 30 Minuten alt
    close_positions_older_than_30m()
    
    print("[done] Aktionen:")
    if actions:
        for action_item in actions:
            if len(action_item) == 5:  # Mit Proxy-Ticker
                sym, ts, action, order_id, trading_sym = action_item
                if trading_sym != sym:
                    print(f"  {action} {sym} @ {ts} -> Trading: {trading_sym} (Order ID: {order_id if order_id else 'N/A'})")
                else:
                    print(f"  {action} {sym} @ {ts} (Order ID: {order_id if order_id else 'N/A'})")
            else:  # Alte Format
                sym, ts, action, order_id = action_item[:4]
                print(f"  {action} {sym} @ {ts} (Order ID: {order_id if order_id else 'N/A'})")
    else:
        print("  Keine Aktionen")

if __name__ == "__main__":
    main()

