"""
Paper Trading Performance-Analyse für GRXEUR Trend Prediction.

Dieses Skript:
- Lädt Live-Daten von Yahoo Finance (z.B. ^GDAXI als Proxy für GRXEUR)
- Generiert Trading-Signale basierend auf dem trainierten Modell
- Simuliert Paper Trading (ohne echte Orders)
- Analysiert Performance über verschiedene Zeitrahmen
- Vergleicht Ergebnisse mit Backtest-Ergebnissen
- Erstellt detaillierte Analysen und Visualisierungen

VARIANTEN:
- Variante 1: Backup in paper_trading_analysis_variante1.py
  (Stand: Exit-Prüfung auf allen Preis-Datenpunkten, 15 Min Exit, 2 Min Mindest-Haltedauer)
"""

from __future__ import annotations

import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pytz
import torch
from torch import nn
import importlib.util
import yfinance as yf

# Pfad-Setup
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
EXP_DIR = os.path.join(PROJECT_ROOT, "experiment")
CONF_DIR = os.path.join(EXP_DIR, "conf")
MODELS_DIR = os.path.join(EXP_DIR, "models")
DATA_DIR = os.path.join(EXP_DIR, "data")
PLOTS_DIR = os.path.join(EXP_DIR, "plots")
FEATURES_PY_PATH = os.path.join(EXP_DIR, "scripts", "03_pre_split_prep", "features.py")

# Lade Feature-Generator
spec = importlib.util.spec_from_file_location("features_module", FEATURES_PY_PATH)
features_module = importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader:
    spec.loader.exec_module(features_module)
else:
    raise RuntimeError(f"Konnte features.py nicht laden von {FEATURES_PY_PATH}")

generate_features = getattr(features_module, "generate_features")

# Konfiguration
with open(os.path.join(CONF_DIR, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)

ema_periods = params["DATA_PREP"]["EMA_PERIODS"]
slope_periods = params["DATA_PREP"]["SLOPE_PERIODS"]
z_norm_window = params["DATA_PREP"]["Z_NORM_WINDOW"]
feature_path = os.path.join(DATA_DIR, "processed", "features.txt")
model_path_cfg = params["MODELING"].get("MODEL_PATH", "experiment/models")
model_path = os.path.abspath(os.path.join(PROJECT_ROOT, os.path.normpath(model_path_cfg)))
HORIZON = int(os.getenv("HORIZON", "15"))
MODEL_TYPE = os.getenv("MODEL_TYPE", "decision_tree")

# Ticker-Konfiguration
TICKERS_ENV = os.getenv("TICKERS")
if TICKERS_ENV:
    TICKERS = [t.strip().upper() for t in TICKERS_ENV.split(",") if t.strip()]
else:
    # Standard: DAX Index als Proxy für GRXEUR
    TICKERS = ["^GDAXI"]

# Lade Features
FEATURES: List[str] = []
if feature_path and os.path.exists(feature_path):
    with open(feature_path, "r") as f:
        for line in f:
            feat = line.strip()
            if feat:
                FEATURES.append(feat)

IN_DIM = len(FEATURES)
hidden1 = params["MODELING"].get("HIDDEN1", 128)
hidden2 = params["MODELING"].get("HIDDEN2", 64)
dropout_p = params["MODELING"].get("DROPOUT", 0.1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Modell-Laden
# -----------------------------
class MLP(nn.Module):
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

# Lade MLP (optional)
try:
    ckpt_path = os.path.join(model_path, f"best_acc_model_h{HORIZON}m.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = MLP(IN_DIM, hidden1, hidden2, dropout_p).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"[init] MLP geladen: {ckpt_path}")
    else:
        model = None
except Exception as e:
    print(f"[init] MLP konnte nicht geladen werden: {e}")
    model = None

# Lade Decision Tree / Random Forest
tree_path = os.path.join(model_path, f"{MODEL_TYPE}_h{HORIZON}m.pkl")
if not os.path.exists(tree_path):
    raise FileNotFoundError(f"Modell nicht gefunden: {tree_path}")

with open(tree_path, "rb") as f:
    bundle = pickle.load(f)

clf = bundle["model"]
tree_feature_cols = bundle.get("feature_cols", FEATURES)
print(f"[init] {MODEL_TYPE} geladen: {tree_path}")
print(f"[init] Ticker: {TICKERS}")

# -----------------------------
# Daten-Laden (Live von Yahoo Finance)
# -----------------------------
EASTERN = pytz.timezone("US/Eastern")

def load_live_data(tickers: List[str], days: int = 7) -> Dict[str, pd.DataFrame]:
    """Lädt Live-Daten von Yahoo Finance.
    
    Hinweis: Yahoo Finance erlaubt nur max. 7 Tage von 1m Daten pro Request.
    """
    # Yahoo Finance Limit: max. 7 Tage für 1m Daten
    max_days_per_request = 7
    actual_days = min(days, max_days_per_request)
    
    if days > max_days_per_request:
        print(f"[data] Hinweis: Yahoo Finance erlaubt nur {max_days_per_request} Tage 1m Daten pro Request.")
        print(f"[data] Verwende {actual_days} Tage (für mehr Tage müsste man mehrere Requests machen).")
    
    print(f"[data] Lade {actual_days} Tage Live-Daten von Yahoo Finance für {tickers}...")
    data = {}
    
    for ticker in tickers:
        try:
            # Verwende period für bessere Kompatibilität
            df = yf.download(ticker, period=f"{actual_days}d", interval="1m", 
                           auto_adjust=True, prepost=True, progress=False)
            
            if df is None or df.empty:
                print(f"[data] {ticker}: Keine Daten verfügbar")
                continue
            
            # Normalisiere Spaltennamen
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            required_cols = ["Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_cols):
                print(f"[data] {ticker}: Fehlende Spalten")
                continue
            
            # TZ-aware UTC Index
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            
            df.columns = df.columns.str.lower()
            data[ticker] = df
            print(f"[data] {ticker}: {len(df)} Zeilen geladen ({df.index[0]} bis {df.index[-1]})")
            
        except Exception as e:
            print(f"[data] {ticker}: Fehler beim Laden - {e}")
    
    return data

# -----------------------------
# Feature-Berechnung und Signal-Generierung
# -----------------------------
def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet Features und generiert Trading-Signale."""
    print(f"[features] Berechne Features und Signale...")
    
    # Bereite Daten vor
    df_features_input = df[["close", "open", "high", "low"]].copy()
    if "high" in df.columns and "low" in df.columns:
        df_features_input["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    else:
        df_features_input["vwap"] = df["close"]
    
    # Unterdrücke print-Ausgabe
    import sys
    from io import StringIO
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        df_feat, _ = generate_features(
            df=df_features_input,
            ema_periods=ema_periods,
            slope_periods=slope_periods,
            z_norm_window=z_norm_window,
            price_col="close",
            volume_col=None
        )
    finally:
        sys.stdout = old_stdout
    
    # Feature-Frame bauen
    X = pd.DataFrame(index=df_feat.index)
    for col in FEATURES:
        if col in df_feat.columns:
            X[col] = df_feat[col]
        else:
            X[col] = 0.0
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    if X.empty:
        return pd.DataFrame()
    
    # Merge Features mit Preisen, um sicherzustellen, dass Indizes übereinstimmen
    # Verwende merge_asof für bessere Synchronisation
    X_reset = X.reset_index()
    # Stelle sicher, dass die erste Spalte "timestamp" heißt und datetime ist
    if X_reset.columns[0] != "timestamp":
        X_reset = X_reset.rename(columns={X_reset.columns[0]: "timestamp"})
    
    # Konvertiere zu datetime und stelle sicher, dass Timezone konsistent ist
    X_reset["timestamp"] = pd.to_datetime(X_reset["timestamp"])
    if X_reset["timestamp"].dt.tz is None:
        # Wenn naive, konvertiere zu UTC (wie df)
        X_reset["timestamp"] = X_reset["timestamp"].dt.tz_localize("UTC")
    else:
        # Wenn timezone-aware, konvertiere zu UTC
        X_reset["timestamp"] = X_reset["timestamp"].dt.tz_convert("UTC")
    
    df_reset = df[["close"]].reset_index()
    if df_reset.columns[0] != "timestamp":
        df_reset = df_reset.rename(columns={df_reset.columns[0]: "timestamp"})
    
    # Stelle sicher, dass df auch UTC timezone-aware ist
    df_reset["timestamp"] = pd.to_datetime(df_reset["timestamp"])
    if df_reset["timestamp"].dt.tz is None:
        df_reset["timestamp"] = df_reset["timestamp"].dt.tz_localize("UTC")
    else:
        df_reset["timestamp"] = df_reset["timestamp"].dt.tz_convert("UTC")
    
    # Sortiere beide DataFrames für merge_asof
    X_reset = X_reset.sort_values("timestamp")
    df_reset = df_reset.sort_values("timestamp")
    
    # Merge für Preis-Zuordnung
    merged = pd.merge_asof(
        X_reset,
        df_reset,
        on="timestamp",
        direction="nearest"
    ).set_index("timestamp")
    
    # Vorhersagen für alle Zeilen
    results = []
    if list(tree_feature_cols) != list(FEATURES):
        x_features = merged[tree_feature_cols].astype(np.float32).values
    else:
        # Nur Feature-Spalten, nicht "close"
        feature_cols = [col for col in merged.columns if col != "close"]
        x_features = merged[feature_cols].astype(np.float32).values
    
    # Batch-Vorhersage
    predictions = clf.predict(x_features)
    
    # Erstelle Ergebnisse mit synchronisierten Preisen
    for timestamp, (pred, price) in zip(merged.index, zip(predictions, merged["close"])):
        pred_value = int(pred) if isinstance(pred, np.ndarray) else int(pred)
        
        results.append({
            "timestamp": timestamp,
            "price": float(price),
            "prediction": pred_value,
            "signal": 1 if pred_value == 1 else 0,
        })
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.set_index("timestamp").sort_index()
    
    num_signals = results_df["signal"].sum() if not results_df.empty else 0
    num_no_signals = len(results_df) - num_signals if not results_df.empty else 0
    
    # Debug: Signal-Statistik für besseres Verständnis
    print(f"[features] {len(results_df)} Zeilen verarbeitet, {num_signals} BUY-Signale gefunden")
    if not results_df.empty:
        signal_changes = (results_df["signal"].diff() != 0).sum()
        print(f"[debug] Signal-Statistik:")
        print(f"  BUY-Signale (signal=1): {num_signals} ({num_signals/len(results_df)*100:.1f}%)")
        print(f"  Keine Signale (signal=0): {num_no_signals} ({num_no_signals/len(results_df)*100:.1f}%)")
        print(f"  Signal-Wechsel: {signal_changes} (wichtig für Trade-Häufigkeit)")
    
    return results_df

# -----------------------------
# Paper Trading Simulation
# -----------------------------
def simulate_paper_trading(signals: pd.DataFrame, prices: pd.DataFrame,
                          exit_minutes: int = 30,
                          initial_capital: float = 10000.0,
                          position_size_pct: float = 0.1,
                          exit_on_signal_change: bool = True,
                          min_hold_minutes_for_signal_exit: int = 5) -> Dict:
    """Simuliert Paper Trading ohne echte Orders.
    
    Args:
        exit_on_signal_change: Wenn True, schließe Position auch wenn Signal von 1 auf 0 wechselt
                               (ermöglicht mehr Trades)
        min_hold_minutes_for_signal_exit: Mindest-Haltedauer in Minuten bevor Signal-Wechsel-Exit erlaubt ist
    """
    print(f"[paper] Simuliere Paper Trading...")
    print(f"  Exit nach: {exit_minutes} Minuten")
    if exit_on_signal_change:
        print(f"  Exit auch bei Signal-Wechsel: Ja (nach {min_hold_minutes_for_signal_exit} Min. Mindest-Haltedauer)")
    print(f"  Initiales Kapital: ${initial_capital:,.2f}")
    print(f"  Position-Größe: {position_size_pct*100:.0f}%")
    
    capital = initial_capital
    positions = []
    open_position = None
    
    # Merge Signale mit Preisen
    # WICHTIG: Wir iterieren über ALLE Preis-Datenpunkte (nicht nur Signal-Zeitpunkte)
    # damit Positions-Exits auch zwischen Signalen funktionieren
    signals_reset = signals[["signal", "prediction"]].reset_index()
    prices_reset = prices[["close"]].reset_index()
    
    if signals_reset.columns[0] != "timestamp":
        signals_reset = signals_reset.rename(columns={signals_reset.columns[0]: "timestamp"})
    if prices_reset.columns[0] != "timestamp":
        prices_reset = prices_reset.rename(columns={prices_reset.columns[0]: "timestamp"})
    
    # Sortiere beide DataFrames für merge_asof
    signals_reset = signals_reset.sort_values("timestamp")
    prices_reset = prices_reset.sort_values("timestamp")
    
    # Verwende Preise als Basis (links), Signale als Nachschlagetabelle (rechts)
    # direction="backward" bedeutet: Für jeden Preis-Zeitpunkt hole das letzte verfügbare Signal
    combined = pd.merge_asof(
        prices_reset,
        signals_reset,
        on="timestamp",
        direction="backward"
    ).set_index("timestamp")
    
    # Debug: Zeige Statistiken vor fillna
    signals_available_before = combined["signal"].notna().sum()
    
    # Fülle fehlende Signale (am Anfang, bevor erste Signale verfügbar sind)
    combined["signal"] = combined["signal"].fillna(0).astype(int)
    combined["prediction"] = combined["prediction"].fillna(0).astype(int)
    
    print(f"[debug] Kombinierte Daten: {len(combined)} Zeilen (alle Preis-Datenpunkte)")
    print(f"[debug] Signale ursprünglich verfügbar für {signals_available_before} Zeilen (vor fillna)")
    
    previous_date = None
    for idx, (timestamp, row) in enumerate(combined.iterrows()):
        price = row["close"]
        signal = row["signal"]
        
        # Prüfe ob ein neuer Tag begonnen hat
        current_date = timestamp.date() if hasattr(timestamp, 'date') else pd.Timestamp(timestamp).date()
        
        # Exit-Prüfung
        if open_position is not None:
            entry_time, entry_price, position_value = open_position
            time_diff = (timestamp - entry_time).total_seconds() / 60
            
            should_exit = False
            exit_reason = ""
            
            # Exit nach Zeit
            # WICHTIG: Wir schließen, wenn time_diff >= exit_minutes
            # Aber time_diff könnte etwas größer sein, wenn Datenpunkte nicht genau im richtigen Moment kommen
            # Das ist in Ordnung - wir schließen so schnell wie möglich nach exit_minutes
            if time_diff >= exit_minutes:
                should_exit = True
                exit_reason = "time"
            
            # Exit bei Signal-Wechsel (wenn aktiviert) - aber nur nach Mindest-Haltedauer
            # Dies verhindert, dass Positionen sofort wieder geschlossen werden
            if exit_on_signal_change and signal == 0 and time_diff >= min_hold_minutes_for_signal_exit:
                should_exit = True
                exit_reason = "signal_change"
            
            # FIX: Exit wenn ein neuer Tag begonnen hat und Position vom vorherigen Tag ist
            # Dies verhindert, dass Positionen über Nacht/Wochenende gehalten werden
            # Die normale Exit-Prüfung (time_diff >= exit_minutes) wird bereits oben behandelt,
            # aber wenn ein Tageswechsel stattfindet, schließen wir auch Positionen, die noch
            # nicht exit_minutes alt sind, um über Nacht halten zu vermeiden
            if previous_date is not None and current_date != previous_date:
                entry_date = entry_time.date() if hasattr(entry_time, 'date') else pd.Timestamp(entry_time).date()
                # Wenn Position vom vorherigen Tag (oder älter) ist, schließe sie
                # (verhindert über Nacht/Wochenende halten)
                if entry_date < current_date:
                    # Cappe die Haltedauer auf maximal exit_minutes für die Berechnung
                    # um realistische Statistiken zu erhalten
                    capped_time_diff = min(time_diff, exit_minutes)
                    should_exit = True
                    exit_reason = "time"
            
            if should_exit:
                profit_pct = ((price - entry_price) / entry_price) * 100
                profit = position_value * (profit_pct / 100)
                capital += profit
                
                # Wenn durch Tageswechsel geschlossen und time_diff > exit_minutes:
                # Cappe Haltedauer für realistische Statistiken
                if exit_reason == "time" and previous_date is not None and current_date != previous_date:
                    entry_date = entry_time.date() if hasattr(entry_time, 'date') else pd.Timestamp(entry_time).date()
                    if entry_date < current_date and time_diff > exit_minutes:
                        # Verwende exit_minutes statt der tatsächlichen Haltedauer
                        # für realistische Statistiken (Position sollte nicht über Nacht gehalten werden)
                        actual_hold_time = exit_minutes
                    else:
                        actual_hold_time = time_diff
                else:
                    actual_hold_time = time_diff
                
                positions.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": timestamp,
                    "exit_price": price,
                    "hold_time_minutes": actual_hold_time,
                    "profit_pct": profit_pct,
                    "profit": profit,
                    "position_value": position_value,
                    "capital_after": capital,
                    "exit_reason": exit_reason
                })
                
                open_position = None
        
        # Entry bei Signal (nur wenn keine Position offen)
        if signal == 1 and open_position is None:
            position_value = capital * position_size_pct
            open_position = (timestamp, price, position_value)
        
        # Aktualisiere previous_date für nächste Iteration
        previous_date = current_date
    
    # Schließe offene Position am Ende
    if open_position is not None:
        entry_time, entry_price, position_value = open_position
        final_price = combined["close"].iloc[-1]
        final_time = combined.index[-1]
        time_diff = (final_time - entry_time).total_seconds() / 60
        
        profit_pct = ((final_price - entry_price) / entry_price) * 100
        profit = position_value * (profit_pct / 100)
        capital += profit
        
        positions.append({
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": final_time,
            "exit_price": final_price,
            "hold_time_minutes": time_diff,
            "profit_pct": profit_pct,
            "profit": profit,
            "position_value": position_value,
            "capital_after": capital,
            "exit_reason": "end_of_data"
        })
    
    positions_df = pd.DataFrame(positions)
    
    # Debug: Trade-Statistik
    if not positions_df.empty:
        exit_reasons = positions_df["exit_reason"].value_counts()
        print(f"[debug] Trade-Statistik:")
        print(f"  Gesamt-Trades: {len(positions_df)}")
        print(f"  Exit-Gründe:")
        for reason, count in exit_reasons.items():
            print(f"    {reason}: {count} ({count/len(positions_df)*100:.1f}%)")
        
        # Detaillierte Haltedauer-Statistik
        avg_hold_time = positions_df["hold_time_minutes"].mean()
        median_hold_time = positions_df["hold_time_minutes"].median()
        min_hold_time = positions_df["hold_time_minutes"].min()
        max_hold_time = positions_df["hold_time_minutes"].max()
        
        print(f"  Haltedauer-Statistik:")
        print(f"    Durchschnitt: {avg_hold_time:.1f} Minuten")
        print(f"    Median: {median_hold_time:.1f} Minuten")
        print(f"    Min: {min_hold_time:.1f} Minuten")
        print(f"    Max: {max_hold_time:.1f} Minuten")
        
        # Haltedauer nach Exit-Grund
        for reason in exit_reasons.index:
            reason_trades = positions_df[positions_df["exit_reason"] == reason]
            if not reason_trades.empty:
                avg_time_for_reason = reason_trades["hold_time_minutes"].mean()
                print(f"    {reason}: {avg_time_for_reason:.1f} Min (Ø)")
        
        # Warnung, wenn durchschnittliche Haltedauer deutlich höher ist als Exit-Zeit
        if avg_hold_time > exit_minutes * 2:
            print(f"  ⚠️  WARNUNG: Durchschnittliche Haltedauer ({avg_hold_time:.1f} Min) ist deutlich höher")
            print(f"     als Exit-Zeit ({exit_minutes} Min). Mögliche Ursachen:")
            print(f"     - Ausreißer durch 'end_of_data' Trade")
            print(f"     - Lücken in den Daten")
            print(f"     - Berechnungsproblem")
    else:
        print(f"[debug] WARNUNG: Keine Trades ausgeführt!")
        print(f"[debug] Mögliche Gründe:")
        print(f"  - Signale waren kontinuierlich aktiv (keine Signalwechsel)")
        print(f"  - Erste Position wurde nie geschlossen (end_of_data)")
    
    # Performance-Metriken
    if not positions_df.empty:
        total_return = ((capital - initial_capital) / initial_capital) * 100
        num_trades = len(positions_df)
        winning_trades = len(positions_df[positions_df["profit"] > 0])
        losing_trades = len(positions_df[positions_df["profit"] <= 0])
        win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0
        
        avg_profit = positions_df["profit"].mean()
        avg_profit_pct = positions_df["profit_pct"].mean()
        total_profit = positions_df["profit"].sum()
        
        max_profit = positions_df["profit"].max()
        max_loss = positions_df["profit"].min()
        
        if positions_df["profit_pct"].std() > 0:
            sharpe = (avg_profit_pct / positions_df["profit_pct"].std()) * np.sqrt(252 * 24 * 60 / exit_minutes)
        else:
            sharpe = 0
    else:
        total_return = 0
        num_trades = 0
        winning_trades = 0
        losing_trades = 0
        win_rate = 0
        avg_profit = 0
        avg_profit_pct = 0
        total_profit = 0
        max_profit = 0
        max_loss = 0
        sharpe = 0
    
    return {
        "initial_capital": initial_capital,
        "final_capital": capital,
        "total_return_pct": total_return,
        "num_trades": num_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_profit_pct": avg_profit_pct,
        "total_profit": total_profit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "sharpe_ratio": sharpe,
        "positions": positions_df,
    }

# -----------------------------
# Zeitrahmen-Analyse
# -----------------------------
def analyze_timeframes(positions_df: pd.DataFrame) -> Dict:
    """Analysiert Performance über verschiedene Zeitrahmen."""
    if positions_df.empty:
        return {}
    
    positions_df = positions_df.copy()
    positions_df["date"] = pd.to_datetime(positions_df["entry_time"]).dt.date
    
    # Tägliche Performance
    daily = positions_df.groupby("date").agg({
        "profit": "sum",
        "profit_pct": "mean",
        "entry_time": "count"  # Anzahl Trades
    }).rename(columns={"entry_time": "num_trades"})
    
    # Wöchentliche Performance
    # Konvertiere zu naive datetime für Period (entfernt Timezone-Warning)
    entry_times_naive = pd.to_datetime(positions_df["entry_time"]).dt.tz_localize(None) if pd.to_datetime(positions_df["entry_time"]).dt.tz is not None else pd.to_datetime(positions_df["entry_time"])
    positions_df["week"] = entry_times_naive.dt.to_period("W")
    weekly = positions_df.groupby("week").agg({
        "profit": "sum",
        "profit_pct": "mean",
        "entry_time": "count"
    }).rename(columns={"entry_time": "num_trades"})
    
    # Monatliche Performance
    positions_df["month"] = entry_times_naive.dt.to_period("M")
    monthly = positions_df.groupby("month").agg({
        "profit": "sum",
        "profit_pct": "mean",
        "entry_time": "count"
    }).rename(columns={"entry_time": "num_trades"})
    
    return {
        "daily": daily,
        "weekly": weekly,
        "monthly": monthly,
    }

# -----------------------------
# Vergleich mit Backtest
# -----------------------------
def load_backtest_results() -> Dict | None:
    """Lädt Backtest-Ergebnisse zum Vergleich."""
    backtest_file = os.path.join(PLOTS_DIR, f"backtest_h{HORIZON}m", f"backtest_results_h{HORIZON}m.csv")
    
    if not os.path.exists(backtest_file):
        print(f"[compare] Backtest-Ergebnisse nicht gefunden: {backtest_file}")
        return None
    
    try:
        df = pd.read_csv(backtest_file)
        print(f"[compare] Backtest-Ergebnisse geladen: {len(df)} Positionen")
        
        # Berechne Metriken wie bei Paper Trading
        if not df.empty and "profit" in df.columns:
            initial_capital = 10000.0
            final_capital = df["capital_after"].iloc[-1] if "capital_after" in df.columns else initial_capital + df["profit"].sum()
            total_return = ((final_capital - initial_capital) / initial_capital) * 100
            
            winning_trades = len(df[df["profit"] > 0])
            losing_trades = len(df[df["profit"] <= 0])
            win_rate = (winning_trades / len(df)) * 100 if len(df) > 0 else 0
            
            avg_profit = df["profit"].mean()
            avg_profit_pct = df["profit_pct"].mean() if "profit_pct" in df.columns else 0
            
            if "profit_pct" in df.columns and df["profit_pct"].std() > 0:
                sharpe = (avg_profit_pct / df["profit_pct"].std()) * np.sqrt(252 * 24 * 60 / 30)
            else:
                sharpe = 0
            
            return {
                "total_return_pct": total_return,
                "num_trades": len(df),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "avg_profit": avg_profit,
                "avg_profit_pct": avg_profit_pct,
                "sharpe_ratio": sharpe,
                "max_profit": df["profit"].max(),
                "max_loss": df["profit"].min(),
            }
    except Exception as e:
        print(f"[compare] Fehler beim Laden der Backtest-Ergebnisse: {e}")
    
    return None

# -----------------------------
# Buy & Hold Berechnung
# -----------------------------
def calculate_buy_and_hold(prices: pd.DataFrame, initial_capital: float = 10000.0) -> Dict:
    """Berechnet Buy & Hold Performance.
    
    Args:
        prices: DataFrame mit 'close' Spalte und DatetimeIndex
        initial_capital: Initiales Kapital
    
    Returns:
        Dict mit Buy & Hold Metriken
    """
    if prices.empty or "close" not in prices.columns:
        return {
            "initial_capital": initial_capital,
            "final_capital": initial_capital,
            "total_return_pct": 0.0,
            "equity_curve": pd.DataFrame(),
        }
    
    # Berechne Buy & Hold Performance
    first_price = prices["close"].iloc[0]
    last_price = prices["close"].iloc[-1]
    
    # Anzahl Aktien die man kaufen könnte
    shares = initial_capital / first_price
    final_capital = shares * last_price
    total_return_pct = ((final_capital - initial_capital) / initial_capital) * 100
    
    # Equity Curve über Zeit
    equity_curve = (prices["close"] / first_price) * initial_capital
    equity_curve_df = pd.DataFrame({
        "timestamp": equity_curve.index,
        "equity": equity_curve.values
    }).set_index("timestamp")
    
    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_return_pct": total_return_pct,
        "equity_curve": equity_curve_df,
    }

# -----------------------------
# Visualisierungen
# -----------------------------
def create_analysis_plots(paper_results: Dict, timeframes: Dict, 
                         backtest_results: Dict | None, output_dir: str,
                         prices: pd.DataFrame | None = None):
    """Erstellt umfassende Analyse-Plots."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[plots] Erstelle Analyse-Plots...")
    
    positions_df = paper_results["positions"]
    
    # 1. Performance-Vergleich: Paper Trading vs. Backtest
    if backtest_results:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Metriken-Vergleich
        metrics = ["total_return_pct", "win_rate", "sharpe_ratio", "num_trades"]
        metric_labels = ["Total Return (%)", "Win Rate (%)", "Sharpe Ratio", "Anzahl Trades"]
        
        paper_values = [paper_results.get(m, 0) for m in metrics]
        backtest_values = [backtest_results.get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, ax in enumerate(axes.flat):
            if i < len(metrics):
                ax.bar(x[i] - width/2, paper_values[i], width, label="Paper Trading", alpha=0.7)
                ax.bar(x[i] + width/2, backtest_values[i], width, label="Backtest", alpha=0.7)
                ax.set_ylabel(metric_labels[i])
                ax.set_title(f"{metric_labels[i]} Vergleich")
                ax.legend()
                ax.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "01_performance_comparison.png"), dpi=300)
        plt.close()
    
    # 2. Performance über Zeitrahmen
    if "daily" in timeframes and not timeframes["daily"].empty:
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Täglich
        daily = timeframes["daily"]
        axes[0].plot(daily.index, daily["profit"].cumsum(), marker="o", markersize=2)
        axes[0].set_title("Kumulativer Profit: Täglich")
        axes[0].set_ylabel("Profit ($)")
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis="x", rotation=45)
        
        # Wöchentlich
        if "weekly" in timeframes and not timeframes["weekly"].empty:
            weekly = timeframes["weekly"]
            axes[1].bar(range(len(weekly)), weekly["profit"])
            axes[1].set_title("Profit: Wöchentlich")
            axes[1].set_ylabel("Profit ($)")
            axes[1].set_xlabel("Woche")
            axes[1].grid(True, alpha=0.3, axis="y")
        
        # Monatlich
        if "monthly" in timeframes and not timeframes["monthly"].empty:
            monthly = timeframes["monthly"]
            axes[2].bar(range(len(monthly)), monthly["profit"])
            axes[2].set_title("Profit: Monatlich")
            axes[2].set_ylabel("Profit ($)")
            axes[2].set_xlabel("Monat")
            axes[2].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "02_timeframe_performance.png"), dpi=300)
        plt.close()
    
    # 3. Performance pro Symbol (wenn mehrere Ticker)
    # (Wird nur relevant wenn mehrere Ticker verwendet werden)
    
    # 4. Equity Curve Vergleich
    if not positions_df.empty:
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Paper Trading Equity Curve
        positions_df_sorted = positions_df.sort_values("exit_time")
        paper_equity = [paper_results["initial_capital"]]
        paper_timestamps = []
        if prices is not None and not prices.empty:
            paper_timestamps.append(prices.index[0])
        else:
            paper_timestamps.append(positions_df_sorted["entry_time"].iloc[0])
        
        for _, pos in positions_df_sorted.iterrows():
            paper_equity.append(paper_equity[-1] + pos["profit"])
            paper_timestamps.append(pos["exit_time"])
        
        paper_equity_normalized = [(e / paper_results["initial_capital"]) * 100 for e in paper_equity]
        
        # Buy & Hold Equity Curve (wenn Preisdaten verfügbar)
        if prices is not None and not prices.empty:
            buy_hold = calculate_buy_and_hold(prices, paper_results["initial_capital"])
            if not buy_hold["equity_curve"].empty:
                # Interpoliere Buy & Hold auf Paper Trading Zeitpunkte
                bh_curve = buy_hold["equity_curve"]
                bh_values_at_trades = []
                for ts in paper_timestamps:
                    # Finde den nächsten Wert in der Buy & Hold Curve
                    if ts in bh_curve.index:
                        bh_val = bh_curve.loc[ts, "equity"]
                    else:
                        # Finde den nächsten verfügbaren Wert
                        before = bh_curve[bh_curve.index <= ts]
                        if not before.empty:
                            bh_val = before.iloc[-1]["equity"]
                        else:
                            bh_val = buy_hold["initial_capital"]
                    bh_values_at_trades.append((bh_val / buy_hold["initial_capital"]) * 100)
                
                ax.plot(range(len(bh_values_at_trades)), bh_values_at_trades,
                       label=f"Buy & Hold ({buy_hold['total_return_pct']:.2f}%)",
                       linewidth=2, color="blue", linestyle="--", alpha=0.7)
        
        ax.plot(range(len(paper_equity_normalized)), paper_equity_normalized, 
               label=f"Paper Trading ({paper_results['total_return_pct']:.2f}%)", 
               linewidth=2, color="orange")
        
        ax.axhline(y=100, color="gray", linestyle=":", label="Initial (100%)", alpha=0.5)
        
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Portfolio Value (Normalisiert, Start=100%)")
        ax.set_title("Equity Curve: Paper Trading vs. Buy & Hold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "03_equity_comparison.png"), dpi=300)
        plt.close()
    
    # 5. Marktvergleich: Paper Trading vs. Buy & Hold (über Zeit)
    if prices is not None and not prices.empty and not positions_df.empty:
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Buy & Hold Performance über Zeit
        buy_hold = calculate_buy_and_hold(prices, paper_results["initial_capital"])
        if not buy_hold["equity_curve"].empty:
            bh_normalized = (buy_hold["equity_curve"]["equity"] / buy_hold["initial_capital"]) * 100
            ax.plot(buy_hold["equity_curve"].index, bh_normalized.values,
                   label=f"Buy & Hold ({buy_hold['total_return_pct']:.2f}%)", 
                   linewidth=2, alpha=0.7, color="blue")
        
        # Paper Trading Equity Curve über Zeit
        positions_df_sorted = positions_df.sort_values("exit_time")
        paper_equity = [paper_results["initial_capital"]]
        paper_timestamps = [prices.index[0]]  # Start mit erstem Preis-Zeitpunkt
        
        for _, pos in positions_df_sorted.iterrows():
            paper_equity.append(paper_equity[-1] + pos["profit"])
            paper_timestamps.append(pos["exit_time"])
        
        paper_equity_normalized = [(e / paper_results["initial_capital"]) * 100 for e in paper_equity]
        ax.plot(paper_timestamps, paper_equity_normalized,
               label=f"Paper Trading ({paper_results['total_return_pct']:.2f}%)",
               linewidth=2, color="orange")
        
        ax.axhline(y=100, color="gray", linestyle="--", label="Initial (100%)", alpha=0.5)
        ax.set_xlabel("Zeit")
        ax.set_ylabel("Normalisierter Wert (Start = 100%)")
        ax.set_title("Marktentwicklung: Paper Trading vs. Buy & Hold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "04_market_comparison.png"), dpi=300)
        plt.close()
    
    print(f"[plots] Analyse-Plots gespeichert in: {output_dir}")

# -----------------------------
# Hauptfunktion
# -----------------------------
def main():
    print("=" * 80)
    print("Paper Trading Performance-Analyse")
    print("=" * 80)
    
    # Lade Live-Daten (max. 7 Tage wegen Yahoo Finance Limit)
    data = load_live_data(TICKERS, days=7)
    
    if not data:
        print("[ERROR] Keine Daten geladen!")
        return
    
    all_results = {}
    
    # Analysiere jedes Symbol
    for ticker, df in data.items():
        print(f"\n{'='*80}")
        print(f"Analyse für {ticker}")
        print(f"{'='*80}")
        
        # Generiere Signale
        signals = compute_signals(df)
        
        if signals.empty:
            print(f"[ERROR] Keine Signale für {ticker}")
            continue
        
        # Simuliere Paper Trading
        # OPTIMIERT: Kürzere Exit-Zeit (15 Min) und kürzere Mindest-Haltedauer (2 Min) 
        # für mehr Trades und aussagekräftigere Ergebnisse
        # exit_on_signal_change=True ermöglicht mehr Trades (schließt Position bei Signal-Wechsel)
        # min_hold_minutes_for_signal_exit=2 verhindert sofortiges Schließen (mindestens 2 Min. halten)
        paper_results = simulate_paper_trading(
            signals=signals,
            prices=df[["close"]],
            exit_minutes=15,  # OPTIMIERT: Von 30 auf 15 Minuten reduziert für mehr Trades
            initial_capital=10000.0,
            position_size_pct=0.1,
            exit_on_signal_change=True,  # True = mehr Trades möglich
            min_hold_minutes_for_signal_exit=2  # OPTIMIERT: Von 5 auf 2 Minuten reduziert für schnellere Exits
        )
        
        # Zeitrahmen-Analyse
        timeframes = analyze_timeframes(paper_results["positions"])
        
        # Berechne Buy & Hold Performance
        buy_hold_results = calculate_buy_and_hold(df[["close"]], paper_results["initial_capital"])
        
        all_results[ticker] = {
            "paper": paper_results,
            "timeframes": timeframes,
            "signals": signals,
            "buy_hold": buy_hold_results,
            "prices": df[["close"]],
        }
        
        # Performance-Report
        print("\n" + "=" * 80)
        print(f"PAPER TRADING ERGEBNISSE - {ticker}")
        print("=" * 80)
        print(f"Initiales Kapital:     ${paper_results['initial_capital']:,.2f}")
        print(f"Finales Kapital:       ${paper_results['final_capital']:,.2f}")
        print(f"Total Return:          {paper_results['total_return_pct']:.2f}%")
        print(f"")
        print(f"Anzahl Trades:         {paper_results['num_trades']}")
        print(f"Gewinnende Trades:     {paper_results['winning_trades']}")
        print(f"Verlierende Trades:    {paper_results['losing_trades']}")
        print(f"Win Rate:              {paper_results['win_rate']:.2f}%")
        print(f"")
        print(f"Durchschn. Profit:     ${paper_results['avg_profit']:.2f}")
        print(f"Durchschn. Profit %:   {paper_results['avg_profit_pct']:.2f}%")
        print(f"Max. Profit:           ${paper_results['max_profit']:.2f}")
        print(f"Max. Verlust:          ${paper_results['max_loss']:.2f}")
        print(f"Sharpe Ratio:          {paper_results['sharpe_ratio']:.2f}")
        print("")
        print(f"BUY & HOLD:")
        print(f"  Finales Kapital:     ${buy_hold_results['final_capital']:,.2f}")
        print(f"  Total Return:        {buy_hold_results['total_return_pct']:.2f}%")
        print(f"  Outperformance:      {paper_results['total_return_pct'] - buy_hold_results['total_return_pct']:.2f}%")
        print("=" * 80)
    
    # Vergleich mit Backtest
    backtest_results = load_backtest_results()
    
    # Vergleich: Paper Trading vs. Buy & Hold
    if all_results:
        print("\n" + "=" * 80)
        print("VERGLEICH: PAPER TRADING vs. BUY & HOLD")
        print("=" * 80)
        print(f"{'Metrik':<25} {'Paper Trading':<20} {'Buy & Hold':<20} {'Differenz':<20}")
        print("-" * 80)
        
        for ticker, results in all_results.items():
            paper_val = results["paper"].get("total_return_pct", 0)
            bh_val = results["buy_hold"].get("total_return_pct", 0)
            diff = paper_val - bh_val
            print(f"{'Total Return (%)':<25} {paper_val:>18.2f} {bh_val:>18.2f} {diff:>18.2f}")
            
            paper_capital = results["paper"].get("final_capital", 0)
            bh_capital = results["buy_hold"].get("final_capital", 0)
            capital_diff = paper_capital - bh_capital
            print(f"{'Finales Kapital ($)':<25} {paper_capital:>18.2f} {bh_capital:>18.2f} {capital_diff:>18.2f}")
        
        print("=" * 80)
    
    # Vergleich: Paper Trading vs. Backtest
    if backtest_results:
        print("\n" + "=" * 80)
        print("VERGLEICH: PAPER TRADING vs. BACKTEST")
        print("=" * 80)
        print(f"{'Metrik':<25} {'Paper Trading':<20} {'Backtest':<20} {'Differenz':<20}")
        print("-" * 80)
        
        metrics = [
            ("Total Return (%)", "total_return_pct"),
            ("Win Rate (%)", "win_rate"),
            ("Sharpe Ratio", "sharpe_ratio"),
            ("Anzahl Trades", "num_trades"),
            ("Avg. Profit ($)", "avg_profit"),
        ]
        
        for metric_name, metric_key in metrics:
            paper_val = all_results[list(all_results.keys())[0]]["paper"].get(metric_key, 0)
            backtest_val = backtest_results.get(metric_key, 0)
            diff = paper_val - backtest_val
            print(f"{metric_name:<25} {paper_val:>18.2f} {backtest_val:>18.2f} {diff:>18.2f}")
        
        print("=" * 80)
    
    # Erstelle Visualisierungen
    output_dir = os.path.join(PLOTS_DIR, f"paper_trading_h{HORIZON}m")
    for ticker, results in all_results.items():
        ticker_dir = os.path.join(output_dir, ticker)
        create_analysis_plots(
            results["paper"],
            results["timeframes"],
            backtest_results,
            ticker_dir,
            prices=results.get("prices")
        )
        
        # Speichere Positionen
        if not results["paper"]["positions"].empty:
            results_file = os.path.join(ticker_dir, f"paper_trading_results_{ticker}_h{HORIZON}m.csv")
            results["paper"]["positions"].to_csv(results_file)
            print(f"[results] Positionen gespeichert: {results_file}")
    
    print("\n[done] Paper Trading Analyse abgeschlossen!")

if __name__ == "__main__":
    main()

