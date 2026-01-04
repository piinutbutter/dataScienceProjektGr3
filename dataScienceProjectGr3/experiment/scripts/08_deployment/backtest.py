"""
Backtesting-Skript für GRXEUR Trend Prediction Model.

Dieses Skript:
- Lädt historische GRXEUR-Daten
- Generiert Trading-Signale basierend auf dem trainierten Modell
- Simuliert Trading mit Entry/Exit-Regeln
- Berechnet Performance-Metriken
- Erstellt Visualisierungen:
  - Trading-Signale über Zeit
  - Performance-Verlauf
  - Verteilung der Trading-Punkte
  - Marktentwicklung im Vergleich
"""

from __future__ import annotations

import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import torch
from torch import nn
import importlib.util

# Pfad-Setup
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Projekt-Root: 3 Ebenen nach oben (von 08_deployment -> scripts -> experiment -> dataScienceProjectGr3)
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
# Konstruiere Pfad relativ zum Projekt-Root (wie in anderen Skripten)
model_path = os.path.abspath(os.path.join(PROJECT_ROOT, os.path.normpath(model_path_cfg)))
HORIZON = int(os.getenv("HORIZON", "15"))
MODEL_TYPE = os.getenv("MODEL_TYPE", "decision_tree")
# Anzahl Tage für Backtesting (Standard: 180 Tage = 6 Monate)
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "180"))

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

# Device
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

# Lade MLP (optional, für Embeddings)
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

# -----------------------------
# Daten-Laden
# -----------------------------
def load_historical_data(days: int = 180) -> pd.DataFrame:
    """Lädt historische GRXEUR-Daten.
    
    Args:
        days: Anzahl der letzten Tage, die verwendet werden sollen (Standard: 180 Tage = 6 Monate)
    """
    grxeur_path = os.path.join(DATA_DIR, "raw", "Bars_1m_GRXEUR", "GRXEUR_M1_2010_2018.parquet")
    if not os.path.exists(grxeur_path):
        raise FileNotFoundError(f"GRXEUR-Daten nicht gefunden: {grxeur_path}")
    
    print(f"[data] Lade historische Daten: {grxeur_path}")
    df = pd.read_parquet(grxeur_path)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp").sort_index()
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
    
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    
    df.columns = df.columns.str.lower()
    
    # Filtere auf die letzten N Tage für Vergleichbarkeit mit Paper Trading
    if days > 0:
        end_date = df.index[-1]
        start_date = end_date - pd.Timedelta(days=days)
        df_filtered = df[df.index >= start_date].copy()
        print(f"[data] Gefiltert auf letzte {days} Tage: {len(df_filtered)} Zeilen ({df_filtered.index[0]} bis {df_filtered.index[-1]})")
        print(f"[data] Original: {len(df)} Zeilen ({df.index[0]} bis {df.index[-1]})")
        df = df_filtered
    else:
        print(f"[data] {len(df)} Zeilen geladen: {df.index[0]} bis {df.index[-1]}")
    
    return df

# -----------------------------
# Feature-Berechnung und Signal-Generierung
# -----------------------------
def compute_features_and_predictions(df: pd.DataFrame, start_idx: int = 0, step_size: int = 60) -> pd.DataFrame:
    """Berechnet Features und Vorhersagen für historische Daten."""
    print(f"[features] Berechne Features und Vorhersagen...")
    
    # Für Backtesting: Schrittweise durch die Daten gehen
    results = []
    
    # Fenster-Größe für Feature-Berechnung (braucht genug Historie)
    min_window = max(z_norm_window, max(ema_periods)) + 100
    start_idx = max(start_idx, min_window)
    
    total_steps = (len(df) - start_idx) // step_size
    print(f"[features] Verarbeite {total_steps} Schritte...")
    
    processed = 0
    for i in range(start_idx, len(df), step_size):
        if i < min_window:
            continue
        
        # Fenster für Features (braucht Historie)
        window_start = max(0, i - min_window)
        window_data = df.iloc[window_start:i+1].copy()
        
        if len(window_data) < min_window:
            continue
        
        try:
            # Features berechnen
            df_features_input = window_data[["close", "open", "high", "low"]].copy()
            if "high" in window_data.columns and "low" in window_data.columns:
                df_features_input["vwap"] = (window_data["high"] + window_data["low"] + window_data["close"]) / 3.0
            else:
                df_features_input["vwap"] = window_data["close"]
            
            # Unterdrücke print-Ausgabe von generate_features
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
                continue
            
            # Letzte Zeile für Vorhersage
            if list(tree_feature_cols) != list(FEATURES):
                x_features = X[tree_feature_cols].iloc[[-1]].astype(np.float32).values
            else:
                x_features = X.iloc[[-1]].astype(np.float32).values
            
            # Vorhersage
            pred = clf.predict(x_features)
            pred_proba = None
            if hasattr(clf, 'predict_proba'):
                try:
                    pred_proba = clf.predict_proba(x_features)[0]
                except:
                    pass
            
            timestamp = window_data.index[-1]
            price = window_data["close"].iloc[-1]
            
            results.append({
                "timestamp": timestamp,
                "price": price,
                "prediction": int(pred[0]) if isinstance(pred, np.ndarray) else int(pred),
                "pred_proba": pred_proba,
                "signal": 1 if (pred[0] >= 0.5 if isinstance(pred, np.ndarray) else pred >= 0.5) else 0,
            })
            
            processed += 1
            if processed % 100 == 0:
                print(f"[features] Fortschritt: {processed}/{total_steps} Schritte verarbeitet...")
            
        except Exception as e:
            if i % (step_size * 100) == 0:  # Nur alle 100 Schritte loggen
                print(f"[features] Fehler bei Index {i}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.set_index("timestamp").sort_index()
    
    print(f"[features] {len(results_df)} Vorhersagen generiert")
    return results_df

# -----------------------------
# Backtesting-Engine
# -----------------------------
def backtest_strategy(signals: pd.DataFrame, prices: pd.DataFrame, 
                     entry_delay_minutes: int = 0,
                     exit_minutes: int = 30,
                     initial_capital: float = 10000.0,
                     position_size_pct: float = 1.0) -> Dict:
    """Simuliert Trading-Strategie.
    
    Args:
        position_size_pct: Anteil des Kapitals pro Trade (1.0 = 100%, 0.1 = 10%)
    """
    print(f"[backtest] Starte Backtesting...")
    print(f"  Entry-Delay: {entry_delay_minutes} Minuten")
    print(f"  Exit nach: {exit_minutes} Minuten")
    print(f"  Initiales Kapital: ${initial_capital:,.2f}")
    print(f"  Position-Größe: {position_size_pct*100:.0f}% des Kapitals pro Trade")
    
    capital = initial_capital
    positions = []  # [(entry_time, entry_price, exit_time, exit_price, profit)]
    open_position = None
    
    # Merge Signale mit Preisen
    # Stelle sicher, dass beide DataFrames den gleichen Index-Namen haben
    signals_reset = signals[["signal", "prediction"]].reset_index()
    prices_reset = prices[["close"]].reset_index()
    
    # Benenne die Index-Spalte um, falls nötig
    if signals_reset.columns[0] != "timestamp":
        signals_reset = signals_reset.rename(columns={signals_reset.columns[0]: "timestamp"})
    if prices_reset.columns[0] != "timestamp":
        prices_reset = prices_reset.rename(columns={prices_reset.columns[0]: "timestamp"})
    
    combined = pd.merge_asof(
        signals_reset,
        prices_reset,
        on="timestamp",
        direction="forward"
    ).set_index("timestamp")
    
    for timestamp, row in combined.iterrows():
        price = row["close"]
        signal = row["signal"]
        
        # Exit-Prüfung für offene Position
        if open_position is not None:
            entry_time, entry_price, position_value = open_position
            time_diff = (timestamp - entry_time).total_seconds() / 60
            
            if time_diff >= exit_minutes:
                # Position schließen
                profit_pct = ((price - entry_price) / entry_price) * 100
                profit = position_value * (profit_pct / 100)
                capital += profit
                
                positions.append({
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": timestamp,
                    "exit_price": price,
                    "hold_time_minutes": time_diff,
                    "profit_pct": profit_pct,
                    "profit": profit,
                    "position_value": position_value,
                    "capital_after": capital
                })
                
                open_position = None
        
        # Entry bei Signal
        if signal == 1 and open_position is None:
            # Berechne Position-Größe basierend auf verfügbarem Kapital
            position_value = capital * position_size_pct
            
            # Entry-Delay simulieren (kann 0 sein)
            if entry_delay_minutes == 0:
                entry_price = price
                open_position = (timestamp, entry_price, position_value)
            else:
                # Suche nächsten Preis nach Delay
                future_times = combined[combined.index > timestamp]
                if not future_times.empty:
                    delay_time = timestamp + pd.Timedelta(minutes=entry_delay_minutes)
                    future_prices = future_times[future_times.index >= delay_time]
                    if not future_prices.empty:
                        entry_price = future_prices.iloc[0]["close"]
                        entry_time = future_prices.index[0]
                        open_position = (entry_time, entry_price, position_value)
    
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
            "capital_after": capital
        })
    
    positions_df = pd.DataFrame(positions)
    
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
        
        # Sharpe Ratio (vereinfacht, ohne Risikofreien Zins)
        if positions_df["profit_pct"].std() > 0:
            sharpe = (avg_profit_pct / positions_df["profit_pct"].std()) * np.sqrt(252 * 24 * 60 / exit_minutes)  # Annualisiert
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
        "equity_curve": _calculate_equity_curve(positions_df, initial_capital, combined.index)
    }

def _calculate_equity_curve(positions_df: pd.DataFrame, initial_capital: float, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Berechnet Equity Curve über Zeit."""
    if positions_df.empty:
        return pd.DataFrame({"timestamp": timestamps, "equity": [initial_capital] * len(timestamps)}).set_index("timestamp")
    
    equity = initial_capital
    equity_data = []
    
    # Sortiere Positionen nach Exit-Zeit
    sorted_positions = positions_df.sort_values("exit_time")
    
    for timestamp in timestamps:
        # Füge Profite von abgeschlossenen Positionen hinzu
        completed = sorted_positions[sorted_positions["exit_time"] <= timestamp]
        if len(completed) > 0:
            equity = initial_capital + completed["profit"].sum()
        equity_data.append(equity)
    
    return pd.DataFrame({"timestamp": timestamps[:len(equity_data)], "equity": equity_data}).set_index("timestamp")

# -----------------------------
# Visualisierungen
# -----------------------------
def create_visualizations(signals: pd.DataFrame, prices: pd.DataFrame, 
                         backtest_results: Dict, output_dir: str):
    """Erstellt alle Visualisierungen."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[plots] Erstelle Visualisierungen...")
    
    # 1. Trading-Signale über Zeit
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Preise
    price_subset = prices.loc[signals.index]
    ax.plot(price_subset.index, price_subset["close"], label="GRXEUR Preis", alpha=0.7, linewidth=1)
    
    # Entry-Signale
    buy_signals = signals[signals["signal"] == 1]
    if not buy_signals.empty:
        buy_prices = price_subset.loc[buy_signals.index, "close"]
        ax.scatter(buy_signals.index, buy_prices, color="green", marker="^", 
                  s=100, label="BUY Signal", zorder=5, alpha=0.7)
    
    # Exit-Punkte
    if not backtest_results["positions"].empty:
        exits = backtest_results["positions"]
        exit_prices = price_subset.loc[exits["exit_time"], "close"]
        ax.scatter(exits["exit_time"], exit_prices, color="red", marker="v", 
                  s=100, label="SELL (Exit)", zorder=5, alpha=0.7)
    
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Preis")
    ax.set_title(f"Trading-Signale über Zeit (Horizon: {HORIZON}m, {MODEL_TYPE})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"01_trading_signals_h{HORIZON}m.png"), dpi=300)
    plt.close()
    
    # 2. Performance-Verlauf (Equity Curve)
    fig, ax = plt.subplots(figsize=(16, 6))
    
    equity_curve = backtest_results["equity_curve"]
    ax.plot(equity_curve.index, equity_curve["equity"], label="Portfolio Value", linewidth=2)
    ax.axhline(y=backtest_results["initial_capital"], color="gray", linestyle="--", 
              label=f"Initial Capital (${backtest_results['initial_capital']:,.2f})")
    
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(f"Equity Curve - Total Return: {backtest_results['total_return_pct']:.2f}%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"02_equity_curve_h{HORIZON}m.png"), dpi=300)
    plt.close()
    
    # 3. Verteilung der Trading-Punkte über die Zeit
    if not backtest_results["positions"].empty:
        positions_df = backtest_results["positions"]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Histogramm: Anzahl Trades pro Tag
        positions_df["date"] = pd.to_datetime(positions_df["entry_time"]).dt.date
        trades_per_day = positions_df.groupby("date").size()
        
        ax1.bar(range(len(trades_per_day)), trades_per_day.values, alpha=0.7)
        ax1.set_xlabel("Tage seit Start")
        ax1.set_ylabel("Anzahl Trades")
        ax1.set_title("Verteilung der Trading-Punkte: Trades pro Tag")
        ax1.grid(True, alpha=0.3, axis="y")
        
        # Histogramm: Trades pro Stunde des Tages
        positions_df["hour"] = pd.to_datetime(positions_df["entry_time"]).dt.hour
        trades_per_hour = positions_df.groupby("hour").size()
        
        ax2.bar(trades_per_hour.index, trades_per_hour.values, alpha=0.7, color="orange")
        ax2.set_xlabel("Stunde des Tages")
        ax2.set_ylabel("Anzahl Trades")
        ax2.set_title("Verteilung der Trading-Punkte: Trades pro Stunde")
        ax2.set_xticks(range(0, 24, 2))
        ax2.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"03_trading_distribution_h{HORIZON}m.png"), dpi=300)
        plt.close()
        
        # 4. Profit-Verteilung
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogramm der Profite
        ax1.hist(positions_df["profit"], bins=30, alpha=0.7, edgecolor="black")
        ax1.axvline(x=0, color="red", linestyle="--", linewidth=2)
        ax1.set_xlabel("Profit ($)")
        ax1.set_ylabel("Häufigkeit")
        ax1.set_title("Verteilung der Trade-Profite")
        ax1.grid(True, alpha=0.3, axis="y")
        
        # Win/Loss Verhältnis
        win_loss = [backtest_results["winning_trades"], backtest_results["losing_trades"]]
        ax2.bar(["Gewinnende Trades", "Verlierende Trades"], win_loss, 
               color=["green", "red"], alpha=0.7)
        ax2.set_ylabel("Anzahl")
        ax2.set_title(f"Win/Loss Verhältnis (Win Rate: {backtest_results['win_rate']:.1f}%)")
        ax2.grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"04_profit_distribution_h{HORIZON}m.png"), dpi=300)
        plt.close()
    
    # 5. Marktentwicklung im Vergleich
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Normalisierte Preise (Start = 100)
    price_subset = prices.loc[signals.index]
    normalized_prices = (price_subset["close"] / price_subset["close"].iloc[0]) * 100
    ax.plot(normalized_prices.index, normalized_prices.values, 
           label="GRXEUR (Buy & Hold)", linewidth=2, alpha=0.7)
    
    # Normalisierte Equity Curve
    if not backtest_results["equity_curve"].empty:
        equity_normalized = (backtest_results["equity_curve"]["equity"] / 
                           backtest_results["initial_capital"]) * 100
        ax.plot(equity_normalized.index, equity_normalized.values, 
               label="Trading-Strategie", linewidth=2, color="orange")
    
    ax.set_xlabel("Zeit")
    ax.set_ylabel("Normalisierter Wert (Start = 100)")
    ax.set_title("Marktentwicklung: Trading-Strategie vs. Buy & Hold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"05_market_comparison_h{HORIZON}m.png"), dpi=300)
    plt.close()
    
    print(f"[plots] Visualisierungen gespeichert in: {output_dir}")

# -----------------------------
# Hauptfunktion
# -----------------------------
def main():
    print("=" * 80)
    print("GRXEUR Trend Prediction - Backtesting")
    print("=" * 80)
    print(f"[config] Backtest-Zeitraum: Letzte {BACKTEST_DAYS} Tage ({BACKTEST_DAYS/30:.1f} Monate)")
    
    # Lade Daten (gefiltert auf letzte N Tage)
    df = load_historical_data(days=BACKTEST_DAYS)
    
    # Berechne Features und Signale
    signals = compute_features_and_predictions(df, step_size=60)  # Jede Stunde evaluieren
    
    if signals.empty:
        print("[ERROR] Keine Signale generiert!")
        return
    
    # Backtesting
    # position_size_pct: 1.0 = 100% (alles in jeden Trade), 0.1 = 10% (realistischer)
    backtest_results = backtest_strategy(
        signals=signals,
        prices=df[["close"]],
        entry_delay_minutes=0,
        exit_minutes=30,
        initial_capital=10000.0,
        position_size_pct=0.1  # 100% - kann auf 0.1 (10%) geändert werden für realistischeres Trading
    )
    
    # Performance-Report
    print("\n" + "=" * 80)
    print("BACKTESTING ERGEBNISSE")
    print("=" * 80)
    print(f"Initiales Kapital:     ${backtest_results['initial_capital']:,.2f}")
    print(f"Finales Kapital:       ${backtest_results['final_capital']:,.2f}")
    print(f"Total Return:          {backtest_results['total_return_pct']:.2f}%")
    print(f"")
    print(f"Anzahl Trades:         {backtest_results['num_trades']}")
    print(f"Gewinnende Trades:     {backtest_results['winning_trades']}")
    print(f"Verlierende Trades:    {backtest_results['losing_trades']}")
    print(f"Win Rate:              {backtest_results['win_rate']:.2f}%")
    print(f"")
    print(f"Durchschn. Profit:     ${backtest_results['avg_profit']:.2f}")
    print(f"Durchschn. Profit %:   {backtest_results['avg_profit_pct']:.2f}%")
    print(f"Max. Profit:           ${backtest_results['max_profit']:.2f}")
    print(f"Max. Verlust:          ${backtest_results['max_loss']:.2f}")
    print(f"Sharpe Ratio:          {backtest_results['sharpe_ratio']:.2f}")
    print("=" * 80)
    
    # Visualisierungen
    output_dir = os.path.join(PLOTS_DIR, f"backtest_h{HORIZON}m")
    create_visualizations(signals, df[["close"]], backtest_results, output_dir)
    
    # Speichere Ergebnisse als CSV
    results_file = os.path.join(output_dir, f"backtest_results_h{HORIZON}m.csv")
    if not backtest_results["positions"].empty:
        backtest_results["positions"].to_csv(results_file)
        print(f"\n[results] Positionen gespeichert: {results_file}")
    
    print("\n[done] Backtesting abgeschlossen!")

if __name__ == "__main__":
    main()

