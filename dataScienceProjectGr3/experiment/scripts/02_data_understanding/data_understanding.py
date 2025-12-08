"""
Data Understanding für GRXEUR 1-Minuten-Daten.

Dieses Skript:
- lädt bereinigte 1-Minuten-Bars für GRXEUR aus einer Parquet-Datei
- erklärt die relevanten Datenspalten
- zeigt beschreibende Statistiken
- erzeugt einfache Plots (Close, Volume, Histogramm der Returns)
- gibt ein paar automatische "Findings" aus
"""

import os
import matplotlib
import pandas as pd
import yaml
from pathlib import Path

# 0) matplotlib Backend: versuche TkAgg, fallback auf Agg (z.B. bei headless/CI)
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Projekt-Root (dataScienceProjectGr3 Verzeichnis)
project_root = Path(__file__).resolve().parents[3]

# 1) Konfiguration laden (relativ zum Projekt-Root)
config_path = project_root / "experiment" / "conf" / "params.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"params.yaml nicht gefunden: {config_path.resolve()}")
with config_path.open("r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# Basis-Datenpfad aus der YAML lesen; relativ zum Projekt-Root interpretieren, falls nicht absolut
data_path = Path(params["DATA_ACQUISITION"]["DATA_PATH"])
if not data_path.is_absolute():
    # Pfad sollte relativ zum project_root sein, füge "experiment/" hinzu falls nicht vorhanden
    if not str(data_path).startswith("experiment/"):
        data_path = project_root / "experiment" / data_path
    else:
        data_path = project_root / data_path
data_path = data_path.resolve()

# robustes Auffinden der Parquet-Datei im Ordner Bars_1m_GRXEUR
bars_dir = data_path / "Bars_1m_GRXEUR"
expected_name = "GRXEUR_M1_2010_2018.parquet"
expected_file = bars_dir / expected_name

if expected_file.exists():
    bars_file = expected_file
else:
    if bars_dir.exists():
        parquet_files = sorted(bars_dir.glob("*.parquet"))
        if parquet_files:
            bars_file = parquet_files[0]
            print(f"WARNUNG: Erwartete Datei nicht gefunden. Verwende stattdessen: {bars_file.resolve()}")
        else:
            raise FileNotFoundError(f"Kein .parquet in {bars_dir.resolve()}")
    else:
        raise FileNotFoundError(f"Verzeichnis nicht gefunden: {bars_dir.resolve()}")

print(f"Lade Daten aus: {bars_file.resolve()}")
try:
    df = pd.read_parquet(bars_file)
except Exception as e:
    # zusätzliche Hinweise, falls Engine fehlt
    engine_hint = ""
    try:
        import pyarrow  # type: ignore
    except Exception:
        engine_hint = " Installiere `pyarrow` oder `fastparquet` (z.B. `pip install pyarrow`)."
    raise RuntimeError(
        f"Fehler beim Lesen von {bars_file.resolve()}: {e}\n{engine_hint}"
    ) from e

# Sicherstellen, dass ein Zeitstempel vorhanden ist und als Index gesetzt wird.
if df.empty:
    raise RuntimeError(f"DataFrame ist leer nach dem Laden von {bars_file.resolve()}")

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
else:
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"

print("\nErste Zeilen der Daten:")
print(df.head())

print("\nSpalten und Datentypen:")
print(df.dtypes)


# 3) Relevante Spalten erklären

column_descriptions = {
    "open": "Eröffnungskurs der Minute in Indexpunkten (GRXEUR).",
    "high": "Höchster Kurs innerhalb dieser Minute.",
    "low": "Tiefster Kurs innerhalb dieser Minute.",
    "close": "Schlusskurs der Minute (letzter gehandelter Preis).",
    "volume": "Gehandeltes Volumen in dieser Minute (Einheit abhängig vom Anbieter).",
}

print("\nSpaltenbeschreibungen:")
for col, desc in column_descriptions.items():
    if col in df.columns:
        print(f"- {col}: {desc}")
    else:
        print(f"- {col}: (nicht im DataFrame vorhanden)")


# 4) Beschreibende Statistiken

print("\nBeschreibende Statistik für alle numerischen Spalten:")
print(df.describe().T)

if "close" in df.columns:
    print("\nBeschreibende Statistik für 'close':")
    print(df["close"].describe())

if "volume" in df.columns:
    print("\nBeschreibende Statistik für 'volume':")
    print(df["volume"].describe())


# 5) 1-Minuten-Returns berechnen

if "close" in df.columns:
    df["return_1m"] = df["close"].pct_change()
    print("\nBeschreibende Statistik für 1-Minuten-Returns:")
    print(df["return_1m"].describe())

    print("\nTop 5 größten positiven Returns:")
    print(df["return_1m"].nlargest(5))

    print("\nTop 5 größten negativen Returns:")
    print(df["return_1m"].nsmallest(5))
else:
    print("\nHinweis: 'close'-Spalte fehlt, Returns können nicht berechnet werden.")
    df["return_1m"] = pd.Series(index=df.index, dtype=float)


# 6) Plots erzeugen

# Beispielzeitraum für Visualisierung
start_date = "2015-01-01"
end_date = "2015-01-10"

df_sample = df.loc[start_date:end_date].copy()

if df_sample.empty:
    print(
        f"\nWARNUNG: Zeitraum {start_date} bis {end_date} enthält keine Daten. "
        "Bitte Datumsspanne im Skript anpassen."
    )
else:
    output_dir = project_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Close
    fig1 = plt.figure(figsize=(14, 7))
    plt.plot(df_sample.index, df_sample["close"], label="Close (GRXEUR)")
    plt.title(f"GRXEUR Schlusskurse von {start_date} bis {end_date}")
    plt.xlabel("Zeit")
    plt.ylabel("Indexstand (Punkte)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig1_path = output_dir / f"close_{start_date}_to_{end_date}.png"
    fig1.savefig(fig1_path, dpi=150)
    plt.close(fig1)

    # Plot 2: Volume
    if "volume" in df_sample.columns:
        fig2 = plt.figure(figsize=(14, 5))
        plt.plot(df_sample.index, df_sample["volume"], label="Volume", alpha=0.7)
        plt.title(f"GRXEUR Volumen von {start_date} bis {end_date}")
        plt.xlabel("Zeit")
        plt.ylabel("Volumen (Einheit Datenanbieter)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig2_path = output_dir / f"volume_{start_date}_to_{end_date}.png"
        fig2.savefig(fig2_path, dpi=150)
        plt.close(fig2)

# Histogramm der 1-Minuten-Returns
if "return_1m" in df.columns and df["return_1m"].notna().any():
    output_dir = project_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig3 = plt.figure(figsize=(10, 6))
    returns = df["return_1m"].dropna()

    returns = returns.clip(
        lower=returns.quantile(0.01),
        upper=returns.quantile(0.99),
    )

    plt.hist(returns, bins=100, edgecolor="black", alpha=0.7)
    plt.title("Histogramm der 1-Minuten-Returns (GRXEUR, 1%- bis 99%-Quantil)")
    plt.xlabel("Return (z.B. 0.001 = 0.1 %)")
    plt.ylabel("Häufigkeit")
    plt.grid(True)
    plt.tight_layout()
    fig3_path = output_dir / f"returns_hist_{start_date}_to_{end_date}.png"
    fig3.savefig(fig3_path, dpi=150)
    plt.close(fig3)

# Einzel-Tag-Plot
sample_day = "2015-01-05"
try:
    df_day = df.loc[sample_day]
except Exception:
    df_day = df[df.index.date == pd.to_datetime(sample_day).date()]

if df_day.empty:
    print(f"\nWARNUNG: Kein Intraday-Daten für {sample_day} vorhanden. Einzel-Tag-Plot übersprungen.")
else:
    from matplotlib.dates import DateFormatter, AutoDateLocator

    fig_day = plt.figure(figsize=(14, 4))
    plt.plot(df_day.index, df_day["close"], marker=".", linestyle="-", label=f"Close {sample_day}")
    ax = plt.gca()
    ax.xaxis.set_major_locator(AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    plt.title(f"GRXEUR Intraday Close am {sample_day}")
    plt.xlabel("Uhrzeit")
    plt.ylabel("Indexstand (Punkte)")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig_day_path = output_dir / f"close_day_{sample_day}.png"
    fig_day.savefig(fig_day_path, dpi=150)
    plt.close(fig_day)


# 7) Einfache automatische Findings

print("\n--- Einfache automatische Beobachtungen ---")

if "close" in df.columns:
    print(
        f"- Der Median des Schlusskurses liegt bei ca. {df['close'].median():.2f} Punkten "
        f"(Min: {df['close'].min():.2f}, Max: {df['close'].max():.2f})."
    )

if "volume" in df.columns:
    print(
        f"- Das typische Minutenvolumen liegt im Median bei {df['volume'].median():.2f}, "
        f"mit Spitzen bis {df['volume'].max():.2f}."
    )

if "return_1m" in df.columns and df["return_1m"].notna().any():
    print(
        f"- Die 1-Minuten-Returns sind im Mittel nahe 0 "
        f"({df['return_1m'].mean():.6f}), mit einer Standardabweichung von "
        f"{df['return_1m'].std():.6f}, was die typische kurzfristige Schwankung beschreibt."
    )

print("\nData Understanding abgeschlossen.")