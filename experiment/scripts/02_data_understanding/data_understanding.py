"""
Data Understanding für GRXEUR 1-Minuten-Daten.

Dieses Skript:
- lädt bereinigte 1-Minuten-Bars für GRXEUR aus einer Parquet-Datei
- erklärt die relevanten Datenspalten
- zeigt beschreibende Statistiken
- erzeugt einfache Plots (Close, Volume, Histogramm der Returns)
- gibt ein paar automatische "Findings" aus
"""

import matplotlib
import pandas as pd
import yaml
from pathlib import Path


matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# 1) Konfiguration laden

# Pfad zur params.yaml
params = yaml.safe_load(open("experiment/conf/params.yaml"))

# Basis-Datenpfad aus der YAML lesen
data_path = Path(params["DATA_ACQUISITION"]["DATA_PATH"])

bars_file = data_path / "Bars_1m_GRXEUR" / "GRXEUR_M1_2010_2018.parquet"


# 2) Daten laden und Grundstruktur prüfen

print(f"Lade Daten aus: {bars_file}")
df = pd.read_parquet(bars_file)

# Sicherstellen, dass ein Zeitstempel vorhanden ist und als Index gesetzt wird.
# Fall A: es gibt eine Spalte 'timestamp'
# Fall B: Index ist schon Zeit, aber noch nicht als datetime typisiert
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


# 3) Relevante Spalten erklären (Explain relevant data columns)

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


# 4) Beschreibende Statistiken (Show relevant descriptive statistics)

print("\nBeschreibende Statistik für alle numerischen Spalten:")
print(df.describe().T)  # .T = transponiert, damit Spalten zeilenweise dargestellt werden

# spezifischer Blick auf 'close' und 'volume'
if "close" in df.columns:
    print("\nBeschreibende Statistik für 'close':")
    print(df["close"].describe())

if "volume" in df.columns:
    print("\nBeschreibende Statistik für 'volume':")
    print(df["volume"].describe())


# 5) 1-Minuten-Returns berechnen und untersuchen

if "close" in df.columns:
    # Prozentuale Veränderung von Minute zu Minute
    df["return_1m"] = df["close"].pct_change()

    print("\nBeschreibende Statistik für 1-Minuten-Returns:")
    print(df["return_1m"].describe())

    print("\nTop 5 größten positiven Returns:")
    print(df["return_1m"].nlargest(5))

    print("\nTop 5 größten negativen Returns:")
    print(df["return_1m"].nsmallest(5))
else:
    print("\nHinweis: 'close'-Spalte fehlt, Returns können nicht berechnet werden.")
    df["return_1m"] = None


# 6) Plots erzeugen (Show relevant plots of variables)

# Wir wählen einen Beispielzeitraum für die Visualisierung
start_date = "2015-01-01"
end_date = "2015-01-10"

df_sample = df.loc[start_date:end_date].copy()

if df_sample.empty:
    print(
        f"\nWARNUNG: Zeitraum {start_date} bis {end_date} enthält keine Daten. "
        "Bitte Datumsspanne im Skript anpassen."
    )
else:
# Plot 1: Schlusskurs (Close) im Zeitverlauf
    output_dir = Path("experiment/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1 = plt.figure(figsize=(14, 7))
    plt.plot(df_sample.index, df_sample["close"], label="Close (GRXEUR)")
    plt.title(f"GRXEUR Schlusskurse von {start_date} bis {end_date}")
    plt.xlabel("Zeit")
    plt.ylabel("Indexstand (Punkte)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig1_path = output_dir / f"close_{start_date}_to_{end_date}.png"
    plt.savefig(fig1_path, dpi=150)
    plt.close(fig1)

    # Plot 2: Volumen im Zeitverlauf
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
        plt.savefig(fig2_path, dpi=150)
        plt.close(fig2)

# Plot 3: Histogramm der 1-Minuten-Returns über den gesamten Datensatz
if "return_1m" in df.columns and df["return_1m"].notna().any():
    output_dir = Path("experiment/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig3 = plt.figure(figsize=(10, 6))
    returns = df["return_1m"].dropna()

    # Extreme Ausreißer abschneiden, damit die Verteilung sichtbar bleibt
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
    plt.savefig(fig3_path, dpi=150)
    plt.close(fig3)


# 7) Einfache automatische Findings (Present findings)

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
