# Paper Trading Performance-Analyse

Dieses Skript führt eine vollständige Paper Trading Performance-Analyse durch, indem es Live-Daten von Yahoo Finance nutzt und die Ergebnisse mit Backtest-Ergebnissen vergleicht.

## Übersicht

Das Skript:
- ✅ Lädt Live-Daten von Yahoo Finance (z.B. ^GDAXI als Proxy für GRXEUR)
- ✅ Generiert Trading-Signale basierend auf dem trainierten Modell
- ✅ Simuliert Paper Trading (ohne echte Orders bei Alpaca)
- ✅ Analysiert Performance über verschiedene Zeitrahmen (täglich, wöchentlich, monatlich)
- ✅ Vergleicht Ergebnisse mit Backtest-Ergebnissen
- ✅ Erstellt detaillierte Visualisierungen

## Voraussetzungen

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch pyyaml yfinance pytz
```

## Verwendung

### Einfachste Verwendung
```bash
python experiment/scripts/08_deployment/paper_trading_analysis.py
```

### Mit Umgebungsvariablen
```bash
# Ticker ändern (Standard: ^GDAXI)
export TICKERS="^GDAXI"

# Horizont ändern
export HORIZON=15

# Modell-Typ
export MODEL_TYPE="decision_tree"

python experiment/scripts/08_deployment/paper_trading_analysis.py
```

### Mit mehreren Tickers
```bash
export TICKERS="^GDAXI,EWG,AAPL"
python experiment/scripts/08_deployment/paper_trading_analysis.py
```

## Was wird analysiert?

### 1. Performance-Metriken
- Total Return (%)
- Win Rate
- Sharpe Ratio
- Anzahl Trades
- Durchschnittlicher Profit
- Max. Profit / Max. Verlust

### 2. Zeitrahmen-Analyse
- **Täglich**: Kumulativer Profit pro Tag
- **Wöchentlich**: Profit pro Woche
- **Monatlich**: Profit pro Monat

### 3. Vergleich mit Backtest
- Side-by-Side Vergleich der Metriken
- Equity Curve Vergleich
- Performance-Differenzen

### 4. Performance pro Symbol
- Falls mehrere Ticker verwendet werden
- Individuelle Analyse pro Symbol

## Ausgabe

### Konsole
- Detaillierte Performance-Reports pro Ticker
- Vergleich Paper Trading vs. Backtest

### Visualisierungen (experiment/plots/paper_trading_h15m/)

1. **01_performance_comparison.png**
   - Vergleich Paper Trading vs. Backtest
   - Metriken: Return, Win Rate, Sharpe Ratio, Anzahl Trades

2. **02_timeframe_performance.png**
   - Performance über Zeitrahmen
   - Täglich, Wöchentlich, Monatlich

3. **03_equity_comparison.png**
   - Equity Curve Vergleich
   - Paper Trading vs. Backtest

### CSV-Export
- `paper_trading_results_{TICKER}_h15m.csv`: Alle Positionen mit Details

## Trading-Parameter

Aktuell fest im Code:
- **Entry**: Bei Signal = 1 (aufwärts)
- **Exit**: Nach 30 Minuten
- **Initial Capital**: $10,000
- **Position Size**: 10% des Kapitals pro Trade

## Wichtige Hinweise

### Live-Daten
- Das Skript lädt die letzten 30 Tage von 1-Minuten-Bars
- Yahoo Finance kann Rate-Limiting haben
- Datenqualität hängt von Yahoo Finance ab

### Vergleich mit Backtest
- Backtest verwendet historische GRXEUR-Daten (2010-2018)
- Paper Trading verwendet Live-Daten (z.B. ^GDAXI)
- Unterschiedliche Datenquellen können zu unterschiedlichen Ergebnissen führen

### Simulation vs. Realität
- Keine echten Orders werden platziert
- Keine Slippage, keine Gebühren simuliert
- Idealisierte Ausführung

## Interpretation

### Paper Trading besser als Backtest?
- Mögliche Gründe: Aktuellere Daten, andere Marktbedingungen
- Oder: Zufall, kleinerer Sample-Size

### Paper Trading schlechter als Backtest?
- Mögliche Gründe: Overfitting im Backtest, andere Marktbedingungen
- Oder: Modell funktioniert besser auf historischen Daten

### Ähnliche Performance?
- Gutes Zeichen: Modell ist robust
- Konsistente Performance über verschiedene Zeiträume

## Nächste Schritte

1. **Längere Zeiträume**: Erhöhe `days` Parameter für mehr Daten
2. **Mehrere Ticker**: Teste verschiedene Symbole
3. **Parameter-Optimierung**: Teste verschiedene Exit-Zeiten
4. **Risk Management**: Füge Stop-Loss / Take-Profit hinzu

