# Backtesting-Skript für GRXEUR Trend Prediction

Dieses Skript führt ein vollständiges Backtesting der Trading-Strategie auf historischen GRXEUR-Daten durch.

## Übersicht

Das Backtesting-Skript:
- ✅ Lädt historische GRXEUR-Daten (2010-2018)
- ✅ Generiert Trading-Signale basierend auf dem trainierten Modell
- ✅ Simuliert Trading mit Entry/Exit-Regeln
- ✅ Berechnet Performance-Metriken
- ✅ Erstellt umfassende Visualisierungen

## Voraussetzungen

```bash
pip install pandas numpy matplotlib seaborn scikit-learn torch pyyaml
```

## Verwendung

### Einfachste Verwendung
```bash
python experiment/scripts/08_deployment/backtest.py
```

### Mit Umgebungsvariablen
```bash
export HORIZON=15        # Prediction Horizon (Standard: 15)
export MODEL_TYPE="decision_tree"  # oder "random_forest"
python experiment/scripts/08_deployment/backtest.py
```

## Ausgabe

### Performance-Report (Konsole)
- Initiales/Finales Kapital
- Total Return (%)
- Anzahl Trades
- Win Rate
- Durchschnittlicher Profit
- Sharpe Ratio

### Visualisierungen (experiment/plots/backtest_h15m/)

1. **01_trading_signals_h15m.png**
   - Trading-Signale über Zeit
   - Entry-Punkte (BUY) und Exit-Punkte (SELL)
   - Preisentwicklung

2. **02_equity_curve_h15m.png**
   - Performance-Verlauf (Equity Curve)
   - Portfolio-Wert über Zeit
   - Vergleich mit initialem Kapital

3. **03_trading_distribution_h15m.png**
   - Verteilung der Trading-Punkte über die Zeit
   - Trades pro Tag
   - Trades pro Stunde des Tages

4. **04_profit_distribution_h15m.png**
   - Verteilung der Trade-Profite
   - Win/Loss Verhältnis

5. **05_market_comparison_h15m.png**
   - Trading-Strategie vs. Buy & Hold
   - Normalisierte Performance-Vergleich

### CSV-Export
- `backtest_results_h15m.csv`: Alle Positionen mit Entry/Exit-Zeiten und Profiten

## Trading-Parameter

Aktuell fest im Code:
- **Entry**: Bei Signal = 1 (aufwärts)
- **Exit**: Nach 30 Minuten (konfigurierbar im Code)
- **Initial Capital**: $10,000 (konfigurierbar im Code)

## Anpassungen

Um Parameter zu ändern, editieren Sie die `backtest_strategy()` Funktion:
```python
backtest_results = backtest_strategy(
    signals=signals,
    prices=df[["close"]],
    entry_delay_minutes=0,    # Delay vor Entry
    exit_minutes=30,          # Minuten bis Exit
    initial_capital=10000.0   # Startkapital
)
```

## Interpretation der Ergebnisse

### Total Return
- **Positiv**: Strategie hat Gewinn erzielt
- **Negativ**: Strategie hat Verlust erzielt
- **Vergleich**: Im Vergleich zu Buy & Hold interessant

### Win Rate
- **> 50%**: Mehr gewinnende als verlierende Trades
- **< 50%**: Mehr verlierende als gewinnende Trades
- **Aber**: Win Rate allein sagt nichts über Profitabilität aus (können wenige große Gewinne viele kleine Verluste ausgleichen)

### Sharpe Ratio
- **> 1**: Gute risk-adjusted Returns
- **> 2**: Sehr gute risk-adjusted Returns
- **< 1**: Schlechte risk-adjusted Returns

### Equity Curve
- **Steigend**: Strategie funktioniert
- **Fallend**: Strategie funktioniert nicht
- **Volatil**: Hohes Risiko

## Beispiele

### Beispiel-Ausgabe:
```
BACKTESTING ERGEBNISSE
================================================================================
Initiales Kapital:     $10,000.00
Finales Kapital:       $10,523.45
Total Return:          5.23%

Anzahl Trades:         145
Gewinnende Trades:     78
Verlierende Trades:    67
Win Rate:              53.79%

Durchschn. Profit:     $3.61
Durchschn. Profit %:   0.36%
Max. Profit:           $45.23
Max. Verlust:          -$32.15
Sharpe Ratio:          0.85
================================================================================
```

## Nächste Schritte

1. **Parameter-Optimierung**: Teste verschiedene Exit-Zeiten, Entry-Delays
2. **Risk Management**: Füge Stop-Loss / Take-Profit hinzu
3. **Weitere Metriken**: Max Drawdown, Sortino Ratio, etc.
4. **Vergleich**: Teste verschiedene Modelle (Decision Tree vs. Random Forest)


