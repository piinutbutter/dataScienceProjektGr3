# Experiment Gruppe 3

### Problem Definition

**Target**

Prediction of short-term trend direction for the German equity index (symbol: GRXEUR) over the next  
t = [5, 10, 15, 30, 60] minutes using 1-minute OHLC data.

For every minute in the period 2010-01-01 to 2018-12-31 we compute the linear regression slope of the
future price window of length t and normalize it by the mean price in that window.  
The sign of this normalized slope is used as target (upward vs. downward / flat trend).

**Input Features (planned)**

- Normalized close price and 1-minute returns
- Normalized exponential moving averages (EMA) over t = [5, 10, 15, 30, 60] minutes
- Slopes and second order slopes of EMAs
- Optionally: intraday time features (minute-of-day, day-of-week)

### Procedure Overview

- Use historical 1-minute bar data (OHLC, volume) for GRXEUR from 2010 → 2018.

  (- Clean and unify all yearly ASCII files into a single, time-indexed dataset.)
- Engineer technical features (returns, EMAs, slopes).
- Compute forward-looking trend targets as described above.
- Later: train and evaluate machine learning models to predict short-term trend direction.


## Data Acquisition

We use historical 1-minute bar data for the German equity index (symbol: GRXEUR) for the years 2010–2018.
The data comes as ASCII CSV files exported from a trading data provider.

**Raw Files**

- `DAT_ASCII_GRXEUR_M1_2010.csv`
- `DAT_ASCII_GRXEUR_M1_2011.csv`
- ...
- `DAT_ASCII_GRXEUR_M1_2018.csv`

Each file contains 1-minute OHLC data with the following format (semicolon-separated, no header):

`YYYYMMDD HHMMSS;open;high;low;close;volume`

Example rows from `DAT_ASCII_GRXEUR_M1_2010.csv`:

[data_acquisition.py](scripts/02_data_understanding.py)

```text
20101115 020000;6709.000000;6709.500000;6703.500000;6705.000000;0
20101115 020100;6705.000000;6710.500000;6705.000000;6710.000000;0
20101115 020200;6710.500000;6713.500000;6710.500000;6713.500000;0
20101115 020300;6713.500000;6713.500000;6711.500000;6712.000000;0
20101115 020400;6712.500000;6715.000000;6712.500000;6714.000000;0
```


### Approach
No external market data API is used. Instead, we work with already downloaded ASCII CSV files.
A Python script reads all DAT_ASCII_GRXEUR_M1_*.csv files, parses the timestamp and OHLC columns, and combines them into a unified, time-indexed DataFrame.
Timestamps are parsed from YYYYMMDD HHMMSS into a proper datetime column and used as index.
The cleaned data is stored as Parquet files for efficient downstream processing.