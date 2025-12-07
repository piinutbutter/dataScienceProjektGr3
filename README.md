# Experiment Gruppe 3

### Problem Definition

**Target**

Prediction of short-term trend direction for the German equity index (symbol: GRXEUR) over the next  
t = [5, 10, 15, 30, 60] minutes using 1-minute OHLC data.

For every minute in the period 2010-01-01 to 2018-12-31 we compute the linear regression slope of the
future price window of length t and normalize it by the mean price in that window.  
The sign of this normalized slope is used as target (upward vs. downward / flat trend).


**Input Variables**
- open, high, low, close

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


## Step 1 - Data Acquisition

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

**Script**

[data_acquisition.py](scripts/02_data_understanding.py)

This script loads the CSV files, converts them into cleaned DataFrames with proper timestamps, and saves each year as an individual Parquet file. 
It then combines all years into one combined dataset and saves that as a full Parquet file as well.

### Approach
No external market data API is used. Instead, we work with already downloaded ASCII CSV files.
A Python script reads all DAT_ASCII_GRXEUR_M1_*.csv files, parses the timestamp and OHLC columns, and combines them into a unified, time-indexed DataFrame.
Timestamps are parsed from YYYYMMDD HHMMSS into a proper datetime column and used as index.
The cleaned data is stored as Parquet files for efficient downstream processing.


## Step 2 – Data Understanding
This step explores the structure and behavior of the GRXEUR price data.
The goal is to understand how the data behaves before building features and training models.

**Script:**[data_understanding.py](experiment/scripts/02_data_understanding/data_understanding.py)
This script loads the cleaned Parquet files, computes descriptive statistics, and visualizes key aspects of the dataset.

### Plots:
**1. Close Prices** (Example: 2015-01-01 to 2015-01-10)

<img src="experiment/plots/close_2015-01-01_to_2015-01-10.png" alt="drawing" width="800"/>

**Interpretation**
The price moves between ~9400 and ~9900 index points during this period.
Strong intraday movement is visible, including sudden drops and recoveries.
Gaps in the line correspond to weekends and holidays, which is expected for index data.
The pattern shows realistic market dynamics: volatility, trends, and short-term fluctuations.
These observations confirm that the timestamp ordering and OHLC values were loaded correctly.

**2. Volume**

<img src="experiment/plots/volume_2015-01-01_to_2015-01-10.png" alt="drawing" width="800"/>

**Interpretation**
The volume is 0 for every single minute in the dataset.
This is normal for synthetic or derivative index feeds (like GRXEUR), because indexes do not carry real trading volume.
As a result, the volume column does not contain usable information.
Conclusion: Volume will not be used for feature engineering.

**3. Histogram of 1-Minute Returns**

<img src="experiment/plots/returns_hist_2015-01-01_to_2015-01-10.png" alt="drawing" width="800"/>

**Interpretation**
The return distribution is centered very close to 0, meaning most 1-minute price changes are small.
The peak around 0 indicates many “no-movement” or minimal-movement periods.
The distribution has fat tails, which is typical for financial time series:
rare but strong positive or negative price movements.
The shape looks symmetric with slightly heavier density on the left tail, which is also normal.
This confirms that the data behaves like a typical financial intraday time series.

### Descriptive Statistics – Key Observations
**Close Prices**

Median price: around 9700–9800 points depending on the year.
Minimum and maximum values look realistic for the DAX-like GRXEUR index.
No extreme outliers or corrupted values.


**Returns**

Mean return is extremely close to 0, as expected for short intervals.
Standard deviation of returns reflects normal market volatility.
A few strong jumps exist, but they are rare and plausible (market openings, news events).


**Timestamps**

All data is chronologically ordered.
Regular gaps correspond to non-trading hours.
No duplicate timestamps detected.

### Findings
Data quality is good: the dataset is clean, chronologically consistent, and contains realistic price movements.
Volume is meaningless in this dataset and will be excluded from modeling.
Returns behave as expected for an intraday financial time series: centered around zero, heavy-tailed, and symmetric.
Close price behavior matches normal DAX-like index dynamics, including volatility clusters and day-to-day patterns.
This analysis confirms that the dataset is suitable for the next phase.


## Step 3 – Pre-Split Data Preparation

This step prepares the data for machine learning by computing technical features and forward-looking trend targets, then splitting the data chronologically.

**Script:** [main.py](experiment/scripts/03_pre_split_prep/main.py)

### Targets

For each prediction period t = [5, 10, 15, 30, 60] minutes, we compute:
- **Normalized Trend Slope**: Linear regression slope of the future price window, normalized by mean price
- **Trend Direction**: Sign of normalized slope (+1 upward, -1 downward, 0 flat)

This creates 10 target columns: `target_trend_{t}m` and `target_direction_{t}m` for each period.

### Features

The script generates 45 technical features (previously 27):

**Basic Price Features:**
- Normalized close price and 1-minute returns
- Z-normalized close price
- Price range (high - low normalized)
- Open-Close spread normalized

**Moving Averages:**
- Exponential Moving Averages (EMA) for periods [5, 10, 15, 30, 60] minutes (normalized and z-normalized)
- First and second-order slopes of EMAs (normalized)

**Momentum & Volatility:**
- Rolling volatility (std of returns) for periods [15, 30, 60] minutes
- Momentum (price change) for periods [15, 30, 60] minutes

**Technical Indicators:**
- RSI (Relative Strength Index) with period 14
- ATR (Average True Range) with period 14 (normalized)
- Bollinger Bands position (period 20)
- MACD (Moving Average Convergence Divergence): normalized MACD, signal, and histogram
- EMA Crossover (fast-slow EMA difference, normalized)

**Price Position Features:**
- Distance from recent high/low (30-minute window)

**Lagged Features:**
- Lagged returns (1 and 2 periods)

**Time Features:**
- Intraday time features (minute-of-day, day-of-week, hour-of-day)

All features use only past/present data (no lookahead bias).

### Data Splits

The data is split chronologically:
- **Train**: 2010-01-01 to 2016-12-31 (7 years)
- **Validation**: 2017-01-01 to 2017-12-31 (1 year)
- **Test**: 2018-01-01 to 2018-12-31 (1 year)

### Output

Processed datasets are saved to `experiment/data/processed/`:
- `GRXEUR_train.parquet`, `GRXEUR_validation.parquet`, `GRXEUR_test.parquet`
- `features.txt` (list of feature names)

Each file contains OHLC data, all 45 features, all 10 targets, with missing values removed.

## Step 4 - Post-Split Data Preparation
In this step, the pre-split datasets (train, validation, test) are prepared for machine learning models.
The goal is to create clean input matrices X and target vectors y for different prediction horizons.

**Input Data**
The script uses the files created in the pre-split step:
* GRXEUR_train.parquet
* GRXEUR_validation.parquet
* GRXEUR_test.parquet
* features.txt (list of all feature names)

Each dataset already contains:
* all engineered features
* all target values for the prediction horizons
* (5, 10, 15, 30, 60 minutes)

**Feature and Target Selection**
For each prediction horizon, the script:
* loads the corresponding train, validation and test split
* selects all features listed in features.txt as input X
* selects:
  * target_trend_<horizon>m as regression target
  * target_direction_<horizon>m as classification target
Example for horizon 5 minutes:
  * target_trend_5m 
  * target_direction_5m

**Shuffling**
Inside each split (train, validation, test):
* the samples are shuffled using one single random permutation
* the same permutation is applied to:
  * features X
  * directional targets y_dir
  * trend targets y_trend
This guarantees that features and targets always stay correctly aligned.
The random seed is fixed using: RANDOM_STATE = 42. This makes the results reproducible.

**Output Files**
For each prediction horizon, one ML-ready file is created:
GRXEUR_h5m_ml_ready.npz
GRXEUR_h10m_ml_ready.npz
GRXEUR_h15m_ml_ready.npz
GRXEUR_h30m_ml_ready.npz
GRXEUR_h60m_ml_ready.npz

Each file contains:
* X_train, y_train_dir, y_train_trend
* X_val, y_val_dir, y_val_trend
* X_test, y_test_dir, y_test_trend
* feature_names
These files can be directly used to train machine learning models.

**Results Summary (from Console Output)**
For each horizon, the following dataset sizes were created:
* Train: ~1.3 million samples
* Validation: ~206,000 samples
* Test: ~213,000 samples

Each sample contains:
* 45 input features (previously 27)
* 1 directional target (up / down)
* 1 trend target (continuous value)

Example of one feature row:
* price_normalized
* return_1m
* ema_5m_normalized
* ema_5m_z
* ema_10m_normalized
* ...
Example of one target:
* target_trend_5m = 0.0002048
* target_direction_5m = 1

**Final Result**
After this step, the data is:
* fully numerical
* shuffled correctly
* split into train, validation, and test sets
* saved in a compact and ML-ready format
The dataset is now ready for model training and evaluation.

## Step 5 \- Feature Selection (Correlation Analysis)

*Skript:* 
`experiment/scripts/05_feature_selection/main.py` 

### Inputs
1. Preprocessed training data 
2. Feature list

### Main Steps
* Load `features.txt` and the Parquet training file
* Choose a representative prediction horizon (default: `representative_horizon = 15`)
* Drop NaN values and optionally sample to `max_samples` (default: 100000) for faster computation
* Compute Pearson correlations:
   - Feature\-feature correlation matrix
   - Feature\-target correlations (ranking)
* Visualize:
   - Heatmap of feature\-feature correlations (`experiment/plots/06_correlations.png`)
   - Horizontal bar plot of feature\-target correlations (`experiment/plots/06_correlations_target.png`)

### Important Parameters / Adjustments
1. `representative_horizon` — horizon in minutes (e.g. 5, 10, 15, 30, 60).  
2. `symbol` — symbol to load (default: `GRXEUR`).  
3. `max_samples` — sampling limit for large datasets.  
4. `threshold` — threshold for strong correlations (e.g. `0.8`).

### Interpretation of Results
1. The heatmap reveals clusters of strong positive/negative correlations between features  
2. `correlations_feature_pairs_ranked.csv` helps to identify redundant features  
3. Highly correlated feature pairs (|r| > threshold) are candidates for dropping or dimensionality reduction  -> EMA Crossover=macd_normalized
4. The target ranking shows which features are most linearly associated with `target_direction` — useful for feature prioritization
