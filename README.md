# Smart Bike Rental Demand Prediction Using Machine Learning

## Overview
Predict hourly total bike rentals (`cnt`) from calendar, time and weather features. Exactly two models are compared: **Linear Regression** (baseline) and **Poisson Regression** (generalized linear model for count data, part of the syllabus).

## Dataset
UCI Bike Sharing, hourly file `hour.csv` (17,379 rows, 2011-01-01 to 2012-12-31): https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
Citation: Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.
A copy is included in `data/hour.csv`. The notebook first tries to download the official ZIP (3 attempts); if that fails it automatically uses the local copy and prints which source was used.

## How to run
```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Bike_Rental_Demand_Prediction.ipynb
```
or open the notebook in Jupyter / VS Code / Kaggle and run all cells (about 1 minute). Run it from the project folder so the relative `data/` and `outputs/` paths work. It creates `outputs/` (figures, `model_comparison.csv`, `test_predictions.csv`, `coefficients.csv`).

## Files
| File | Purpose |
|---|---|
| `Bike_Rental_Demand_Prediction.ipynb` | Full analysis, executed, with all outputs |
| `data/hour.csv` | Local copy of the dataset (offline fallback) |
| `outputs/` | Generated figures, metrics, predictions, coefficients |

## Method summary
1. Data-quality checks: missing values, duplicates, dtypes, value ranges, target integrity, missing hours.
2. IQR outlier inspection (flagged, not removed).
3. EDA: hour/working-day profile, weekday, month, season, weather, trend, correlations.
4. Features: `hr`, `weekday`, `mnth`, `weathersit` (one-hot), `hr x workingday` interaction, `yr`, `holiday`, `workingday`, `temp`, `hum`, `windspeed` (scaled). Excluded: `cnt`, `casual`, `registered` (sum to `cnt`, target leakage), `instant`, `dteday`, `atemp` (collinear with `temp`), `season` (redundant with month).
5. Chronological split on whole days: first 80 % train (2011-01-01 to 2012-08-06), last 20 % test (2012-08-07 to 2012-12-31). Preprocessing is fitted on training rows only (asserted in the notebook). Both models use the same split and the same transformed features.
6. Poisson `alpha` tuned by `TimeSeriesSplit` CV on the training data only (alpha = 1).
7. Metrics: MAE, RMSE, RÂ², Poisson deviance. Diagnostics: negative predictions, overdispersion, residual autocorrelation, calibration by prediction band.

## Results (held-out test set, 3,488 hours)
| Model | MAE | RMSE | RÂ² | Mean Poisson deviance |
|---|---|---|---|---|
| Linear Regression | 79.49 | 106.93 | 0.765 | undefined (14 negative predictions) |
| Poisson Regression | 58.91 | 81.15 | 0.865 | 28.18 |

Poisson Regression performed better on every metric in this test period. Without the hour x workingday interaction both models are worse (Linear MAE 100.3 / Poisson 88.8) and the ranking is unchanged.

## Key limitations (details in the notebook)
- Poisson's equidispersion assumption is clearly violated (Pearson dispersion 17.8 train / 25.3 test; about 1 expected) and residuals are autocorrelated (lag-1 about 0.73). It is used as a mean predictor only; its intervals/p-values are not trustworthy.
- Observed weather of the predicted hour is used as input; real use needs forecasts, so real accuracy would be lower.
- A chronological split does not remove all temporal leakage; only Aug-Dec 2012 is tested; one split, no confidence intervals.
- The hour x workingday feature was chosen after EDA and both variants were evaluated on the test set, so final scores are slightly optimistic.
- The file contains no zero-rental hours (165 hours have no row).
- Coefficients are associations, not causal effects.
