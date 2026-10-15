# Project Report: Smart Bike Rental Demand Prediction Using Machine Learning

## Abstract
This project predicts hourly bike rental demand from calendar, time and weather features using the UCI Bike Sharing dataset (17,379 hours, 2011-2012). It compares Linear Regression (baseline) with Poisson Regression on a chronological hold-out set (the last 20 % of days, 3,488 hours). Poisson Regression performed better on every metric (MAE 58.9 vs 79.5, RMSE 81.1 vs 106.9, R² 0.865 vs 0.765), but its equidispersion assumption is clearly violated (Pearson dispersion about 25 instead of about 1), so it is used here as a mean predictor only.

## 1. Problem and objectives
Bike-sharing operators need demand estimates to plan bike availability. Objectives: explore demand patterns, prepare features without target leakage, compare two regression methods on identical data, and interpret the errors honestly.

## 2. Dataset
File `hour.csv` from the UCI Bike Sharing dataset. Target `cnt` (total rentals per hour). Citation: Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.

**Data quality (checked in the notebook):** no missing values, no duplicates, no out-of-range values, `cnt` is a non-negative integer (1 to 977). `casual + registered = cnt` in every row. 165 of 17,544 possible hours have no row and no zero-rental hours exist, so the target is effectively zero-truncated. Humidity is exactly 0 in 22 rows and windspeed is 0 in 2,180 rows (kept, noted as likely recording floors).

**Outliers:** the global IQR rule flags 505 rows (2.9 %), all at peak hours; within (hour, workingday) groups only 130 (0.7 %). They are genuine busy periods and were not removed.

## 3. Exploratory findings
- `cnt` is right-skewed: mean 189.5, variance 32,901 (variance/mean = 174).
- Hour of day is the dominant driver, and the shape depends on the day type: working days show commuter peaks at 8h (about 480) and 17h (about 525); non-working days show one midday hump (about 370).
- Demand grew from a mean of 144 (2011) to 235 (2012). Seasonal means: winter 111, spring 208, summer 236, autumn 199.
- Clear weather averages 205 rentals/hour, mist 175, light rain/snow 112. Correlation with `cnt`: temperature +0.40, humidity -0.32, windspeed +0.09.
- `temp` and `atemp` correlate at 0.99; `season` is largely determined by `mnth`.

## 4. Methodology
**Features:** `hr`, `weekday`, `mnth`, `weathersit` (levels 3 and 4 merged because level 4 has 3 rows), and an `hr x workingday` interaction as one-hot categories; `yr`, `holiday`, `workingday` as binary; `temp`, `hum`, `windspeed` standardised. Excluded: `cnt`, `casual`, `registered` (target leakage), `instant` (time counter), `dteday` (replaced by calendar parts), `atemp`, `season` (redundant). No lag features.

**Split:** data sorted by date and hour; the first 80 % of rows by day (13,891 rows, 2011-01-01 to 2012-08-06) train, the last days (3,488 rows, 2012-08-07 to 2012-12-31) test. The cut is moved to a day boundary so no day is shared. Preprocessing is fitted on training rows only inside a scikit-learn `Pipeline` (verified by an assertion). Both models use the same split and the same 95 transformed features.

**Models:** (1) `LinearRegression` (OLS), baseline. (2) `PoissonRegressor` (log link, L2 penalty `alpha`), with `alpha` selected by 5-fold `TimeSeriesSplit` CV on the training data only (alpha = 1). The target was verified to be non-negative and integer-valued.

**Metrics:** MAE, RMSE, R², mean Poisson deviance (defined only for strictly positive predictions). Linear Regression predictions were **not clipped** in the main table.

## 5. Results (held-out test set)
| Model | MAE | RMSE | R² | Mean Poisson deviance |
|---|---|---|---|---|
| Linear Regression | 79.49 | 106.93 | 0.765 | undefined (negative predictions); 49.08 if clipped |
| Poisson Regression | 58.91 | 81.15 | 0.865 | 28.18 |

Test target mean 248.6, std 220.5. The Poisson model's average error is about 24 % of the mean test demand, the baseline's about 32 %.

**Ablation without the hour x workingday interaction:** Linear 100.27 / 135.53 / 0.622; Poisson 88.79 / 121.81 / 0.695 (MAE / RMSE / R²). Poisson still wins, and the interaction improves both models substantially, so feature design matters about as much as the choice of model.

**Diagnostics**
- Linear Regression predicted a negative rental count for 14 of 3,488 test hours (0.4 %), minimum -50.5, all between midnight and 5 a.m. Poisson had none (minimum 19.3).
- **Overdispersion:** Pearson dispersion of the Poisson model is 17.8 (train) and 25.3 (test); the variance/mean ratio is 11-30 in every prediction band. The Poisson variance assumption is not satisfied.
- Lag-1 residual autocorrelation: 0.79 (Linear), 0.73 (Poisson), so the independence assumption is also violated.
- Calibration: Poisson over-predicts the lowest-demand hours (38.6 predicted vs 11.9 actual) and under-predicts the busiest (594 vs 667).
- Largest errors for both models at 8 a.m. and 5-6 p.m.; Poisson is clearly better in the evening peak and at night.

**Coefficients (associations):** one standard deviation higher temperature is associated with about 36 % more rentals (x1.36); light rain/snow or worse with x0.68 compared with clear weather; 2012 with x1.54 compared with 2011; holidays with x0.91.

## 6. Discussion
Poisson Regression is better here, plausibly because its log link produces multiplicative effects, its predictions are always positive, and its loss lets variance grow with the mean. These are plausible explanations and were not isolated by separate experiments. R² is based on squared error and is dominated by busy hours, so it is reported together with MAE and deviance. The comparison is from one chronological split and one test period without confidence intervals, so it is not a statistical significance test.

## 7. Limitations and leakage notes
- Poisson assumptions (equidispersion, independence) do not hold; its standard errors and intervals are unreliable. Negative Binomial/quasi-Poisson or time-series models would be better suited but were outside the two-model scope.
- Observed weather for the predicted hour is used as input; a real system only has forecasts, so real accuracy would be lower.
- A chronological split does not remove all temporal leakage: days adjacent to the boundary are correlated, and no gap was used between train and test.
- The hour x workingday feature was introduced after EDA and both variants were evaluated on the test set, so the final scores are slightly optimistic (the model ranking is the same either way). The Poisson `alpha` was chosen without test data.
- Only August-December 2012 is tested; one system and period; the file has no zero-rental hours.
- Coefficients show associations, not causation.

## 8. Conclusion
On the held-out last 20 % of the data, Poisson Regression outperformed the Linear Regression baseline (MAE 58.9 vs 79.5, RMSE 81.1 vs 106.9, R² 0.865 vs 0.765) and never predicted negative counts. Hour of day interacting with working day is the strongest demand pattern, followed by the 2011-2012 growth, temperature and weather. Even the better model misses by about 59 rentals per hour on average, mostly at rush hours, and the Poisson variance assumption is clearly violated. Suggested improvements: Negative Binomial regression, lag/short-term features or a time-series model, weather forecasts instead of observed weather, and evaluation on several chronological folds.
