# Viva Preparation: Smart Bike Rental Demand Prediction

Numbers below come from the executed notebook (test set = 2012-08-07 to 2012-12-31, 3,488 hours).

1. **Objective?** Predict hourly total rentals (`cnt`) from time, calendar and weather features, and compare Linear Regression with Poisson Regression.
2. **Why regression, not classification?** The target is a number (a count), so regression metrics, not accuracy, apply.
3. **Dataset?** UCI Bike Sharing `hour.csv`, 17,379 hourly rows, 2011-2012 (Washington D.C.).
4. **Why exclude `casual` and `registered`?** They add up exactly to `cnt` (checked on every row); using them would leak the answer.
5. **Why exclude `instant`, `atemp`, `season`?** `instant` is a time counter and cannot extrapolate; `atemp` is almost identical to `temp` (corr 0.99); `season` is redundant given the month.
6. **Data-quality findings?** No missing values or duplicates, no invalid values; 165 hours have no row and there are no zero-rental hours (zero-truncated target).
7. **Outliers?** The IQR rule flags 2.9 % of rows, all at peak hours. They are real busy periods, so they were kept.
8. **Main EDA patterns?** Working days have commuter peaks at 8 a.m. and 5-6 p.m.; non-working days have a midday hump. 2012 demand was about 63 % higher than 2011. Clear weather and warmer temperatures go with more rentals.
9. **Why the hour x workingday interaction?** The daily pattern differs between working and non-working days; an additive model cannot represent that. It lowered test MAE from 100.3 to 79.5 (Linear) and from 88.8 to 58.9 (Poisson). Caveat: I chose it after EDA and saw the test results for both variants, so the final scores are slightly optimistic.
10. **Linear Regression?** Models the target as an additive linear combination of features. Simple, interpretable baseline, but can predict negative values and assumes constant variance.
11. **Poisson Regression?** A GLM for counts with a log link: expected count = exp(linear predictor), so predictions are positive and effects are multiplicative; variance is assumed equal to the mean.
12. **Poisson assumptions - are they met?** No. Equidispersion fails: Pearson dispersion is 17.8 (train) and 25.3 (test) instead of about 1. Independence also fails: residual lag-1 autocorrelation is 0.73. So it is used only as a mean predictor; its confidence intervals and p-values would be too narrow.
13. **What is overdispersion and what is the remedy?** Variance larger than the mean. Use Negative Binomial or quasi-Poisson regression (not used because only two models were allowed).
14. **Results?** Poisson: MAE 58.9, RMSE 81.1, R² 0.865, deviance 28.2. Linear: MAE 79.5, RMSE 106.9, R² 0.765. Poisson was better on all metrics here.
15. **How big are the errors?** Poisson is off by about 59 rentals/hour (about 24 % of the mean test demand of 248.6); Linear by about 79 (32 %). Biggest errors at 8 a.m. and 5-6 p.m.
16. **Negative predictions?** Linear Regression predicted negative rentals for 14 of 3,488 hours (0.4 %), minimum -50.5, at night. Poisson never does. The main table does not clip them.
17. **Why might Poisson win?** Multiplicative effects from the log link, positive predictions, and variance growing with the mean. These are plausible reasons, not separately proven.
18. **Limitations of R²?** Based on squared error, dominated by busy hours, ignores that count variance grows with the mean, can be negative on test data. Report it with MAE and deviance.
19. **What is Poisson deviance?** A goodness-of-fit measure for counts. It is undefined when a prediction is zero or negative, so it cannot be computed for the raw Linear predictions.
20. **Why a chronological split?** Train on the past, test on the future. A random split would let the model see neighbouring hours of test hours.
21. **Does the chronological split remove all leakage?** No. Neighbouring days are correlated, and the model uses the observed weather of the predicted hour (in practice only a forecast is available). I also made a feature choice after seeing test results. These are documented in the notebook.
22. **Why did the split move to a day boundary?** The raw 80 % cut fell mid-day on 2012-08-07; I moved it so no day appears in both sets.
23. **How was data leakage prevented in preprocessing?** Scaling and encoding are in a Pipeline fitted on the training rows only; the notebook asserts that the scaler means equal the training means. Poisson's `alpha` was tuned with `TimeSeriesSplit` on training data only.
24. **Why one-hot encode hour/weekday/month?** Demand is not linear in these (hour 8 and 17 are peaks, hour 12 is not), and one-hot avoids imposing an order.
25. **How do you interpret a Poisson coefficient?** exp(coefficient) is a multiplicative effect on expected rentals: for example one standard deviation warmer is about x1.36 (+36 %), light rain/snow about x0.68. These are associations, not causal effects.
26. **Why is test demand higher than training demand?** 2012 had more users than 2011 (mean 235 vs 144), and the test set is entirely in 2012.
27. **Limitations overall?** Single split and test period (Aug-Dec 2012), no confidence intervals, one system, violated Poisson assumptions, zero-truncated data, forecast weather unavailable.
28. **Future improvements?** Negative Binomial regression, lag features or a time-series model, weather forecasts as inputs, multiple chronological folds, adding event data.
