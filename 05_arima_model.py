"""
STEP 5: ARIMA MODEL BUILDING
=============================
Project: Monthly Sales Forecast and Trend Analysis
Dataset: Superstore Sales
File: 05_arima_model.py
=============================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("STEP 5: ARIMA MODEL BUILDING")
print("="*80)

# ============================================================================
# 5.1 LOAD TRAIN AND TEST DATA
# ============================================================================
print("\n[5.1] Loading Train and Test Data...")

train = pd.read_csv('data_train.csv', index_col=0, parse_dates=True)['Sales']
test = pd.read_csv('data_test.csv', index_col=0, parse_dates=True)['Sales']

print(f"✓ Training data: {len(train)} months")
print(f"✓ Test data: {len(test)} months")

# ============================================================================
# 5.2 DETERMINE ARIMA PARAMETERS
# ============================================================================
print("\n[5.2] Determining ARIMA Parameters (p, d, q)...")

# Check stationarity to determine 'd' parameter
def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1] <= 0.05

is_stationary = check_stationarity(train)
print(f"\n  Stationarity Test:")
print(f"    Original series stationary: {is_stationary}")

# Determine d (order of differencing)
d = 0
series = train.copy()
max_diff = 2

while not is_stationary and d < max_diff:
    d += 1
    series = series.diff().dropna()
    is_stationary = check_stationarity(series)
    print(f"    After {d} differencing: {is_stationary}")

print(f"\n  Recommended d (differencing order): {d}")

# Analyze ACF and PACF for p and q
print("\n  Analyzing ACF and PACF plots...")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

plot_acf(train, lags=min(20, len(train)//2), ax=axes[0], color='#2E86AB')
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Lag', fontsize=11)
axes[0].set_ylabel('Correlation', fontsize=11)
axes[0].grid(True, alpha=0.3)

plot_pacf(train, lags=min(20, len(train)//2), ax=axes[1], color='#A23B72')
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Lag', fontsize=11)
axes[1].set_ylabel('Correlation', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/14_arima_acf_pacf.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/14_arima_acf_pacf.png")
plt.close()

print("\n  Initial parameter suggestions:")
print("    p (AR order): Check PACF - number of significant lags")
print("    d (differencing): Based on stationarity test")
print("    q (MA order): Check ACF - number of significant lags")

# ============================================================================
# 5.3 GRID SEARCH FOR BEST PARAMETERS (OPTIONAL)
# ============================================================================
print("\n[5.3] Grid Search for Optimal Parameters...")
print("  (This may take a few minutes...)")

# Define parameter range
p_values = range(0, 3)  # AR order
d_values = range(0, 2)  # Differencing
q_values = range(0, 3)  # MA order

best_aic = np.inf
best_params = None
results = []

total_combinations = len(p_values) * len(d_values) * len(q_values)
current = 0

for p in p_values:
    for d in d_values:
        for q in q_values:
            current += 1
            try:
                model = ARIMA(train, order=(p, d, q))
                fitted = model.fit()
                aic = fitted.aic
                results.append({
                    'p': p, 'd': d, 'q': q,
                    'AIC': aic,
                    'BIC': fitted.bic
                })
                
                if aic < best_aic:
                    best_aic = aic
                    best_params = (p, d, q)
                
                if current % 5 == 0:
                    print(f"  Progress: {current}/{total_combinations} combinations tested...")
                    
            except:
                continue

print(f"\n  ✓ Grid search complete!")
print(f"  Best parameters: ARIMA{best_params}")
print(f"  Best AIC: {best_aic:.2f}")

# Display top 5 models
results_df = pd.DataFrame(results).sort_values('AIC').head(5)
print("\n  Top 5 Models by AIC:")
print(results_df.to_string(index=False))

# ============================================================================
# 5.4 BUILD ARIMA MODEL WITH BEST PARAMETERS
# ============================================================================
print(f"\n[5.4] Building ARIMA{best_params} Model...")

arima_model = ARIMA(train, order=best_params)
arima_fit = arima_model.fit()

print(f"  ✓ Model trained successfully")
print(f"\n  Model Summary:")
print(arima_fit.summary())

# ============================================================================
# 5.5 MODEL DIAGNOSTICS
# ============================================================================
print("\n[5.5] Model Diagnostics...")

# Plot diagnostics
fig = arima_fit.plot_diagnostics(figsize=(15, 10))
plt.tight_layout()
plt.savefig('visualizations/15_arima_diagnostics.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/15_arima_diagnostics.png")
plt.close()

# Residual analysis
residuals = arima_fit.resid

print("\n  Residual Statistics:")
print(f"    Mean: ${residuals.mean():.2f}")
print(f"    Std Dev: ${residuals.std():.2f}")
print(f"    Min: ${residuals.min():.2f}")
print(f"    Max: ${residuals.max():.2f}")

# Test for residual autocorrelation (Ljung-Box test)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
print(f"\n  Ljung-Box Test (Residual Autocorrelation):")
print(f"    p-values > 0.05 indicate no significant autocorrelation")
print(lb_test)

# ============================================================================
# 5.6 MAKE PREDICTIONS ON TEST SET
# ============================================================================
print("\n[5.6] Making Predictions on Test Set...")

# Forecast
arima_forecast = arima_fit.forecast(steps=len(test))

print(f"  ✓ Generated {len(test)} forecasts")
print(f"\n  First 5 predictions:")
for i in range(min(5, len(test))):
    print(f"    {test.index[i].strftime('%Y-%m')}: ${arima_forecast.iloc[i]:,.2f} (Actual: ${test.iloc[i]:,.2f})")

# ============================================================================
# 5.7 CALCULATE PERFORMANCE METRICS
# ============================================================================
print("\n[5.7] Calculating Performance Metrics...")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(test, arima_forecast)
rmse = np.sqrt(mean_squared_error(test, arima_forecast))
mape = np.mean(np.abs((test - arima_forecast) / test)) * 100
r2 = r2_score(test, arima_forecast)

print(f"\n  ARIMA{best_params} Performance:")
print(f"    MAE (Mean Absolute Error): ${mae:,.2f}")
print(f"    RMSE (Root Mean Squared Error): ${rmse:,.2f}")
print(f"    MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
print(f"    R² (R-squared): {r2:.4f}")

# Interpretation
print(f"\n  Performance Interpretation:")
if mape < 10:
    print(f"    MAPE < 10%: ⭐⭐⭐⭐⭐ Excellent accuracy!")
elif mape < 20:
    print(f"    MAPE 10-20%: ⭐⭐⭐⭐ Good accuracy")
elif mape < 30:
    print(f"    MAPE 20-30%: ⭐⭐⭐ Acceptable accuracy")
else:
    print(f"    MAPE > 30%: ⭐⭐ Needs improvement")

# ============================================================================
# 5.8 VISUALIZE PREDICTIONS
# ============================================================================
print("\n[5.8] Visualizing Predictions...")

fig, ax = plt.subplots(figsize=(15, 7))

# Plot training data
ax.plot(train.index, train.values, label='Training Data', 
        color='#2E86AB', linewidth=2, marker='o', markersize=4)

# Plot actual test data
ax.plot(test.index, test.values, label='Actual Test Data', 
        color='#0F4C5C', linewidth=2.5, marker='o', markersize=6)

# Plot predictions
ax.plot(test.index, arima_forecast.values, label=f'ARIMA{best_params} Forecast', 
        color='#E71D36', linestyle='--', linewidth=2, marker='s', markersize=5)

# Add confidence interval
forecast_obj = arima_fit.get_forecast(steps=len(test))
forecast_ci = forecast_obj.conf_int()
ax.fill_between(test.index, 
                forecast_ci.iloc[:, 0], 
                forecast_ci.iloc[:, 1], 
                color='#E71D36', alpha=0.2, label='95% Confidence Interval')

ax.set_title(f'ARIMA{best_params} Model - Sales Forecast', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/16_arima_forecast.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/16_arima_forecast.png")
plt.close()

# ============================================================================
# 5.9 FORECAST ERRORS ANALYSIS
# ============================================================================
print("\n[5.9] Analyzing Forecast Errors...")

errors = test.values - arima_forecast.values
errors_pct = (errors / test.values) * 100

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Error distribution
axes[0].hist(errors, bins=15, color='#E71D36', alpha=0.7, edgecolor='black')
axes[0].axvline(0, color='black', linestyle='--', linewidth=2)
axes[0].set_title('Forecast Error Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Error ($)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# Error over time
axes[1].plot(test.index, errors, marker='o', color='#E71D36', linewidth=2)
axes[1].axhline(0, color='black', linestyle='--', linewidth=2)
axes[1].set_title('Forecast Errors Over Time', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_ylabel('Error ($)', fontsize=11)
axes[1].grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('visualizations/17_arima_errors.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/17_arima_errors.png")
plt.close()

# ============================================================================
# 5.10 SAVE RESULTS
# ============================================================================
print("\n[5.10] Saving Results...")

# Save predictions
results_df = pd.DataFrame({
    'Actual': test.values,
    'Predicted': arima_forecast.values,
    'Error': errors,
    'Error_Pct': errors_pct
}, index=test.index)

results_df.to_csv('arima_predictions.csv')
print("  ✓ Saved: arima_predictions.csv")

# Save model performance
performance = {
    'Model': f'ARIMA{best_params}',
    'MAE': mae,
    'RMSE': rmse,
    'MAPE': mape,
    'R2': r2,
    'AIC': arima_fit.aic,
    'BIC': arima_fit.bic
}

performance_df = pd.DataFrame([performance])
performance_df.to_csv('arima_performance.csv', index=False)
print("  ✓ Saved: arima_performance.csv")

# Save model parameters
param_search_df = pd.DataFrame(results)
param_search_df.to_csv('arima_parameter_search.csv', index=False)
print("  ✓ Saved: arima_parameter_search.csv")

# ============================================================================
# 5.11 SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ARIMA MODEL BUILDING COMPLETE!")
print("="*80)

print(f"\nBest Model: ARIMA{best_params}")
print(f"\nPerformance Metrics:")
print(f"  MAE: ${mae:,.2f}")
print(f"  RMSE: ${rmse:,.2f}")
print(f"  MAPE: {mape:.2f}%")
print(f"  R²: {r2:.4f}")

print(f"\nFiles Created:")
print(f"  • arima_predictions.csv")
print(f"  • arima_performance.csv")
print(f"  • arima_parameter_search.csv")
print(f"  • visualizations/14_arima_acf_pacf.png")
print(f"  • visualizations/15_arima_diagnostics.png")
print(f"  • visualizations/16_arima_forecast.png")
print(f"  • visualizations/17_arima_errors.png")

print(f"\nReady for next step: Holt-Winters Model")
print("="*80)
