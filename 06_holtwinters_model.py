"""
STEP 6: HOLT-WINTERS (EXPONENTIAL SMOOTHING) MODEL
===================================================
Project: Monthly Sales Forecast and Trend Analysis
Dataset: Superstore Sales
File: 06_holtwinters_model.py
===================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("STEP 6: HOLT-WINTERS (EXPONENTIAL SMOOTHING) MODEL")
print("="*80)

# ============================================================================
# 6.1 LOAD TRAIN AND TEST DATA
# ============================================================================
print("\n[6.1] Loading Train and Test Data...")

train = pd.read_csv('data_train.csv', index_col=0, parse_dates=True)['Sales']
test = pd.read_csv('data_test.csv', index_col=0, parse_dates=True)['Sales']

print(f"✓ Training data: {len(train)} months")
print(f"✓ Test data: {len(test)} months")

# ============================================================================
# 6.2 CHECK DATA REQUIREMENTS
# ============================================================================
print("\n[6.2] Checking Data Requirements...")

seasonal_period = 12  # Monthly data with yearly seasonality

if len(train) < 2 * seasonal_period:
    print(f"  ⚠ Warning: Training data has only {len(train)} months")
    print(f"    Seasonal Holt-Winters requires at least {2 * seasonal_period} months")
    print(f"    (2 complete seasonal cycles)")
    use_seasonal = False
else:
    print(f"  ✓ Sufficient data for seasonal model ({len(train)} months)")
    use_seasonal = True

# ============================================================================
# 6.3 BUILD MULTIPLE MODEL VARIANTS
# ============================================================================
print("\n[6.3] Building Multiple Holt-Winters Variants...")

models = {}
forecasts = {}
performance = []

# Model 1: Simple Exponential Smoothing (no trend, no seasonality)
print("\n  [1] Simple Exponential Smoothing...")
try:
    model_simple = ExponentialSmoothing(train, trend=None, seasonal=None)
    fit_simple = model_simple.fit()
    forecast_simple = fit_simple.forecast(steps=len(test))
    models['Simple'] = fit_simple
    forecasts['Simple'] = forecast_simple
    print("      ✓ Model fitted successfully")
except Exception as e:
    print(f"      ✗ Error: {e}")

# Model 2: Holt's Linear (trend, no seasonality)
print("\n  [2] Holt's Linear Trend...")
try:
    model_linear = ExponentialSmoothing(train, trend='add', seasonal=None)
    fit_linear = model_linear.fit()
    forecast_linear = fit_linear.forecast(steps=len(test))
    models['Linear_Trend'] = fit_linear
    forecasts['Linear_Trend'] = forecast_linear
    print("      ✓ Model fitted successfully")
except Exception as e:
    print(f"      ✗ Error: {e}")

# Model 3: Additive Seasonal (if enough data)
if use_seasonal:
    print("\n  [3] Additive Seasonal...")
    try:
        model_add = ExponentialSmoothing(train, 
                                        trend='add', 
                                        seasonal='add', 
                                        seasonal_periods=seasonal_period)
        fit_add = model_add.fit()
        forecast_add = fit_add.forecast(steps=len(test))
        models['Additive'] = fit_add
        forecasts['Additive'] = forecast_add
        print("      ✓ Model fitted successfully")
        print(f"        AIC: {fit_add.aic:.2f}")
    except Exception as e:
        print(f"      ✗ Error: {e}")
    
    # Model 4: Multiplicative Seasonal
    print("\n  [4] Multiplicative Seasonal...")
    try:
        # Check for zero or negative values
        if (train <= 0).any():
            print("      ⚠ Skipping: Zero or negative values detected")
        else:
            model_mult = ExponentialSmoothing(train, 
                                             trend='add', 
                                             seasonal='mul', 
                                             seasonal_periods=seasonal_period)
            fit_mult = model_mult.fit()
            forecast_mult = fit_mult.forecast(steps=len(test))
            models['Multiplicative'] = fit_mult
            forecasts['Multiplicative'] = forecast_mult
            print("      ✓ Model fitted successfully")
            print(f"        AIC: {fit_mult.aic:.2f}")
    except Exception as e:
        print(f"      ✗ Error: {e}")

print(f"\n  Total models built: {len(models)}")

# ============================================================================
# 6.4 EVALUATE ALL MODELS
# ============================================================================
print("\n[6.4] Evaluating All Models...")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

for name, forecast in forecasts.items():
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    r2 = r2_score(test, forecast)
    
    performance.append({
        'Model': name,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'AIC': models[name].aic if hasattr(models[name], 'aic') else np.nan
    })
    
    print(f"\n  {name} Model:")
    print(f"    MAE: ${mae:,.2f}")
    print(f"    RMSE: ${rmse:,.2f}")
    print(f"    MAPE: {mape:.2f}%")
    print(f"    R²: {r2:.4f}")

# Find best model
performance_df = pd.DataFrame(performance)
best_model_name = performance_df.loc[performance_df['MAPE'].idxmin(), 'Model']
print(f"\n  🏆 Best Model: {best_model_name} (lowest MAPE)")

# ============================================================================
# 6.5 DETAILED ANALYSIS OF BEST MODEL
# ============================================================================
print(f"\n[6.5] Detailed Analysis of Best Model ({best_model_name})...")

best_model = models[best_model_name]
best_forecast = forecasts[best_model_name]

print(f"\n  Model Parameters:")
print(f"    Smoothing Level (alpha): {best_model.params['smoothing_level']:.4f}")
if 'smoothing_trend' in best_model.params:
    print(f"    Smoothing Trend (beta): {best_model.params['smoothing_trend']:.4f}")
if 'smoothing_seasonal' in best_model.params:
    print(f"    Smoothing Seasonal (gamma): {best_model.params['smoothing_seasonal']:.4f}")

# Component analysis
if hasattr(best_model, 'level'):
    print(f"\n  Components:")
    print(f"    Final Level: ${best_model.level[-1]:,.2f}")
    if hasattr(best_model, 'trend') and best_model.trend is not None:
        print(f"    Final Trend: ${best_model.trend[-1]:,.2f}")
    if hasattr(best_model, 'season') and best_model.season is not None:
        print(f"    Seasonal components available: Yes")

# ============================================================================
# 6.6 VISUALIZE BEST MODEL FORECAST
# ============================================================================
print(f"\n[6.6] Visualizing {best_model_name} Model Forecast...")

fig, ax = plt.subplots(figsize=(15, 7))

# Plot training data
ax.plot(train.index, train.values, label='Training Data', 
        color='#2E86AB', linewidth=2, marker='o', markersize=4)

# Plot actual test data
ax.plot(test.index, test.values, label='Actual Test Data', 
        color='#0F4C5C', linewidth=2.5, marker='o', markersize=6)

# Plot best forecast
ax.plot(test.index, best_forecast.values, label=f'{best_model_name} Forecast', 
        color='#F18F01', linestyle='--', linewidth=2, marker='s', markersize=5)

ax.set_title(f'Holt-Winters ({best_model_name}) Model - Sales Forecast', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/18_holtwinters_forecast.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/18_holtwinters_forecast.png")
plt.close()

# ============================================================================
# 6.7 COMPARE ALL MODEL FORECASTS
# ============================================================================
print("\n[6.7] Comparing All Model Forecasts...")

fig, ax = plt.subplots(figsize=(15, 7))

# Plot training and actual test data
ax.plot(train.index, train.values, label='Training Data', 
        color='#2E86AB', linewidth=2, marker='o', markersize=4, alpha=0.5)
ax.plot(test.index, test.values, label='Actual Test Data', 
        color='#0F4C5C', linewidth=3, marker='o', markersize=7)

# Plot all forecasts
colors = ['#E71D36', '#F18F01', '#9B59B6', '#27AE60']
for idx, (name, forecast) in enumerate(forecasts.items()):
    ax.plot(test.index, forecast.values, label=f'{name} Forecast', 
            linestyle='--', linewidth=2, marker='s', markersize=4, 
            color=colors[idx % len(colors)])

ax.set_title('Holt-Winters Models Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/19_holtwinters_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/19_holtwinters_comparison.png")
plt.close()

# ============================================================================
# 6.8 RESIDUAL ANALYSIS
# ============================================================================
print("\n[6.8] Residual Analysis...")

residuals = test.values - best_forecast.values

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Residuals over time
axes[0, 0].plot(test.index, residuals, marker='o', color='#F18F01', linewidth=2)
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=2)
axes[0, 0].set_title('Residuals Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date', fontsize=11)
axes[0, 0].set_ylabel('Residual ($)', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Residual distribution
axes[0, 1].hist(residuals, bins=15, color='#F18F01', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=2)
axes[0, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Residual ($)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Actual vs Predicted
axes[1, 0].scatter(test.values, best_forecast.values, alpha=0.6, color='#F18F01', s=100)
min_val = min(test.min(), best_forecast.min())
max_val = max(test.max(), best_forecast.max())
axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
axes[1, 0].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Actual Sales ($)', fontsize=11)
axes[1, 0].set_ylabel('Predicted Sales ($)', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/20_holtwinters_residuals.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/20_holtwinters_residuals.png")
plt.close()

print(f"\n  Residual Statistics:")
print(f"    Mean: ${residuals.mean():,.2f}")
print(f"    Std Dev: ${residuals.std():,.2f}")
print(f"    Min: ${residuals.min():,.2f}")
print(f"    Max: ${residuals.max():,.2f}")

# ============================================================================
# 6.9 SAVE RESULTS
# ============================================================================
print("\n[6.9] Saving Results...")

# Save best model predictions
results_df = pd.DataFrame({
    'Actual': test.values,
    'Predicted': best_forecast.values,
    'Error': residuals,
    'Error_Pct': (residuals / test.values) * 100
}, index=test.index)

results_df.to_csv('holtwinters_predictions.csv')
print("  ✓ Saved: holtwinters_predictions.csv")

# Save all model performance
performance_df.to_csv('holtwinters_performance.csv', index=False)
print("  ✓ Saved: holtwinters_performance.csv")

# Save all forecasts
all_forecasts = pd.DataFrame({name: fc.values for name, fc in forecasts.items()}, 
                             index=test.index)
all_forecasts.insert(0, 'Actual', test.values)
all_forecasts.to_csv('holtwinters_all_forecasts.csv')
print("  ✓ Saved: holtwinters_all_forecasts.csv")

# ============================================================================
# 6.10 SUMMARY
# ============================================================================
print("\n" + "="*80)
print("HOLT-WINTERS MODEL BUILDING COMPLETE!")
print("="*80)

print(f"\nBest Model: {best_model_name}")
best_perf = performance_df[performance_df['Model'] == best_model_name].iloc[0]

print(f"\nPerformance Metrics:")
print(f"  MAE: ${best_perf['MAE']:,.2f}")
print(f"  RMSE: ${best_perf['RMSE']:,.2f}")
print(f"  MAPE: {best_perf['MAPE']:.2f}%")
print(f"  R²: {best_perf['R2']:.4f}")

print(f"\nAll Models Comparison:")
print(performance_df.to_string(index=False))

print(f"\nFiles Created:")
print(f"  • holtwinters_predictions.csv")
print(f"  • holtwinters_performance.csv")
print(f"  • holtwinters_all_forecasts.csv")
print(f"  • visualizations/18_holtwinters_forecast.png")
print(f"  • visualizations/19_holtwinters_comparison.png")
print(f"  • visualizations/20_holtwinters_residuals.png")

print(f"\nReady for next step: Prophet Model")
print("="*80)
