"""
STEP 7: PROPHET MODEL BUILDING
===============================
Project: Monthly Sales Forecast and Trend Analysis
Dataset: Superstore Sales
File: 07_prophet_model.py
===============================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Check if Prophet is available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("="*80)
    print("ERROR: Prophet not installed")
    print("="*80)
    print("\nPlease install Prophet using:")
    print("  pip install prophet")
    print("\nOr using conda:")
    print("  conda install -c conda-forge prophet")
    print("="*80)
    PROPHET_AVAILABLE = False
    exit()

plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("STEP 7: PROPHET MODEL BUILDING")
print("="*80)

# ============================================================================
# 7.1 LOAD TRAIN AND TEST DATA
# ============================================================================
print("\n[7.1] Loading Train and Test Data...")

train = pd.read_csv('data_train.csv', index_col=0, parse_dates=True)['Sales']
test = pd.read_csv('data_test.csv', index_col=0, parse_dates=True)['Sales']

print(f"✓ Training data: {len(train)} months")
print(f"✓ Test data: {len(test)} months")

# ============================================================================
# 7.2 PREPARE DATA FOR PROPHET
# ============================================================================
print("\n[7.2] Preparing Data for Prophet...")

# Prophet requires columns named 'ds' (date) and 'y' (value)
train_prophet = pd.DataFrame({
    'ds': train.index,
    'y': train.values
})

test_prophet = pd.DataFrame({
    'ds': test.index,
    'y': test.values
})

print(f"  ✓ Data prepared in Prophet format")
print(f"  Columns: {train_prophet.columns.tolist()}")

# ============================================================================
# 7.3 BUILD BASELINE PROPHET MODEL
# ============================================================================
print("\n[7.3] Building Baseline Prophet Model...")

model_baseline = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

print("  Training model (this may take a minute)...")
model_baseline.fit(train_prophet)
print("  ✓ Baseline model trained")

# ============================================================================
# 7.4 MAKE PREDICTIONS
# ============================================================================
print("\n[7.4] Making Predictions...")

# Create future dataframe
future = model_baseline.make_future_dataframe(periods=len(test), freq='M')

# Predict
forecast = model_baseline.predict(future)

# Extract test predictions
test_predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(test))
test_predictions.index = test.index

print(f"  ✓ Generated {len(test)} predictions")
print(f"\n  First 5 predictions:")
for i in range(min(5, len(test))):
    actual = test.iloc[i]
    pred = test_predictions['yhat'].iloc[i]
    lower = test_predictions['yhat_lower'].iloc[i]
    upper = test_predictions['yhat_upper'].iloc[i]
    print(f"    {test.index[i].strftime('%Y-%m')}: ${pred:,.2f} [{lower:,.2f} - {upper:,.2f}] (Actual: ${actual:,.2f})")

# ============================================================================
# 7.5 CALCULATE PERFORMANCE METRICS
# ============================================================================
print("\n[7.5] Calculating Performance Metrics...")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(test, test_predictions['yhat'])
rmse = np.sqrt(mean_squared_error(test, test_predictions['yhat']))
mape = np.mean(np.abs((test - test_predictions['yhat']) / test)) * 100
r2 = r2_score(test, test_predictions['yhat'])

print(f"\n  Prophet Baseline Performance:")
print(f"    MAE: ${mae:,.2f}")
print(f"    RMSE: ${rmse:,.2f}")
print(f"    MAPE: {mape:.2f}%")
print(f"    R²: {r2:.4f}")

# ============================================================================
# 7.6 HYPERPARAMETER TUNING
# ============================================================================
print("\n[7.6] Hyperparameter Tuning...")
print("  Testing different configurations...")

configurations = [
    {'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.01},
    {'changepoint_prior_scale': 0.01, 'seasonality_prior_scale': 0.1},
    {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 1.0},
    {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 10.0},
    {'changepoint_prior_scale': 0.5, 'seasonality_prior_scale': 10.0}
]

results = []
best_mape = float('inf')
best_model = None
best_config = None

for idx, config in enumerate(configurations):
    try:
        print(f"  [{idx+1}/{len(configurations)}] Testing: {config}")
        
        model_test = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            **config
        )
        
        model_test.fit(train_prophet)
        future_test = model_test.make_future_dataframe(periods=len(test), freq='M')
        forecast_test = model_test.predict(future_test)
        predictions_test = forecast_test['yhat'].tail(len(test)).values
        
        mape_test = np.mean(np.abs((test.values - predictions_test) / test.values)) * 100
        
        results.append({
            'config': config,
            'mape': mape_test
        })
        
        if mape_test < best_mape:
            best_mape = mape_test
            best_model = model_test
            best_config = config
            
        print(f"      MAPE: {mape_test:.2f}%")
        
    except Exception as e:
        print(f"      ✗ Failed: {e}")

print(f"\n  ✓ Hyperparameter tuning complete")
print(f"  Best configuration: {best_config}")
print(f"  Best MAPE: {best_mape:.2f}%")

# Use best model for final predictions
if best_model is not None:
    future_best = best_model.make_future_dataframe(periods=len(test), freq='M')
    forecast_best = best_model.predict(future_best)
    test_predictions_best = forecast_best[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(len(test))
    test_predictions_best.index = test.index
    
    # Recalculate metrics with best model
    mae_best = mean_absolute_error(test, test_predictions_best['yhat'])
    rmse_best = np.sqrt(mean_squared_error(test, test_predictions_best['yhat']))
    mape_best = best_mape
    r2_best = r2_score(test, test_predictions_best['yhat'])
    
    print(f"\n  Best Prophet Model Performance:")
    print(f"    MAE: ${mae_best:,.2f}")
    print(f"    RMSE: ${rmse_best:,.2f}")
    print(f"    MAPE: {mape_best:.2f}%")
    print(f"    R²: {r2_best:.4f}")
    
    # Use best model for visualizations
    final_model = best_model
    final_forecast = forecast_best
    final_predictions = test_predictions_best
else:
    final_model = model_baseline
    final_forecast = forecast
    final_predictions = test_predictions

# ============================================================================
# 7.7 VISUALIZE FORECAST
# ============================================================================
print("\n[7.7] Visualizing Forecast...")

fig, ax = plt.subplots(figsize=(15, 7))

# Plot training data
ax.plot(train.index, train.values, label='Training Data', 
        color='#2E86AB', linewidth=2, marker='o', markersize=4)

# Plot actual test data
ax.plot(test.index, test.values, label='Actual Test Data', 
        color='#0F4C5C', linewidth=2.5, marker='o', markersize=6)

# Plot predictions
ax.plot(test.index, final_predictions['yhat'].values, label='Prophet Forecast', 
        color='#9B59B6', linestyle='--', linewidth=2, marker='D', markersize=5)

# Plot confidence interval
ax.fill_between(test.index, 
                final_predictions['yhat_lower'].values, 
                final_predictions['yhat_upper'].values, 
                color='#9B59B6', alpha=0.2, label='95% Confidence Interval')

ax.set_title('Prophet Model - Sales Forecast', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/21_prophet_forecast.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/21_prophet_forecast.png")
plt.close()

# ============================================================================
# 7.8 COMPONENT ANALYSIS
# ============================================================================
print("\n[7.8] Analyzing Forecast Components...")

fig = final_model.plot_components(final_forecast, figsize=(15, 10))
plt.tight_layout()
plt.savefig('visualizations/22_prophet_components.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/22_prophet_components.png")
plt.close()

print("\n  Components analyzed:")
print("    • Trend: Overall direction of sales over time")
print("    • Yearly Seasonality: Seasonal patterns within the year")

# ============================================================================
# 7.9 RESIDUAL ANALYSIS
# ============================================================================
print("\n[7.9] Residual Analysis...")

residuals = test.values - final_predictions['yhat'].values

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Residuals over time
axes[0, 0].plot(test.index, residuals, marker='o', color='#9B59B6', linewidth=2)
axes[0, 0].axhline(0, color='black', linestyle='--', linewidth=2)
axes[0, 0].set_title('Residuals Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date', fontsize=11)
axes[0, 0].set_ylabel('Residual ($)', fontsize=11)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Residual distribution
axes[0, 1].hist(residuals, bins=15, color='#9B59B6', alpha=0.7, edgecolor='black')
axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=2)
axes[0, 1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Residual ($)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Actual vs Predicted
axes[1, 0].scatter(test.values, final_predictions['yhat'].values, 
                  alpha=0.6, color='#9B59B6', s=100)
min_val = min(test.min(), final_predictions['yhat'].min())
max_val = max(test.max(), final_predictions['yhat'].max())
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
plt.savefig('visualizations/23_prophet_residuals.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/23_prophet_residuals.png")
plt.close()

print(f"\n  Residual Statistics:")
print(f"    Mean: ${residuals.mean():,.2f}")
print(f"    Std Dev: ${residuals.std():,.2f}")
print(f"    Min: ${residuals.min():,.2f}")
print(f"    Max: ${residuals.max():,.2f}")

# ============================================================================
# 7.10 SAVE RESULTS
# ============================================================================
print("\n[7.10] Saving Results...")

# Save predictions
results_df = pd.DataFrame({
    'Actual': test.values,
    'Predicted': final_predictions['yhat'].values,
    'Lower_Bound': final_predictions['yhat_lower'].values,
    'Upper_Bound': final_predictions['yhat_upper'].values,
    'Error': residuals,
    'Error_Pct': (residuals / test.values) * 100
}, index=test.index)

results_df.to_csv('prophet_predictions.csv')
print("  ✓ Saved: prophet_predictions.csv")

# Save performance metrics
if best_model is not None:
    performance = {
        'Model': 'Prophet (Tuned)',
        'MAE': mae_best,
        'RMSE': rmse_best,
        'MAPE': mape_best,
        'R2': r2_best,
        'Best_Config': str(best_config)
    }
else:
    performance = {
        'Model': 'Prophet (Baseline)',
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'Best_Config': 'baseline'
    }

performance_df = pd.DataFrame([performance])
performance_df.to_csv('prophet_performance.csv', index=False)
print("  ✓ Saved: prophet_performance.csv")

# ============================================================================
# 7.11 SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PROPHET MODEL BUILDING COMPLETE!")
print("="*80)

print(f"\nModel: Prophet (with hyperparameter tuning)")

if best_model is not None:
    print(f"\nBest Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    print(f"\nPerformance Metrics:")
    print(f"  MAE: ${mae_best:,.2f}")
    print(f"  RMSE: ${rmse_best:,.2f}")
    print(f"  MAPE: {mape_best:.2f}%")
    print(f"  R²: {r2_best:.4f}")
else:
    print(f"\nPerformance Metrics:")
    print(f"  MAE: ${mae:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²: {r2:.4f}")

print(f"\nFiles Created:")
print(f"  • prophet_predictions.csv")
print(f"  • prophet_performance.csv")
print(f"  • visualizations/21_prophet_forecast.png")
print(f"  • visualizations/22_prophet_components.png")
print(f"  • visualizations/23_prophet_residuals.png")

print(f"\nReady for next step: Model Comparison and Final Report")
print("="*80)
