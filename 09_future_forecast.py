"""
STEP 9: FUTURE FORECASTING (NEXT 6 MONTHS)
===========================================
Project: Monthly Sales Forecast and Trend Analysis
Dataset: Superstore Sales
File: 09_future_forecast.py
===========================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("STEP 9: FUTURE FORECASTING (NEXT 6 MONTHS)")
print("="*80)

# ============================================================================
# 9.1 LOAD BEST MODEL INFORMATION
# ============================================================================
print("\n[9.1] Loading Best Model Information...")

try:
    best_model_info = pd.read_csv('best_model_selection.csv')
    best_model_name = best_model_info['Best_Model'].iloc[0]
    print(f"  ✓ Best model: {best_model_name}")
    print(f"  MAPE: {best_model_info['Best_MAPE'].iloc[0]:.2f}%")
except FileNotFoundError:
    print("  ⚠ Best model information not found. Using Prophet as default.")
    best_model_name = "Prophet"

# ============================================================================
# 9.2 LOAD FULL DATASET
# ============================================================================
print("\n[9.2] Loading Full Dataset for Retraining...")

df_monthly = pd.read_csv('data_with_features.csv', index_col=0, parse_dates=True)
print(f"  ✓ Loaded {len(df_monthly)} months of historical data")
print(f"  Date range: {df_monthly.index.min().strftime('%Y-%m')} to {df_monthly.index.max().strftime('%Y-%m')}")

# ============================================================================
# 9.3 SET FORECAST PARAMETERS
# ============================================================================
print("\n[9.3] Setting Forecast Parameters...")

FORECAST_PERIODS = 6  # Next 6 months
print(f"  Forecast horizon: {FORECAST_PERIODS} months")

# ============================================================================
# 9.4 RETRAIN BEST MODEL ON FULL DATASET
# ============================================================================
print(f"\n[9.4] Retraining {best_model_name} on Full Dataset...")

future_forecast = None
confidence_lower = None
confidence_upper = None

if best_model_name == "Prophet":
    print("  Using Prophet model...")
    try:
        from prophet import Prophet
        
        # Prepare data
        prophet_df = pd.DataFrame({
            'ds': df_monthly.index,
            'y': df_monthly['Sales'].values
        })
        
        # Build and train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        print("  Training model on full dataset...")
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=FORECAST_PERIODS, freq='M')
        
        # Make predictions
        forecast_df = model.predict(future)
        
        # Extract future predictions
        future_forecast = forecast_df['yhat'].tail(FORECAST_PERIODS).values
        confidence_lower = forecast_df['yhat_lower'].tail(FORECAST_PERIODS).values
        confidence_upper = forecast_df['yhat_upper'].tail(FORECAST_PERIODS).values
        
        # Create future dates
        last_date = df_monthly.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                     periods=FORECAST_PERIODS, freq='M')
        
        print("  ✓ Prophet model trained and forecast generated")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        
elif best_model_name == "ARIMA":
    print("  Using ARIMA model...")
    try:
        from statsmodels.tsa.arima.model import ARIMA
        
        # Load ARIMA parameters from previous run
        arima_perf = pd.read_csv('arima_performance.csv')
        model_name = arima_perf['Model'].iloc[0]
        # Extract (p,d,q) from model name like "ARIMA(1,1,1)"
        import re
        match = re.search(r'\((\d+),\s*(\d+),\s*(\d+)\)', model_name)
        if match:
            p, d, q = map(int, match.groups())
        else:
            p, d, q = 1, 1, 1  # Default
        
        print(f"  Using ARIMA({p},{d},{q})...")
        
        # Build and train model
        model = ARIMA(df_monthly['Sales'], order=(p, d, q))
        fitted_model = model.fit()
        
        # Forecast
        forecast_obj = fitted_model.get_forecast(steps=FORECAST_PERIODS)
        future_forecast = forecast_obj.predicted_mean.values
        forecast_ci = forecast_obj.conf_int()
        confidence_lower = forecast_ci.iloc[:, 0].values
        confidence_upper = forecast_ci.iloc[:, 1].values
        
        # Create future dates
        last_date = df_monthly.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                     periods=FORECAST_PERIODS, freq='M')
        
        print("  ✓ ARIMA model trained and forecast generated")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        
elif best_model_name == "Holt-Winters":
    print("  Using Holt-Winters model...")
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Build model
        model = ExponentialSmoothing(
            df_monthly['Sales'],
            trend='add',
            seasonal='add',
            seasonal_periods=12
        )
        
        fitted_model = model.fit()
        
        # Forecast
        future_forecast = fitted_model.forecast(steps=FORECAST_PERIODS).values
        
        # Holt-Winters doesn't provide confidence intervals by default
        # Estimate using historical error
        historical_fit = fitted_model.fittedvalues
        residuals = df_monthly['Sales'] - historical_fit
        std_error = residuals.std()
        
        confidence_lower = future_forecast - 1.96 * std_error
        confidence_upper = future_forecast + 1.96 * std_error
        
        # Create future dates
        last_date = df_monthly.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                     periods=FORECAST_PERIODS, freq='M')
        
        print("  ✓ Holt-Winters model trained and forecast generated")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

# ============================================================================
# 9.5 DISPLAY FUTURE PREDICTIONS
# ============================================================================
if future_forecast is not None:
    print("\n[9.5] Future Sales Predictions...")
    
    print("\n" + "="*80)
    print(f"NEXT {FORECAST_PERIODS} MONTHS FORECAST")
    print("="*80)
    
    future_df = pd.DataFrame({
        'Month': [d.strftime('%Y-%m') for d in future_dates],
        'Predicted_Sales': future_forecast,
        'Lower_Bound': confidence_lower,
        'Upper_Bound': confidence_upper
    })
    
    print(future_df.to_string(index=False))
    print("="*80)
    
    # Summary statistics
    total_forecast = future_forecast.sum()
    avg_forecast = future_forecast.mean()
    
    print(f"\nForecast Summary:")
    print(f"  Total Sales (6 months): ${total_forecast:,.2f}")
    print(f"  Average Monthly Sales: ${avg_forecast:,.2f}")
    print(f"  Minimum Predicted: ${future_forecast.min():,.2f}")
    print(f"  Maximum Predicted: ${future_forecast.max():,.2f}")
    
    # Compare with historical average
    historical_avg = df_monthly['Sales'].mean()
    growth = ((avg_forecast - historical_avg) / historical_avg) * 100
    
    print(f"\n  Historical Average: ${historical_avg:,.2f}")
    print(f"  Forecast vs Historical: {growth:+.2f}%")

# ============================================================================
# 9.6 VISUALIZE FUTURE FORECAST
# ============================================================================
if future_forecast is not None:
    print("\n[9.6] Visualizing Future Forecast...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot historical data
    ax.plot(df_monthly.index, df_monthly['Sales'], label='Historical Data', 
            color='#2E86AB', linewidth=2, marker='o', markersize=5)
    
    # Plot future forecast
    ax.plot(future_dates, future_forecast, label=f'{best_model_name} Forecast (Next 6 Months)', 
            color='#E71D36', linewidth=3, marker='s', markersize=7, linestyle='--')
    
    # Plot confidence interval
    ax.fill_between(future_dates, confidence_lower, confidence_upper, 
                    color='#E71D36', alpha=0.2, label='95% Confidence Interval')
    
    # Add vertical line at forecast start
    ax.axvline(x=df_monthly.index[-1], color='black', linestyle=':', linewidth=2, alpha=0.7)
    ax.text(df_monthly.index[-1], ax.get_ylim()[1]*0.95, 'Forecast Start', 
            ha='right', va='top', fontsize=11, fontweight='bold')
    
    ax.set_title(f'Sales Forecast - Next {FORECAST_PERIODS} Months ({best_model_name} Model)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/28_future_forecast.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: visualizations/28_future_forecast.png")
    plt.close()

# ============================================================================
# 9.7 MONTHLY BREAKDOWN VISUALIZATION
# ============================================================================
if future_forecast is not None:
    print("\n[9.7] Creating Monthly Breakdown Visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    months = [d.strftime('%b\n%Y') for d in future_dates]
    x_pos = np.arange(len(months))
    
    # Bar chart with error bars
    bars = ax.bar(x_pos, future_forecast, color='#2E86AB', alpha=0.7, 
                  edgecolor='black', linewidth=1.5)
    
    # Error bars for confidence interval
    errors = [future_forecast - confidence_lower, confidence_upper - future_forecast]
    ax.errorbar(x_pos, future_forecast, yerr=errors, fmt='none', 
               color='black', capsize=5, capthick=2, linewidth=2)
    
    # Value labels on bars
    for i, (bar, val) in enumerate(zip(bars, future_forecast)):
        ax.text(bar.get_x() + bar.get_width()/2, val, f'${val:,.0f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(months)
    ax.set_title('Monthly Sales Forecast Breakdown', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Predicted Sales ($)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('visualizations/29_monthly_breakdown.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: visualizations/29_monthly_breakdown.png")
    plt.close()

# ============================================================================
# 9.8 SAVE FUTURE FORECAST
# ============================================================================
if future_forecast is not None:
    print("\n[9.8] Saving Future Forecast...")
    
    future_df.to_csv('future_forecast_6months.csv', index=False)
    print("  ✓ Saved: future_forecast_6months.csv")
    
    # Save with additional details
    detailed_forecast = pd.DataFrame({
        'Month': [d.strftime('%Y-%m') for d in future_dates],
        'Date': future_dates,
        'Predicted_Sales': future_forecast,
        'Lower_Bound_95': confidence_lower,
        'Upper_Bound_95': confidence_upper,
        'Confidence_Range': confidence_upper - confidence_lower,
        'Model_Used': best_model_name
    })
    
    detailed_forecast.to_csv('future_forecast_detailed.csv', index=False)
    print("  ✓ Saved: future_forecast_detailed.csv")

# ============================================================================
# 9.9 BUSINESS INSIGHTS
# ============================================================================
if future_forecast is not None:
    print("\n[9.9] Generating Business Insights...")
    
    print("\n" + "="*80)
    print("BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("="*80)
    
    # Identify peak and low months
    peak_month_idx = future_forecast.argmax()
    low_month_idx = future_forecast.argmin()
    
    print(f"\n📈 Peak Sales Expected:")
    print(f"  Month: {future_dates[peak_month_idx].strftime('%B %Y')}")
    print(f"  Predicted Sales: ${future_forecast[peak_month_idx]:,.2f}")
    
    print(f"\n📉 Lowest Sales Expected:")
    print(f"  Month: {future_dates[low_month_idx].strftime('%B %Y')}")
    print(f"  Predicted Sales: ${future_forecast[low_month_idx]:,.2f}")
    
    # Growth trend
    first_month = future_forecast[0]
    last_month = future_forecast[-1]
    trend_growth = ((last_month - first_month) / first_month) * 100
    
    print(f"\n📊 Trend Analysis:")
    if trend_growth > 0:
        print(f"  Trend: UPWARD ↗")
        print(f"  Growth: +{trend_growth:.2f}% from first to last forecast month")
    else:
        print(f"  Trend: DOWNWARD ↘")
        print(f"  Decline: {trend_growth:.2f}% from first to last forecast month")
    
    print(f"\n💡 Recommendations:")
    print(f"  1. Prepare inventory for {future_dates[peak_month_idx].strftime('%B %Y')} (peak month)")
    print(f"  2. Plan promotions for {future_dates[low_month_idx].strftime('%B %Y')} (low month)")
    print(f"  3. Total budget needed for next 6 months: ~${total_forecast:,.2f}")
    print(f"  4. Monitor actual sales vs. forecast and adjust monthly")

# ============================================================================
# 9.10 SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FUTURE FORECASTING COMPLETE!")
print("="*80)

if future_forecast is not None:
    print(f"\nModel Used: {best_model_name}")
    print(f"Forecast Horizon: {FORECAST_PERIODS} months")
    print(f"Total Predicted Sales: ${total_forecast:,.2f}")
    print(f"Average Monthly Sales: ${avg_forecast:,.2f}")
    
    print(f"\nFiles Created:")
    print(f"  • future_forecast_6months.csv")
    print(f"  • future_forecast_detailed.csv")
    print(f"  • visualizations/28_future_forecast.png")
    print(f"  • visualizations/29_monthly_breakdown.png")
    
    print(f"\nReady for next step: Final Report Generation")
else:
    print("\n⚠ Future forecast could not be generated")
    print("  Please check previous steps and model availability")

print("="*80)
