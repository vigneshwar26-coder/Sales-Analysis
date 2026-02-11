"""
STEP 3: TIME SERIES DECOMPOSITION
==================================
Project: Monthly Sales Forecast and Trend Analysis
Dataset: Superstore Sales
File: 03_time_series_decomposition.py
==================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("STEP 3: TIME SERIES DECOMPOSITION")
print("="*80)

# ============================================================================
# 3.1 LOAD PROCESSED DATA
# ============================================================================
print("\n[3.1] Loading Processed Data...")

df_monthly = pd.read_csv('data_with_features.csv', index_col=0, parse_dates=True)
print(f"✓ Loaded {len(df_monthly)} months of data")

# ============================================================================
# 3.2 CHECK DATA REQUIREMENTS
# ============================================================================
print("\n[3.2] Checking Data Requirements...")

if len(df_monthly) < 24:
    print(f"⚠ Warning: Only {len(df_monthly)} months of data available")
    print("  Decomposition requires at least 24 months (2 complete cycles)")
    print("  Results may be less reliable")
    period = 12
else:
    print(f"✓ Sufficient data: {len(df_monthly)} months")
    period = 12

# ============================================================================
# 3.3 PERFORM DECOMPOSITION
# ============================================================================
print("\n[3.3] Performing Time Series Decomposition...")

try:
    # Additive decomposition
    print("  Attempting ADDITIVE decomposition...")
    decomposition_add = seasonal_decompose(df_monthly['Sales'], 
                                          model='additive', 
                                          period=period,
                                          extrapolate_trend='freq')
    print("  ✓ Additive decomposition complete")
    
    # Multiplicative decomposition
    print("  Attempting MULTIPLICATIVE decomposition...")
    # Check for zero or negative values
    if (df_monthly['Sales'] <= 0).any():
        print("  ⚠ Warning: Zero or negative values found. Skipping multiplicative decomposition.")
        decomposition_mult = None
    else:
        decomposition_mult = seasonal_decompose(df_monthly['Sales'], 
                                               model='multiplicative', 
                                               period=period,
                                               extrapolate_trend='freq')
        print("  ✓ Multiplicative decomposition complete")
    
except Exception as e:
    print(f"  ✗ Error during decomposition: {e}")
    decomposition_add = None
    decomposition_mult = None

# ============================================================================
# 3.4 VISUALIZE ADDITIVE DECOMPOSITION
# ============================================================================
if decomposition_add is not None:
    print("\n[3.4] Visualizing Additive Decomposition...")
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Observed
    decomposition_add.observed.plot(ax=axes[0], color='#2E86AB', linewidth=2)
    axes[0].set_ylabel('Observed', fontsize=12, fontweight='bold')
    axes[0].set_title('Time Series Decomposition (Additive Model)', 
                     fontsize=16, fontweight='bold', pad=20)
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Trend
    decomposition_add.trend.plot(ax=axes[1], color='#A23B72', linewidth=2)
    axes[1].set_ylabel('Trend', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Seasonal
    decomposition_add.seasonal.plot(ax=axes[2], color='#F18F01', linewidth=2)
    axes[2].set_ylabel('Seasonal', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Residual
    decomposition_add.resid.plot(ax=axes[3], color='#C73E1D', linewidth=2)
    axes[3].set_ylabel('Residual', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Date', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig('visualizations/08_decomposition_additive.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: visualizations/08_decomposition_additive.png")
    plt.close()

# ============================================================================
# 3.5 VISUALIZE MULTIPLICATIVE DECOMPOSITION
# ============================================================================
if decomposition_mult is not None:
    print("\n[3.5] Visualizing Multiplicative Decomposition...")
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    decomposition_mult.observed.plot(ax=axes[0], color='#2E86AB', linewidth=2)
    axes[0].set_ylabel('Observed', fontsize=12, fontweight='bold')
    axes[0].set_title('Time Series Decomposition (Multiplicative Model)', 
                     fontsize=16, fontweight='bold', pad=20)
    axes[0].grid(True, alpha=0.3)
    
    decomposition_mult.trend.plot(ax=axes[1], color='#A23B72', linewidth=2)
    axes[1].set_ylabel('Trend', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    decomposition_mult.seasonal.plot(ax=axes[2], color='#F18F01', linewidth=2)
    axes[2].set_ylabel('Seasonal', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    decomposition_mult.resid.plot(ax=axes[3], color='#C73E1D', linewidth=2)
    axes[3].set_ylabel('Residual', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('Date', fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/09_decomposition_multiplicative.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: visualizations/09_decomposition_multiplicative.png")
    plt.close()

# ============================================================================
# 3.6 ANALYZE TREND COMPONENT
# ============================================================================
if decomposition_add is not None:
    print("\n[3.6] Analyzing Trend Component...")
    
    trend = decomposition_add.trend.dropna()
    
    trend_start = trend.iloc[0]
    trend_end = trend.iloc[-1]
    trend_change = trend_end - trend_start
    trend_change_pct = (trend_change / trend_start) * 100
    
    print(f"\n  Trend Analysis:")
    print(f"    Starting Value: ${trend_start:,.2f}")
    print(f"    Ending Value: ${trend_end:,.2f}")
    print(f"    Absolute Change: ${trend_change:,.2f}")
    print(f"    Percentage Change: {trend_change_pct:.2f}%")
    
    if trend_change > 0:
        print(f"    Direction: ↗ UPWARD TREND")
    else:
        print(f"    Direction: ↘ DOWNWARD TREND")

# ============================================================================
# 3.7 ANALYZE SEASONAL COMPONENT
# ============================================================================
if decomposition_add is not None:
    print("\n[3.7] Analyzing Seasonal Component...")
    
    seasonal = decomposition_add.seasonal
    
    # Get one complete seasonal cycle (12 months)
    seasonal_pattern = seasonal.iloc[:12]
    
    print(f"\n  Seasonal Pattern (First 12 Months):")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i, (month, value) in enumerate(zip(months, seasonal_pattern)):
        print(f"    {month}: ${value:,.2f}")
    
    max_seasonal_month = seasonal_pattern.idxmax()
    min_seasonal_month = seasonal_pattern.idxmin()
    
    print(f"\n  Peak Seasonal Month: {months[max_seasonal_month.month - 1]}")
    print(f"  Lowest Seasonal Month: {months[min_seasonal_month.month - 1]}")
    print(f"  Seasonal Range: ${seasonal_pattern.max() - seasonal_pattern.min():,.2f}")

# ============================================================================
# 3.8 ANALYZE RESIDUALS
# ============================================================================
if decomposition_add is not None:
    print("\n[3.8] Analyzing Residuals...")
    
    residuals = decomposition_add.resid.dropna()
    
    print(f"\n  Residual Statistics:")
    print(f"    Mean: ${residuals.mean():,.2f}")
    print(f"    Std Dev: ${residuals.std():,.2f}")
    print(f"    Min: ${residuals.min():,.2f}")
    print(f"    Max: ${residuals.max():,.2f}")
    
    # Plot residual distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(residuals, bins=20, color='#C73E1D', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Residual Value', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/10_residual_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: visualizations/10_residual_analysis.png")
    plt.close()

# ============================================================================
# 3.9 STATIONARITY TEST (ADF TEST)
# ============================================================================
print("\n[3.9] Testing for Stationarity (Augmented Dickey-Fuller Test)...")

def perform_adf_test(series, name):
    """Perform Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna())
    
    print(f"\n  {name}:")
    print(f"    ADF Statistic: {result[0]:.4f}")
    print(f"    p-value: {result[1]:.4f}")
    print(f"    Critical Values:")
    for key, value in result[4].items():
        print(f"      {key}: {value:.4f}")
    
    if result[1] <= 0.05:
        print(f"    Result: ✓ STATIONARY (p-value ≤ 0.05)")
        return True
    else:
        print(f"    Result: ✗ NON-STATIONARY (p-value > 0.05)")
        return False

# Test original series
is_stationary = perform_adf_test(df_monthly['Sales'], "Original Sales Series")

# Test differenced series if non-stationary
if not is_stationary:
    df_monthly['Sales_Diff'] = df_monthly['Sales'].diff()
    perform_adf_test(df_monthly['Sales_Diff'], "First Differenced Series")

# Test detrended series
if decomposition_add is not None:
    detrended = df_monthly['Sales'] - decomposition_add.trend
    perform_adf_test(detrended, "Detrended Series (Sales - Trend)")

# ============================================================================
# 3.10 AUTOCORRELATION ANALYSIS
# ============================================================================
print("\n[3.10] Autocorrelation Analysis...")

fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# ACF Plot
plot_acf(df_monthly['Sales'].dropna(), lags=24, ax=axes[0], color='#2E86AB')
axes[0].set_title('Autocorrelation Function (ACF)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Lag', fontsize=11)
axes[0].set_ylabel('Correlation', fontsize=11)
axes[0].grid(True, alpha=0.3)

# PACF Plot
plot_pacf(df_monthly['Sales'].dropna(), lags=24, ax=axes[1], color='#A23B72')
axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Lag', fontsize=11)
axes[1].set_ylabel('Correlation', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/11_autocorrelation.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/11_autocorrelation.png")
plt.close()

# ============================================================================
# 3.11 SAVE DECOMPOSITION RESULTS
# ============================================================================
print("\n[3.11] Saving Decomposition Results...")

if decomposition_add is not None:
    decomp_df = pd.DataFrame({
        'Sales': df_monthly['Sales'],
        'Trend': decomposition_add.trend,
        'Seasonal': decomposition_add.seasonal,
        'Residual': decomposition_add.resid
    })
    
    decomp_df.to_csv('data_decomposed.csv')
    print("  ✓ Saved: data_decomposed.csv")

# ============================================================================
# 3.12 SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TIME SERIES DECOMPOSITION COMPLETE!")
print("="*80)

if decomposition_add is not None:
    print(f"\nDecomposition Summary:")
    print(f"  Model: Additive")
    print(f"  Period: {period} months")
    print(f"  Trend Direction: {'Upward ↗' if trend_change > 0 else 'Downward ↘'}")
    print(f"  Trend Change: {trend_change_pct:.2f}%")
    print(f"  Stationarity: {'Yes ✓' if is_stationary else 'No ✗'}")

print(f"\nVisualizations Created:")
print(f"  8. Additive Decomposition")
if decomposition_mult is not None:
    print(f"  9. Multiplicative Decomposition")
print(f"  10. Residual Analysis")
print(f"  11. Autocorrelation Analysis")

print(f"\nReady for next step: Model Building")
print("="*80)
