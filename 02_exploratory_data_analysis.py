"""
STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
========================================
Project: Monthly Sales Forecast and Trend Analysis
Dataset: Superstore Sales
File: 02_exploratory_data_analysis.py
========================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

# ============================================================================
# 2.1 LOAD PROCESSED DATA
# ============================================================================
print("\n[2.1] Loading Processed Data...")

df_monthly = pd.read_csv('data_processed_monthly.csv', index_col=0, parse_dates=True)
print(f"✓ Loaded {len(df_monthly)} months of data")

# ============================================================================
# 2.2 CREATE VISUALIZATIONS DIRECTORY
# ============================================================================
import os
os.makedirs('visualizations', exist_ok=True)
print("✓ Visualizations directory ready")

# ============================================================================
# 2.3 BASIC STATISTICS
# ============================================================================
print("\n[2.3] Basic Statistics...")

print("\nDescriptive Statistics:")
print(df_monthly['Sales'].describe())

print(f"\nAdditional Metrics:")
print(f"  Total Sales: ${df_monthly['Sales'].sum():,.2f}")
print(f"  Average Monthly Sales: ${df_monthly['Sales'].mean():,.2f}")
print(f"  Standard Deviation: ${df_monthly['Sales'].std():,.2f}")
print(f"  Coefficient of Variation: {(df_monthly['Sales'].std() / df_monthly['Sales'].mean() * 100):.2f}%")

# ============================================================================
# 2.4 SALES TREND VISUALIZATION
# ============================================================================
print("\n[2.4] Creating Sales Trend Visualization...")

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df_monthly.index, df_monthly['Sales'], marker='o', linewidth=2, markersize=5, color='#2E86AB')
ax.set_title('Monthly Sales Trend - Superstore Dataset', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/01_sales_trend.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/01_sales_trend.png")
plt.close()

# ============================================================================
# 2.5 MOVING AVERAGES
# ============================================================================
print("\n[2.5] Calculating Moving Averages...")

df_monthly['MA_3'] = df_monthly['Sales'].rolling(window=3).mean()
df_monthly['MA_6'] = df_monthly['Sales'].rolling(window=6).mean()
df_monthly['MA_12'] = df_monthly['Sales'].rolling(window=12).mean()

print("  ✓ 3-Month Moving Average calculated")
print("  ✓ 6-Month Moving Average calculated")
print("  ✓ 12-Month Moving Average calculated")

# Plot Moving Averages
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df_monthly.index, df_monthly['Sales'], label='Actual Sales', marker='o', 
        linewidth=2, markersize=4, color='#2E86AB', alpha=0.7)
ax.plot(df_monthly.index, df_monthly['MA_3'], label='3-Month MA', 
        linestyle='--', linewidth=2, color='#A23B72')
ax.plot(df_monthly.index, df_monthly['MA_6'], label='6-Month MA', 
        linestyle='--', linewidth=2, color='#F18F01')
ax.plot(df_monthly.index, df_monthly['MA_12'], label='12-Month MA', 
        linestyle='--', linewidth=2, color='#C73E1D')
ax.set_title('Sales with Moving Averages', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/02_moving_averages.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/02_moving_averages.png")
plt.close()

# ============================================================================
# 2.6 GROWTH RATES
# ============================================================================
print("\n[2.6] Calculating Growth Rates...")

# Month-over-Month Growth
df_monthly['MoM_Growth'] = df_monthly['Sales'].pct_change() * 100
print(f"  ✓ Month-over-Month Growth calculated")
print(f"    Average MoM Growth: {df_monthly['MoM_Growth'].mean():.2f}%")

# Year-over-Year Growth
df_monthly['YoY_Growth'] = df_monthly['Sales'].pct_change(12) * 100
print(f"  ✓ Year-over-Year Growth calculated")
print(f"    Average YoY Growth: {df_monthly['YoY_Growth'].dropna().mean():.2f}%")

# Plot Month-over-Month Growth
fig, ax = plt.subplots(figsize=(15, 6))
colors = ['green' if x > 0 else 'red' for x in df_monthly['MoM_Growth'].dropna()]
ax.bar(df_monthly.index[1:], df_monthly['MoM_Growth'].dropna(), color=colors, alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_title('Month-over-Month Growth Rate (%)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Growth Rate (%)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/03_mom_growth.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/03_mom_growth.png")
plt.close()

# Plot Year-over-Year Growth
if len(df_monthly) >= 13:
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ['green' if x > 0 else 'red' for x in df_monthly['YoY_Growth'].dropna()]
    ax.bar(df_monthly.index[12:], df_monthly['YoY_Growth'].dropna(), color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_title('Year-over-Year Growth Rate (%)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Growth Rate (%)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/04_yoy_growth.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: visualizations/04_yoy_growth.png")
    plt.close()

# ============================================================================
# 2.7 DISTRIBUTION ANALYSIS
# ============================================================================
print("\n[2.7] Distribution Analysis...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Histogram
axes[0].hist(df_monthly['Sales'], bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
axes[0].axvline(df_monthly['Sales'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
axes[0].axvline(df_monthly['Sales'].median(), color='green', linestyle='--', linewidth=2, label='Median')
axes[0].set_title('Sales Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sales ($)', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Box Plot
axes[1].boxplot(df_monthly['Sales'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_title('Sales Box Plot', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Sales ($)', fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations/05_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/05_distribution.png")
plt.close()

# ============================================================================
# 2.8 MONTHLY PATTERN ANALYSIS
# ============================================================================
print("\n[2.8] Monthly Pattern Analysis...")

# Extract month from index
df_monthly['Month'] = df_monthly.index.month

# Average sales by month
monthly_avg = df_monthly.groupby('Month')['Sales'].mean()

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(1, 13), monthly_avg.values, color='#2E86AB', alpha=0.7, edgecolor='black')
ax.set_title('Average Sales by Month', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Month', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Sales ($)', fontsize=12, fontweight='bold')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Highlight highest and lowest months
max_month = monthly_avg.idxmax()
min_month = monthly_avg.idxmin()
bars[max_month-1].set_color('green')
bars[min_month-1].set_color('red')

plt.tight_layout()
plt.savefig('visualizations/06_monthly_pattern.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/06_monthly_pattern.png")
plt.close()

print(f"\n  Peak Month: {month_names[max_month-1]} (${monthly_avg[max_month]:,.2f})")
print(f"  Lowest Month: {month_names[min_month-1]} (${monthly_avg[min_month]:,.2f})")

# ============================================================================
# 2.9 YEARLY COMPARISON
# ============================================================================
print("\n[2.9] Yearly Comparison...")

df_monthly['Year'] = df_monthly.index.year
yearly_sales = df_monthly.groupby('Year')['Sales'].sum()

if len(yearly_sales) > 1:
    fig, ax = plt.subplots(figsize=(12, 6))
    years = yearly_sales.index
    ax.bar(years, yearly_sales.values, color='#2E86AB', alpha=0.7, edgecolor='black', width=0.6)
    ax.set_title('Total Sales by Year', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Sales ($)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add value labels on bars
    for i, v in enumerate(yearly_sales.values):
        ax.text(years[i], v, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/07_yearly_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: visualizations/07_yearly_comparison.png")
    plt.close()
    
    print("\n  Yearly Sales:")
    for year, sales in yearly_sales.items():
        print(f"    {year}: ${sales:,.2f}")

# ============================================================================
# 2.10 CORRELATION ANALYSIS
# ============================================================================
print("\n[2.10] Correlation Analysis...")

# Create lagged features
df_monthly['Sales_Lag1'] = df_monthly['Sales'].shift(1)
df_monthly['Sales_Lag3'] = df_monthly['Sales'].shift(3)
df_monthly['Sales_Lag12'] = df_monthly['Sales'].shift(12)

# Calculate correlations
correlations = df_monthly[['Sales', 'Sales_Lag1', 'Sales_Lag3', 'Sales_Lag12']].corr()

print("\n  Autocorrelation Results:")
print(f"    Lag 1 (Previous Month): {correlations.loc['Sales', 'Sales_Lag1']:.3f}")
print(f"    Lag 3 (3 Months Ago): {correlations.loc['Sales', 'Sales_Lag3']:.3f}")
if not pd.isna(correlations.loc['Sales', 'Sales_Lag12']):
    print(f"    Lag 12 (Same Month Last Year): {correlations.loc['Sales', 'Sales_Lag12']:.3f}")

# ============================================================================
# 2.11 SAVE ENHANCED DATA
# ============================================================================
print("\n[2.11] Saving Enhanced Data...")

df_monthly.to_csv('data_with_features.csv')
print("  ✓ Saved: data_with_features.csv")

# ============================================================================
# 2.12 SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS COMPLETE!")
print("="*80)

print(f"\nKey Insights:")
print(f"  Average Monthly Sales: ${df_monthly['Sales'].mean():,.2f}")
print(f"  Sales Volatility (Std Dev): ${df_monthly['Sales'].std():,.2f}")
print(f"  Average MoM Growth: {df_monthly['MoM_Growth'].mean():.2f}%")
print(f"  Peak Sales Month: {month_names[max_month-1]}")
print(f"  Lowest Sales Month: {month_names[min_month-1]}")

print(f"\nVisualizations Created:")
print(f"  1. Sales Trend")
print(f"  2. Moving Averages")
print(f"  3. Month-over-Month Growth")
print(f"  4. Year-over-Year Growth")
print(f"  5. Distribution Analysis")
print(f"  6. Monthly Pattern")
print(f"  7. Yearly Comparison")

print(f"\nReady for next step: Time Series Decomposition")
print("="*80)
