"""
STEP 4: TRAIN-TEST SPLIT
=========================
Project: Monthly Sales Forecast and Trend Analysis
Dataset: Superstore Sales
File: 04_train_test_split.py
=========================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("STEP 4: TRAIN-TEST SPLIT")
print("="*80)

# ============================================================================
# 4.1 LOAD PROCESSED DATA
# ============================================================================
print("\n[4.1] Loading Processed Data...")

df_monthly = pd.read_csv('data_with_features.csv', index_col=0, parse_dates=True)
print(f"✓ Loaded {len(df_monthly)} months of data")
print(f"  Date range: {df_monthly.index.min().strftime('%Y-%m')} to {df_monthly.index.max().strftime('%Y-%m')}")

# ============================================================================
# 4.2 DETERMINE SPLIT RATIO
# ============================================================================
print("\n[4.2] Determining Split Ratio...")

total_months = len(df_monthly)

# Recommended split ratios based on data size
if total_months < 24:
    split_ratio = 0.75  # 75-25 for small datasets
elif total_months < 48:
    split_ratio = 0.80  # 80-20 for medium datasets
else:
    split_ratio = 0.85  # 85-15 for large datasets

print(f"  Total months: {total_months}")
print(f"  Recommended split ratio: {int(split_ratio*100)}-{int((1-split_ratio)*100)}")

# Allow user override (you can change this value)
CUSTOM_SPLIT_RATIO = 0.80  # Change this if needed

if CUSTOM_SPLIT_RATIO != split_ratio:
    print(f"  Using custom split ratio: {int(CUSTOM_SPLIT_RATIO*100)}-{int((1-CUSTOM_SPLIT_RATIO)*100)}")
    split_ratio = CUSTOM_SPLIT_RATIO

# ============================================================================
# 4.3 PERFORM THE SPLIT
# ============================================================================
print("\n[4.3] Performing Train-Test Split...")

train_size = int(len(df_monthly) * split_ratio)
test_size = len(df_monthly) - train_size

# Split the data
train = df_monthly['Sales'][:train_size]
test = df_monthly['Sales'][train_size:]

print(f"\n  Training Set:")
print(f"    Size: {len(train)} months")
print(f"    Period: {train.index.min().strftime('%Y-%m')} to {train.index.max().strftime('%Y-%m')}")
print(f"    Total Sales: ${train.sum():,.2f}")
print(f"    Average Monthly Sales: ${train.mean():,.2f}")

print(f"\n  Test Set:")
print(f"    Size: {len(test)} months")
print(f"    Period: {test.index.min().strftime('%Y-%m')} to {test.index.max().strftime('%Y-%m')}")
print(f"    Total Sales: ${test.sum():,.2f}")
print(f"    Average Monthly Sales: ${test.mean():,.2f}")

# ============================================================================
# 4.4 VALIDATE SPLIT
# ============================================================================
print("\n[4.4] Validating Split...")

# Check for sufficient training data
min_training_months = 12
if len(train) < min_training_months:
    print(f"  ⚠ Warning: Training set has only {len(train)} months")
    print(f"    Recommended minimum: {min_training_months} months")
else:
    print(f"  ✓ Training set size is adequate ({len(train)} months)")

# Check for sufficient test data
min_test_months = 3
if len(test) < min_test_months:
    print(f"  ⚠ Warning: Test set has only {len(test)} months")
    print(f"    Recommended minimum: {min_test_months} months")
else:
    print(f"  ✓ Test set size is adequate ({len(test)} months)")

# Check for data leakage
assert train.index.max() < test.index.min(), "Data leakage detected!"
print(f"  ✓ No data leakage detected (train ends before test begins)")

# ============================================================================
# 4.5 VISUALIZE THE SPLIT
# ============================================================================
print("\n[4.5] Visualizing Train-Test Split...")

fig, ax = plt.subplots(figsize=(15, 6))

# Plot training data
ax.plot(train.index, train.values, label='Training Set', 
        color='#2E86AB', linewidth=2, marker='o', markersize=4)

# Plot test data
ax.plot(test.index, test.values, label='Test Set', 
        color='#E71D36', linewidth=2, marker='s', markersize=5)

# Add vertical line at split point
split_date = train.index.max()
ax.axvline(x=split_date, color='black', linestyle='--', linewidth=2, 
           label=f'Split Point ({split_date.strftime("%Y-%m")})')

# Add shaded regions
ax.axvspan(train.index.min(), train.index.max(), alpha=0.1, color='blue', label='Training Period')
ax.axvspan(test.index.min(), test.index.max(), alpha=0.1, color='red', label='Test Period')

ax.set_title('Train-Test Split Visualization', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/12_train_test_split.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/12_train_test_split.png")
plt.close()

# ============================================================================
# 4.6 STATISTICAL COMPARISON
# ============================================================================
print("\n[4.6] Statistical Comparison of Train and Test Sets...")

from scipy import stats

# Descriptive statistics
train_stats = train.describe()
test_stats = test.describe()

print("\n  Descriptive Statistics:")
print("\n  Training Set:")
print(f"    Mean: ${train_stats['mean']:,.2f}")
print(f"    Std Dev: ${train_stats['std']:,.2f}")
print(f"    Min: ${train_stats['min']:,.2f}")
print(f"    Max: ${train_stats['max']:,.2f}")

print("\n  Test Set:")
print(f"    Mean: ${test_stats['mean']:,.2f}")
print(f"    Std Dev: ${test_stats['std']:,.2f}")
print(f"    Min: ${test_stats['min']:,.2f}")
print(f"    Max: ${test_stats['max']:,.2f}")

# T-test for mean difference
t_stat, p_value = stats.ttest_ind(train, test)
print(f"\n  T-Test for Mean Difference:")
print(f"    T-statistic: {t_stat:.4f}")
print(f"    P-value: {p_value:.4f}")

if p_value > 0.05:
    print(f"    Result: ✓ No significant difference in means (p > 0.05)")
else:
    print(f"    Result: ⚠ Significant difference in means (p ≤ 0.05)")
    print(f"    This may indicate trend or structural changes in the data")

# ============================================================================
# 4.7 DISTRIBUTION COMPARISON
# ============================================================================
print("\n[4.7] Comparing Distributions...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Box plots
box_data = [train.values, test.values]
box_labels = ['Training Set', 'Test Set']
bp = axes[0].boxplot(box_data, labels=box_labels, patch_artist=True,
                     boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
axes[0].set_title('Distribution Comparison (Box Plot)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Sales ($)', fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Histograms
axes[1].hist(train.values, bins=15, alpha=0.6, label='Training Set', color='#2E86AB', edgecolor='black')
axes[1].hist(test.values, bins=15, alpha=0.6, label='Test Set', color='#E71D36', edgecolor='black')
axes[1].set_title('Distribution Comparison (Histogram)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Sales ($)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('visualizations/13_distribution_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/13_distribution_comparison.png")
plt.close()

# ============================================================================
# 4.8 SEASONALITY CHECK
# ============================================================================
print("\n[4.8] Checking Seasonality Coverage...")

train_months = train.index.month.unique()
test_months = test.index.month.unique()

print(f"\n  Training Set Months Coverage:")
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
train_month_names = [month_names[m-1] for m in sorted(train_months)]
print(f"    Months present: {', '.join(train_month_names)}")
print(f"    Coverage: {len(train_months)}/12 months")

print(f"\n  Test Set Months Coverage:")
test_month_names = [month_names[m-1] for m in sorted(test_months)]
print(f"    Months present: {', '.join(test_month_names)}")
print(f"    Coverage: {len(test_months)}/12 months")

if len(train_months) == 12:
    print(f"  ✓ Training set covers all 12 months (good for seasonality learning)")
else:
    print(f"  ⚠ Training set missing some months (may affect seasonal modeling)")

# ============================================================================
# 4.9 SAVE SPLIT DATA
# ============================================================================
print("\n[4.9] Saving Split Data...")

# Save as separate CSV files
train_df = pd.DataFrame({'Sales': train})
test_df = pd.DataFrame({'Sales': test})

train_df.to_csv('data_train.csv')
test_df.to_csv('data_test.csv')

print("  ✓ Saved: data_train.csv")
print("  ✓ Saved: data_test.csv")

# Also save split information
split_info = {
    'Total_Months': total_months,
    'Train_Months': len(train),
    'Test_Months': len(test),
    'Split_Ratio': split_ratio,
    'Train_Start': train.index.min().strftime('%Y-%m'),
    'Train_End': train.index.max().strftime('%Y-%m'),
    'Test_Start': test.index.min().strftime('%Y-%m'),
    'Test_End': test.index.max().strftime('%Y-%m'),
    'Train_Total_Sales': train.sum(),
    'Test_Total_Sales': test.sum(),
    'Train_Avg_Sales': train.mean(),
    'Test_Avg_Sales': test.mean()
}

split_info_df = pd.DataFrame([split_info])
split_info_df.to_csv('split_information.csv', index=False)
print("  ✓ Saved: split_information.csv")

# ============================================================================
# 4.10 SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TRAIN-TEST SPLIT COMPLETE!")
print("="*80)

print(f"\nSplit Summary:")
print(f"  Split Ratio: {int(split_ratio*100)}-{int((1-split_ratio)*100)}")
print(f"  Training: {len(train)} months ({train.index.min().strftime('%Y-%m')} to {train.index.max().strftime('%Y-%m')})")
print(f"  Test: {len(test)} months ({test.index.min().strftime('%Y-%m')} to {test.index.max().strftime('%Y-%m')})")
print(f"  Data Quality: {'✓ Good' if len(train) >= min_training_months and len(test) >= min_test_months else '⚠ Review recommended'}")

print(f"\nFiles Created:")
print(f"  • data_train.csv")
print(f"  • data_test.csv")
print(f"  • split_information.csv")
print(f"  • visualizations/12_train_test_split.png")
print(f"  • visualizations/13_distribution_comparison.png")

print(f"\nReady for next step: Model Building and Training")
print("="*80)
