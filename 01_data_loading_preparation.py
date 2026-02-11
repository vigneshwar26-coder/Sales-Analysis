"""
STEP 1: DATA LOADING AND PREPARATION
=====================================
Project: Monthly Sales Forecast and Trend Analysis
Dataset: Superstore Sales
File: 01_data_loading_preparation.py
=====================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 1: DATA LOADING AND PREPARATION")
print("="*80)

# ============================================================================
# 1.1 LOAD THE DATASET
# ============================================================================
print("\n[1.1] Loading Superstore Sales Dataset...")

try:
    # Load the dataset
    df = pd.read_csv('"C:/Users/vigneshwar/OneDrive/Desktop/Data"')
    print(f"✓ Data loaded successfully!")
    print(f"  Total records: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
except FileNotFoundError:
    print("✗ Error: 'superstore_sales.csv' not found!")
    print("  Please ensure the file is in the same directory.")
    exit()

# ============================================================================
# 1.2 DISPLAY BASIC INFORMATION
# ============================================================================
print("\n[1.2] Dataset Overview...")

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset Info:")
print(df.info())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# ============================================================================
# 1.3 IDENTIFY KEY COLUMNS
# ============================================================================
print("\n[1.3] Identifying Key Columns...")

# Find date column
date_columns = ['Order Date', 'order_date', 'Date', 'date', 'OrderDate']
date_col = None
for col in date_columns:
    if col in df.columns:
        date_col = col
        print(f"  ✓ Date column found: '{date_col}'")
        break

if date_col is None:
    print(f"  ✗ Error: Could not find date column!")
    print(f"  Available columns: {df.columns.tolist()}")
    exit()

# Find sales column
sales_columns = ['Sales', 'sales', 'Amount', 'Revenue', 'revenue']
sales_col = None
for col in sales_columns:
    if col in df.columns:
        sales_col = col
        print(f"  ✓ Sales column found: '{sales_col}'")
        break

if sales_col is None:
    print(f"  ✗ Error: Could not find sales column!")
    print(f"  Available columns: {df.columns.tolist()}")
    exit()

# ============================================================================
# 1.4 DATA CLEANING
# ============================================================================
print("\n[1.4] Data Cleaning...")

# Check for missing values
print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("  ✓ No missing values found!")
else:
    print(f"  Total missing values: {missing_values.sum()}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
if duplicates > 0:
    print(f"  Removing {duplicates} duplicate rows...")
    df = df.drop_duplicates()
    print(f"  ✓ Duplicates removed. Remaining records: {len(df):,}")

# ============================================================================
# 1.5 DATE CONVERSION AND VALIDATION
# ============================================================================
print("\n[1.5] Date Conversion and Validation...")

# Convert to datetime
print(f"  Converting '{date_col}' to datetime format...")
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

# Check for invalid dates
invalid_dates = df[date_col].isnull().sum()
if invalid_dates > 0:
    print(f"  ⚠ Warning: {invalid_dates} invalid dates found and removed")
    df = df.dropna(subset=[date_col])

print(f"  ✓ Date range: {df[date_col].min().strftime('%Y-%m-%d')} to {df[date_col].max().strftime('%Y-%m-%d')}")

# ============================================================================
# 1.6 SALES DATA VALIDATION
# ============================================================================
print("\n[1.6] Sales Data Validation...")

# Check for missing sales values
missing_sales = df[sales_col].isnull().sum()
if missing_sales > 0:
    print(f"  ⚠ Warning: {missing_sales} missing sales values found and removed")
    df = df.dropna(subset=[sales_col])

# Check for negative sales
negative_sales = (df[sales_col] < 0).sum()
if negative_sales > 0:
    print(f"  ⚠ Warning: {negative_sales} negative sales values found")
    print("  You may want to investigate these records")

# Check for zero sales
zero_sales = (df[sales_col] == 0).sum()
if zero_sales > 0:
    print(f"  ℹ Info: {zero_sales} zero sales values found")

print(f"\n  Sales Statistics:")
print(f"    Minimum: ${df[sales_col].min():,.2f}")
print(f"    Maximum: ${df[sales_col].max():,.2f}")
print(f"    Mean: ${df[sales_col].mean():,.2f}")
print(f"    Median: ${df[sales_col].median():,.2f}")

# ============================================================================
# 1.7 SORT AND INDEX
# ============================================================================
print("\n[1.7] Sorting and Indexing...")

# Sort by date
df = df.sort_values(date_col)
print(f"  ✓ Data sorted by {date_col}")

# Create a working copy with date as index
df_working = df.set_index(date_col)
print(f"  ✓ Date set as index")

# ============================================================================
# 1.8 AGGREGATE TO MONTHLY LEVEL
# ============================================================================
print("\n[1.8] Aggregating to Monthly Level...")

# Resample to monthly frequency
df_monthly = df_working[sales_col].resample('M').sum()
df_monthly = pd.DataFrame(df_monthly)
df_monthly.columns = ['Sales']

print(f"  ✓ Monthly aggregation complete")
print(f"  Total months: {len(df_monthly)}")
print(f"  Date range: {df_monthly.index.min().strftime('%Y-%m')} to {df_monthly.index.max().strftime('%Y-%m')}")

# Display monthly data
print("\n  First 12 months:")
print(df_monthly.head(12))

# ============================================================================
# 1.9 SAVE PROCESSED DATA
# ============================================================================
print("\n[1.9] Saving Processed Data...")

# Save original processed data
df.to_csv('data_processed_daily.csv', index=False)
print("  ✓ Saved: data_processed_daily.csv")

# Save monthly aggregated data
df_monthly.to_csv('data_processed_monthly.csv')
print("  ✓ Saved: data_processed_monthly.csv")

# ============================================================================
# 1.10 SUMMARY STATISTICS
# ============================================================================
print("\n[1.10] Summary Statistics...")

print("\nMonthly Sales Summary:")
print(df_monthly['Sales'].describe())

print("\n" + "="*80)
print("DATA LOADING AND PREPARATION COMPLETE!")
print("="*80)
print(f"\nProcessed Records:")
print(f"  Daily transactions: {len(df):,}")
print(f"  Monthly periods: {len(df_monthly)}")
print(f"  Date range: {df_monthly.index.min().strftime('%Y-%m')} to {df_monthly.index.max().strftime('%Y-%m')}")
print(f"  Total sales: ${df_monthly['Sales'].sum():,.2f}")
print(f"\nReady for next step: Exploratory Data Analysis")
print("="*80)
