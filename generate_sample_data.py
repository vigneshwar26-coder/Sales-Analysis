"""
Generate Sample Superstore Sales Data
Use this if you don't have the actual Superstore dataset yet
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Generating sample Superstore Sales data...")

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 12, 31)
num_days = (end_date - start_date).days

# Generate dates
dates = [start_date + timedelta(days=x) for x in range(num_days)]

# Categories and regions
categories = ['Furniture', 'Office Supplies', 'Technology']
sub_categories = {
    'Furniture': ['Chairs', 'Tables', 'Bookcases', 'Furnishings'],
    'Office Supplies': ['Paper', 'Binders', 'Art', 'Appliances', 'Storage'],
    'Technology': ['Phones', 'Accessories', 'Machines', 'Copiers']
}
regions = ['East', 'West', 'Central', 'South']
segments = ['Consumer', 'Corporate', 'Home Office']

# Generate transactions
num_transactions = 10000
records = []

for i in range(num_transactions):
    date = np.random.choice(dates)
    category = np.random.choice(categories)
    sub_category = np.random.choice(sub_categories[category])
    region = np.random.choice(regions)
    segment = np.random.choice(segments)
    
    # Generate sales with seasonality
    month = date.month
    
    # Base sales amount by category
    if category == 'Technology':
        base_sales = np.random.uniform(100, 2000)
    elif category == 'Furniture':
        base_sales = np.random.uniform(50, 1500)
    else:
        base_sales = np.random.uniform(10, 500)
    
    # Add seasonality (Q4 boost)
    if month in [11, 12]:
        base_sales *= 1.4
    elif month in [6, 7]:
        base_sales *= 0.9
    
    # Add trend (growth over time)
    year_factor = 1 + (date.year - 2020) * 0.08
    sales = base_sales * year_factor
    
    # Generate quantity and discount
    quantity = np.random.randint(1, 10)
    discount = np.random.choice([0, 0, 0, 0.1, 0.15, 0.2], p=[0.5, 0.2, 0.15, 0.1, 0.03, 0.02])
    
    # Calculate profit (15-40% margin)
    profit_margin = np.random.uniform(0.15, 0.40)
    profit = sales * profit_margin * (1 - discount)
    
    records.append({
        'Order Date': date,
        'Ship Date': date + timedelta(days=np.random.randint(1, 7)),
        'Ship Mode': np.random.choice(['Standard Class', 'Second Class', 'First Class', 'Same Day']),
        'Segment': segment,
        'Category': category,
        'Sub-Category': sub_category,
        'Region': region,
        'Sales': round(sales, 2),
        'Quantity': quantity,
        'Discount': discount,
        'Profit': round(profit, 2)
    })

# Create DataFrame
df = pd.DataFrame(records)

# Sort by date
df = df.sort_values('Order Date')

# Add Order ID
df.insert(0, 'Order ID', ['ORDER-' + str(i).zfill(6) for i in range(1, len(df) + 1)])

# Save to CSV
df.to_csv('superstore_sales.csv', index=False)

print(f"\n✓ Generated {len(df):,} transactions")
print(f"  Date Range: {df['Order Date'].min().strftime('%Y-%m-%d')} to {df['Order Date'].max().strftime('%Y-%m-%d')}")
print(f"  Total Sales: ${df['Sales'].sum():,.2f}")
print(f"  Total Profit: ${df['Profit'].sum():,.2f}")
print(f"\n  File saved: superstore_sales.csv")

# Display sample
print("\nFirst 10 rows:")
print(df.head(10))

print("\nDataset Statistics:")
print(df[['Sales', 'Quantity', 'Discount', 'Profit']].describe())

print("\n" + "="*60)
print("Sample data generated successfully!")
print("You can now run: python superstore_sales_forecast.py")
print("="*60)
