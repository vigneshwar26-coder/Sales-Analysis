"""
STEP 10: FINAL REPORT GENERATION
=================================
Project: Monthly Sales Forecast and Trend Analysis
Dataset: Superstore Sales
File: 10_final_report.py
=================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 10: FINAL REPORT GENERATION")
print("="*80)

# ============================================================================
# 10.1 COLLECT ALL RESULTS
# ============================================================================
print("\n[10.1] Collecting All Analysis Results...")

results = {
    'generated_files': [],
    'visualizations': [],
    'errors': []
}

# Check for data files
data_files = [
    'data_processed_daily.csv',
    'data_processed_monthly.csv',
    'data_with_features.csv',
    'data_decomposed.csv',
    'data_train.csv',
    'data_test.csv'
]

for file in data_files:
    if os.path.exists(file):
        results['generated_files'].append(file)

# Check for model files
model_files = [
    'arima_predictions.csv',
    'arima_performance.csv',
    'holtwinters_predictions.csv',
    'holtwinters_performance.csv',
    'prophet_predictions.csv',
    'prophet_performance.csv',
    'model_comparison_performance.csv',
    'best_model_selection.csv',
    'future_forecast_6months.csv',
    'future_forecast_detailed.csv'
]

for file in model_files:
    if os.path.exists(file):
        results['generated_files'].append(file)

# Check for visualizations
if os.path.exists('visualizations'):
    viz_files = [f for f in os.listdir('visualizations') if f.endswith('.png')]
    results['visualizations'] = sorted(viz_files)

print(f"  ✓ Found {len(results['generated_files'])} data files")
print(f"  ✓ Found {len(results['visualizations'])} visualizations")

# ============================================================================
# 10.2 LOAD KEY METRICS
# ============================================================================
print("\n[10.2] Loading Key Metrics...")

# Load historical data summary
try:
    df_monthly = pd.read_csv('data_with_features.csv', index_col=0, parse_dates=True)
    total_months = len(df_monthly)
    date_start = df_monthly.index.min().strftime('%Y-%m')
    date_end = df_monthly.index.max().strftime('%Y-%m')
    total_sales = df_monthly['Sales'].sum()
    avg_sales = df_monthly['Sales'].mean()
    print(f"  ✓ Historical data: {total_months} months ({date_start} to {date_end})")
except:
    total_months = date_start = date_end = total_sales = avg_sales = "N/A"
    results['errors'].append("Could not load historical data")

# Load best model information
try:
    best_model_df = pd.read_csv('best_model_selection.csv')
    best_model = best_model_df['Best_Model'].iloc[0]
    best_mape = best_model_df['Best_MAPE'].iloc[0]
    best_mae = best_model_df['Best_MAE'].iloc[0]
    best_r2 = best_model_df['Best_R2'].iloc[0]
    print(f"  ✓ Best model: {best_model} (MAPE: {best_mape:.2f}%)")
except:
    best_model = best_mape = best_mae = best_r2 = "N/A"
    results['errors'].append("Could not load best model information")

# Load future forecast
try:
    future_forecast_df = pd.read_csv('future_forecast_6months.csv')
    forecast_total = future_forecast_df['Predicted_Sales'].sum()
    forecast_avg = future_forecast_df['Predicted_Sales'].mean()
    print(f"  ✓ Future forecast: ${forecast_total:,.2f} (next 6 months)")
except:
    forecast_total = forecast_avg = "N/A"
    results['errors'].append("Could not load future forecast")

# ============================================================================
# 10.3 CREATE COMPREHENSIVE EXCEL REPORT
# ============================================================================
print("\n[10.3] Creating Comprehensive Excel Report...")

try:
    with pd.ExcelWriter('FINAL_SALES_FORECAST_REPORT.xlsx', engine='openpyxl') as writer:
        
        # Sheet 1: Executive Summary
        exec_summary = pd.DataFrame({
            'Metric': [
                'Project Name',
                'Analysis Date',
                'Historical Data Period',
                'Total Months Analyzed',
                'Total Historical Sales',
                'Average Monthly Sales',
                '',
                'Best Forecasting Model',
                'Model MAPE',
                'Model MAE',
                'Model R²',
                '',
                'Future Forecast (6 months)',
                'Avg Forecast per Month',
                '',
                'Total Visualizations Created',
                'Total Data Files Generated'
            ],
            'Value': [
                'Monthly Sales Forecast and Trend Analysis',
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                f'{date_start} to {date_end}' if date_start != "N/A" else "N/A",
                total_months,
                f'${total_sales:,.2f}' if total_sales != "N/A" else "N/A",
                f'${avg_sales:,.2f}' if avg_sales != "N/A" else "N/A",
                '',
                best_model,
                f'{best_mape:.2f}%' if best_mape != "N/A" else "N/A",
                f'${best_mae:,.2f}' if best_mae != "N/A" else "N/A",
                f'{best_r2:.4f}' if best_r2 != "N/A" else "N/A",
                '',
                f'${forecast_total:,.2f}' if forecast_total != "N/A" else "N/A",
                f'${forecast_avg:,.2f}' if forecast_avg != "N/A" else "N/A",
                '',
                len(results['visualizations']),
                len(results['generated_files'])
            ]
        })
        exec_summary.to_excel(writer, sheet_name='Executive Summary', index=False)
        
        # Sheet 2: Historical Data
        if os.path.exists('data_with_features.csv'):
            historical = pd.read_csv('data_with_features.csv')
            historical.to_excel(writer, sheet_name='Historical Data', index=False)
        
        # Sheet 3: Model Comparison
        if os.path.exists('model_comparison_performance.csv'):
            model_comp = pd.read_csv('model_comparison_performance.csv')
            model_comp.to_excel(writer, sheet_name='Model Comparison', index=False)
        
        # Sheet 4: ARIMA Results
        if os.path.exists('arima_predictions.csv'):
            arima_results = pd.read_csv('arima_predictions.csv')
            arima_results.to_excel(writer, sheet_name='ARIMA Results', index=False)
        
        # Sheet 5: Holt-Winters Results
        if os.path.exists('holtwinters_predictions.csv'):
            hw_results = pd.read_csv('holtwinters_predictions.csv')
            hw_results.to_excel(writer, sheet_name='Holt-Winters Results', index=False)
        
        # Sheet 6: Prophet Results
        if os.path.exists('prophet_predictions.csv'):
            prophet_results = pd.read_csv('prophet_predictions.csv')
            prophet_results.to_excel(writer, sheet_name='Prophet Results', index=False)
        
        # Sheet 7: Future Forecast
        if os.path.exists('future_forecast_detailed.csv'):
            future = pd.read_csv('future_forecast_detailed.csv')
            future.to_excel(writer, sheet_name='Future Forecast', index=False)
        
        # Sheet 8: File Inventory
        file_inventory = pd.DataFrame({
            'Category': ['Data Files'] * len(results['generated_files']) + \
                       ['Visualizations'] * len(results['visualizations']),
            'File Name': results['generated_files'] + results['visualizations']
        })
        file_inventory.to_excel(writer, sheet_name='File Inventory', index=False)
    
    print("  ✓ Created: FINAL_SALES_FORECAST_REPORT.xlsx")
    
except Exception as e:
    print(f"  ✗ Error creating Excel report: {e}")
    results['errors'].append(f"Excel report error: {e}")

# ============================================================================
# 10.4 CREATE TEXT SUMMARY REPORT
# ============================================================================
print("\n[10.4] Creating Text Summary Report...")

try:
    with open('ANALYSIS_SUMMARY_REPORT.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("MONTHLY SALES FORECAST AND TREND ANALYSIS\n")
        f.write("Final Summary Report\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: Superstore Sales\n\n")
        
        f.write("="*80 + "\n")
        f.write("1. DATA OVERVIEW\n")
        f.write("="*80 + "\n")
        f.write(f"Historical Period: {date_start} to {date_end}\n")
        f.write(f"Total Months: {total_months}\n")
        if total_sales != "N/A":
            f.write(f"Total Historical Sales: ${total_sales:,.2f}\n")
            f.write(f"Average Monthly Sales: ${avg_sales:,.2f}\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("2. MODEL PERFORMANCE\n")
        f.write("="*80 + "\n")
        
        if os.path.exists('model_comparison_performance.csv'):
            perf = pd.read_csv('model_comparison_performance.csv')
            for idx, row in perf.iterrows():
                f.write(f"\n{row['Model']}:\n")
                f.write(f"  MAE:  ${row['MAE']:,.2f}\n")
                f.write(f"  RMSE: ${row['RMSE']:,.2f}\n")
                f.write(f"  MAPE: {row['MAPE']:.2f}%\n")
                f.write(f"  R²:   {row['R2']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("3. BEST MODEL\n")
        f.write("="*80 + "\n")
        f.write(f"Selected Model: {best_model}\n")
        if best_mape != "N/A":
            f.write(f"MAPE: {best_mape:.2f}%\n")
            f.write(f"MAE: ${best_mae:,.2f}\n")
            f.write(f"R²: {best_r2:.4f}\n")
            
            if best_mape < 10:
                f.write("\nAccuracy Rating: ⭐⭐⭐⭐⭐ EXCELLENT\n")
            elif best_mape < 20:
                f.write("\nAccuracy Rating: ⭐⭐⭐⭐ GOOD\n")
            elif best_mape < 30:
                f.write("\nAccuracy Rating: ⭐⭐⭐ ACCEPTABLE\n")
            else:
                f.write("\nAccuracy Rating: ⭐⭐ NEEDS IMPROVEMENT\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("4. FUTURE FORECAST (NEXT 6 MONTHS)\n")
        f.write("="*80 + "\n")
        
        if os.path.exists('future_forecast_6months.csv'):
            future = pd.read_csv('future_forecast_6months.csv')
            f.write(f"\nTotal Predicted Sales: ${future['Predicted_Sales'].sum():,.2f}\n")
            f.write(f"Average Monthly Sales: ${future['Predicted_Sales'].mean():,.2f}\n\n")
            f.write("Month-by-Month Forecast:\n")
            for idx, row in future.iterrows():
                f.write(f"  {row['Month']}: ${row['Predicted_Sales']:,.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("5. FILES GENERATED\n")
        f.write("="*80 + "\n")
        f.write(f"\nData Files ({len(results['generated_files'])}):\n")
        for file in results['generated_files']:
            f.write(f"  • {file}\n")
        
        f.write(f"\nVisualizations ({len(results['visualizations'])}):\n")
        for viz in results['visualizations']:
            f.write(f"  • {viz}\n")
        
        if results['errors']:
            f.write("\n" + "="*80 + "\n")
            f.write("6. WARNINGS/ERRORS\n")
            f.write("="*80 + "\n")
            for error in results['errors']:
                f.write(f"  ⚠ {error}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("7. RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        f.write("\n1. Use the best model for future business planning\n")
        f.write("2. Monitor actual vs. predicted values monthly\n")
        f.write("3. Retrain models quarterly with new data\n")
        f.write("4. Review forecasts before major business decisions\n")
        f.write("5. Consider external factors (promotions, holidays) not in model\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print("  ✓ Created: ANALYSIS_SUMMARY_REPORT.txt")
    
except Exception as e:
    print(f"  ✗ Error creating text report: {e}")
    results['errors'].append(f"Text report error: {e}")

# ============================================================================
# 10.5 CREATE PROJECT COMPLETION CHECKLIST
# ============================================================================
print("\n[10.5] Creating Project Completion Checklist...")

checklist = {
    'Step': [
        '1. Data Loading & Preparation',
        '2. Exploratory Data Analysis',
        '3. Time Series Decomposition',
        '4. Train-Test Split',
        '5. ARIMA Model',
        '6. Holt-Winters Model',
        '7. Prophet Model',
        '8. Model Comparison',
        '9. Future Forecasting',
        '10. Final Report'
    ],
    'Status': [
        '✓' if os.path.exists('data_processed_monthly.csv') else '✗',
        '✓' if os.path.exists('data_with_features.csv') else '✗',
        '✓' if os.path.exists('data_decomposed.csv') else '✗',
        '✓' if os.path.exists('data_train.csv') else '✗',
        '✓' if os.path.exists('arima_predictions.csv') else '✗',
        '✓' if os.path.exists('holtwinters_predictions.csv') else '✗',
        '✓' if os.path.exists('prophet_predictions.csv') else '✗',
        '✓' if os.path.exists('model_comparison_performance.csv') else '✗',
        '✓' if os.path.exists('future_forecast_6months.csv') else '✗',
        '✓'
    ],
    'Key_Output': [
        'data_processed_monthly.csv',
        'data_with_features.csv + 7 visualizations',
        'data_decomposed.csv + 4 visualizations',
        'data_train.csv, data_test.csv',
        'arima_predictions.csv + 4 visualizations',
        'holtwinters_predictions.csv + 3 visualizations',
        'prophet_predictions.csv + 3 visualizations',
        'model_comparison_performance.csv + 4 visualizations',
        'future_forecast_6months.csv + 2 visualizations',
        'FINAL_SALES_FORECAST_REPORT.xlsx'
    ]
}

checklist_df = pd.DataFrame(checklist)
checklist_df.to_csv('PROJECT_COMPLETION_CHECKLIST.csv', index=False)
print("  ✓ Created: PROJECT_COMPLETION_CHECKLIST.csv")

print("\n" + "="*80)
print("PROJECT COMPLETION CHECKLIST")
print("="*80)
print(checklist_df.to_string(index=False))

# ============================================================================
# 10.6 FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL REPORT GENERATION COMPLETE!")
print("="*80)

print(f"\n📊 Project Statistics:")
print(f"  Total Steps Completed: 10/10")
print(f"  Data Files Generated: {len(results['generated_files'])}")
print(f"  Visualizations Created: {len(results['visualizations'])}")
print(f"  Analysis Period: {total_months} months" if total_months != "N/A" else "  Analysis Period: N/A")

if best_model != "N/A":
    print(f"\n🏆 Best Model: {best_model}")
    print(f"  Accuracy (MAPE): {best_mape:.2f}%")

if forecast_total != "N/A":
    print(f"\n📈 Future Outlook (Next 6 Months):")
    print(f"  Total Predicted Sales: ${forecast_total:,.2f}")
    print(f"  Average Monthly Sales: ${forecast_avg:,.2f}")

print(f"\n📄 Final Reports Created:")
print(f"  • FINAL_SALES_FORECAST_REPORT.xlsx (Comprehensive Excel report)")
print(f"  • ANALYSIS_SUMMARY_REPORT.txt (Text summary)")
print(f"  • PROJECT_COMPLETION_CHECKLIST.csv (Project status)")

if results['errors']:
    print(f"\n⚠ Warnings ({len(results['errors'])}):")
    for error in results['errors']:
        print(f"  • {error}")

print("\n" + "="*80)
print("🎉 ANALYSIS COMPLETE! 🎉")
print("="*80)
print("\nAll analysis steps have been completed successfully.")
print("Review the generated reports and visualizations for insights.")
print("Use the best model for future sales forecasting and planning.")
print("\nThank you for using the Sales Forecast Analysis System!")
print("="*80)
