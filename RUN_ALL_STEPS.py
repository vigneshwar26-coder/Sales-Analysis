"""
MASTER EXECUTION SCRIPT
=======================
Run All Sales Forecasting Steps Sequentially
=============================================
"""

import os
import sys
import subprocess
from datetime import datetime

print("="*80)
print("SUPERSTORE SALES FORECASTING - MASTER EXECUTION")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# List of all scripts in order
scripts = [
    ("01_data_loading_preparation.py", "Data Loading & Preparation"),
    ("02_exploratory_data_analysis.py", "Exploratory Data Analysis"),
    ("03_time_series_decomposition.py", "Time Series Decomposition"),
    ("04_train_test_split.py", "Train-Test Split"),
    ("05_arima_model.py", "ARIMA Model Building"),
    ("06_holtwinters_model.py", "Holt-Winters Model Building"),
    ("07_prophet_model.py", "Prophet Model Building"),
    ("08_model_comparison.py", "Model Comparison & Selection"),
    ("09_future_forecast.py", "Future Forecasting"),
    ("10_final_report.py", "Final Report Generation")
]

# Track execution
results = []
start_time = datetime.now()

for idx, (script, description) in enumerate(scripts, 1):
    print(f"\n{'='*80}")
    print(f"STEP {idx}/10: {description}")
    print(f"Script: {script}")
    print(f"{'='*80}")
    
    if not os.path.exists(script):
        print(f"  ✗ ERROR: Script '{script}' not found!")
        results.append({
            'step': idx,
            'script': script,
            'description': description,
            'status': 'FAILED',
            'error': 'Script not found'
        })
        continue
    
    try:
        # Execute the script
        result = subprocess.run([sys.executable, script], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout per script
        
        if result.returncode == 0:
            print(f"\n  ✓ {description} completed successfully!")
            results.append({
                'step': idx,
                'script': script,
                'description': description,
                'status': 'SUCCESS',
                'error': None
            })
        else:
            print(f"\n  ✗ {description} failed!")
            print(f"  Error: {result.stderr[:200]}")
            results.append({
                'step': idx,
                'script': script,
                'description': description,
                'status': 'FAILED',
                'error': result.stderr[:200]
            })
    
    except subprocess.TimeoutExpired:
        print(f"\n  ✗ {description} timed out (exceeded 5 minutes)!")
        results.append({
            'step': idx,
            'script': script,
            'description': description,
            'status': 'TIMEOUT',
            'error': 'Execution exceeded 5 minutes'
        })
    
    except Exception as e:
        print(f"\n  ✗ {description} encountered an error!")
        print(f"  Error: {str(e)}")
        results.append({
            'step': idx,
            'script': script,
            'description': description,
            'status': 'ERROR',
            'error': str(e)
        })

# Summary
end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print("\n" + "="*80)
print("EXECUTION SUMMARY")
print("="*80)

successful = sum(1 for r in results if r['status'] == 'SUCCESS')
failed = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR', 'TIMEOUT'])

print(f"\nTotal Steps: {len(scripts)}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Success Rate: {(successful/len(scripts)*100):.1f}%")
print(f"Total Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")

print("\n" + "="*80)
print("STEP-BY-STEP RESULTS")
print("="*80)

for result in results:
    status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗"
    print(f"\n{status_symbol} Step {result['step']}: {result['description']}")
    print(f"  Status: {result['status']}")
    if result['error']:
        print(f"  Error: {result['error']}")

print("\n" + "="*80)

if failed == 0:
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)
    print("\nYour analysis is complete. Check the following:")
    print("  1. FINAL_SALES_FORECAST_REPORT.xlsx - Complete Excel report")
    print("  2. ANALYSIS_SUMMARY_REPORT.txt - Text summary")
    print("  3. visualizations/ folder - All charts and graphs")
    print("  4. Individual CSV files for detailed data")
else:
    print("⚠ SOME STEPS FAILED")
    print("="*80)
    print("\nPlease check the errors above and:")
    print("  1. Ensure all dependencies are installed")
    print("  2. Verify data file exists (superstore_sales.csv)")
    print("  3. Check individual script outputs for details")
    print("  4. You can run failed scripts individually")

print("\n" + "="*80)
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
