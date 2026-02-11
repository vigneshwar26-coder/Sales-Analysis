"""
STEP 8: MODEL COMPARISON AND SELECTION
=======================================
Project: Monthly Sales Forecast and Trend Analysis
Dataset: Superstore Sales
File: 08_model_comparison.py
=======================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("="*80)
print("STEP 8: MODEL COMPARISON AND SELECTION")
print("="*80)

# ============================================================================
# 8.1 LOAD ALL MODEL RESULTS
# ============================================================================
print("\n[8.1] Loading All Model Results...")

# Load test data
test = pd.read_csv('data_test.csv', index_col=0, parse_dates=True)['Sales']

# Load predictions from all models
models_data = {}

# ARIMA
try:
    arima_pred = pd.read_csv('arima_predictions.csv', index_col=0, parse_dates=True)
    arima_perf = pd.read_csv('arima_performance.csv')
    models_data['ARIMA'] = {
        'predictions': arima_pred['Predicted'],
        'performance': arima_perf.iloc[0]
    }
    print("  ✓ ARIMA results loaded")
except FileNotFoundError:
    print("  ⚠ ARIMA results not found")

# Holt-Winters
try:
    hw_pred = pd.read_csv('holtwinters_predictions.csv', index_col=0, parse_dates=True)
    hw_perf = pd.read_csv('holtwinters_performance.csv')
    # Get the best model (lowest MAPE)
    best_hw = hw_perf.loc[hw_perf['MAPE'].idxmin()]
    models_data['Holt-Winters'] = {
        'predictions': hw_pred['Predicted'],
        'performance': best_hw
    }
    print("  ✓ Holt-Winters results loaded")
except FileNotFoundError:
    print("  ⚠ Holt-Winters results not found")

# Prophet
try:
    prophet_pred = pd.read_csv('prophet_predictions.csv', index_col=0, parse_dates=True)
    prophet_perf = pd.read_csv('prophet_performance.csv')
    models_data['Prophet'] = {
        'predictions': prophet_pred['Predicted'],
        'performance': prophet_perf.iloc[0]
    }
    print("  ✓ Prophet results loaded")
except FileNotFoundError:
    print("  ⚠ Prophet results not found")

print(f"\n  Total models loaded: {len(models_data)}")

# ============================================================================
# 8.2 CREATE PERFORMANCE COMPARISON TABLE
# ============================================================================
print("\n[8.2] Creating Performance Comparison Table...")

performance_list = []
for model_name, data in models_data.items():
    perf = data['performance']
    performance_list.append({
        'Model': model_name,
        'MAE': perf['MAE'],
        'RMSE': perf['RMSE'],
        'MAPE': perf['MAPE'],
        'R2': perf['R2']
    })

performance_df = pd.DataFrame(performance_list)
performance_df = performance_df.sort_values('MAPE')

print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)
print(performance_df.to_string(index=False))
print("="*80)

# Identify best model
best_model = performance_df.iloc[0]['Model']
print(f"\n🏆 BEST MODEL: {best_model} (Lowest MAPE: {performance_df.iloc[0]['MAPE']:.2f}%)")

# ============================================================================
# 8.3 VISUALIZE PERFORMANCE METRICS
# ============================================================================
print("\n[8.3] Visualizing Performance Metrics...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
titles = ['Mean Absolute Error ($)', 'Root Mean Squared Error ($)', 
          'Mean Absolute Percentage Error (%)', 'R-Squared Score']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#9B59B6']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    
    values = performance_df[metric].values
    models = performance_df['Model'].values
    
    bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Highlight best model (lowest for MAE, RMSE, MAPE; highest for R2)
    if metric == 'R2':
        best_idx = values.argmax()
    else:
        best_idx = values.argmin()
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('darkgreen')
    bars[best_idx].set_linewidth(3)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (model, value) in enumerate(zip(models, values)):
        if metric in ['MAE', 'RMSE']:
            label = f'${value:,.0f}'
        elif metric == 'MAPE':
            label = f'{value:.1f}%'
        else:
            label = f'{value:.3f}'
        ax.text(i, value, label, ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('visualizations/24_performance_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/24_performance_comparison.png")
plt.close()

# ============================================================================
# 8.4 COMPARE PREDICTIONS VISUALLY
# ============================================================================
print("\n[8.4] Comparing Predictions Visually...")

train = pd.read_csv('data_train.csv', index_col=0, parse_dates=True)['Sales']

fig, ax = plt.subplots(figsize=(16, 8))

# Plot training data
ax.plot(train.index, train.values, label='Training Data', 
        color='#2E86AB', linewidth=2, marker='o', markersize=4, alpha=0.6)

# Plot actual test data
ax.plot(test.index, test.values, label='Actual Test Data', 
        color='#0F4C5C', linewidth=3, marker='o', markersize=7, zorder=5)

# Plot predictions from all models
colors_pred = ['#E71D36', '#F18F01', '#9B59B6']
markers = ['s', '^', 'D']

for idx, (model_name, data) in enumerate(models_data.items()):
    predictions = data['predictions']
    ax.plot(test.index, predictions.values, 
            label=f'{model_name} Forecast', 
            color=colors_pred[idx % len(colors_pred)],
            linestyle='--', linewidth=2, 
            marker=markers[idx % len(markers)], 
            markersize=5)

ax.set_title('All Models - Forecast Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Sales ($)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/25_all_models_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/25_all_models_comparison.png")
plt.close()

# ============================================================================
# 8.5 ERROR ANALYSIS COMPARISON
# ============================================================================
print("\n[8.5] Error Analysis Comparison...")

fig, axes = plt.subplots(1, len(models_data), figsize=(5*len(models_data), 5))

if len(models_data) == 1:
    axes = [axes]

for idx, (model_name, data) in enumerate(models_data.items()):
    errors = test.values - data['predictions'].values
    
    axes[idx].hist(errors, bins=15, alpha=0.7, edgecolor='black', 
                   color=colors_pred[idx % len(colors_pred)])
    axes[idx].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[idx].set_title(f'{model_name}\nError Distribution', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Prediction Error ($)', fontsize=11)
    axes[idx].set_ylabel('Frequency', fontsize=11)
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_error = errors.mean()
    std_error = errors.std()
    axes[idx].text(0.05, 0.95, f'Mean: ${mean_error:,.0f}\nStd: ${std_error:,.0f}',
                   transform=axes[idx].transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/26_error_distributions.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/26_error_distributions.png")
plt.close()

# ============================================================================
# 8.6 MONTH-BY-MONTH COMPARISON
# ============================================================================
print("\n[8.6] Month-by-Month Comparison...")

comparison_df = pd.DataFrame({'Actual': test.values}, index=test.index)

for model_name, data in models_data.items():
    comparison_df[f'{model_name}_Pred'] = data['predictions'].values
    comparison_df[f'{model_name}_Error'] = test.values - data['predictions'].values

print("\n  Month-by-Month Predictions:")
print(comparison_df.head(10).to_string())

# ============================================================================
# 8.7 ACCURACY CONSISTENCY ANALYSIS
# ============================================================================
print("\n[8.7] Accuracy Consistency Analysis...")

fig, ax = plt.subplots(figsize=(15, 6))

for idx, (model_name, data) in enumerate(models_data.items()):
    errors_pct = np.abs((test.values - data['predictions'].values) / test.values) * 100
    ax.plot(test.index, errors_pct, label=model_name, 
            marker=markers[idx % len(markers)], linewidth=2, markersize=6)

ax.set_title('Model Accuracy Over Time (Absolute Percentage Error)', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Absolute Percentage Error (%)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('visualizations/27_accuracy_over_time.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: visualizations/27_accuracy_over_time.png")
plt.close()

# ============================================================================
# 8.8 SAVE COMPARISON RESULTS
# ============================================================================
print("\n[8.8] Saving Comparison Results...")

# Save performance comparison
performance_df.to_csv('model_comparison_performance.csv', index=False)
print("  ✓ Saved: model_comparison_performance.csv")

# Save month-by-month comparison
comparison_df.to_csv('model_comparison_monthly.csv')
print("  ✓ Saved: model_comparison_monthly.csv")

# Save best model information
best_model_info = {
    'Best_Model': best_model,
    'Best_MAE': performance_df.iloc[0]['MAE'],
    'Best_RMSE': performance_df.iloc[0]['RMSE'],
    'Best_MAPE': performance_df.iloc[0]['MAPE'],
    'Best_R2': performance_df.iloc[0]['R2'],
    'Selection_Criteria': 'Lowest MAPE',
    'Number_of_Models_Compared': len(models_data)
}

best_model_df = pd.DataFrame([best_model_info])
best_model_df.to_csv('best_model_selection.csv', index=False)
print("  ✓ Saved: best_model_selection.csv")

# ============================================================================
# 8.9 RECOMMENDATIONS
# ============================================================================
print("\n[8.9] Model Recommendations...")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

best_mape = performance_df.iloc[0]['MAPE']

if best_mape < 10:
    rating = "⭐⭐⭐⭐⭐ EXCELLENT"
    recommendation = "The model shows excellent accuracy. Safe to use for business decisions."
elif best_mape < 20:
    rating = "⭐⭐⭐⭐ GOOD"
    recommendation = "The model shows good accuracy. Reliable for most business planning."
elif best_mape < 30:
    rating = "⭐⭐⭐ ACCEPTABLE"
    recommendation = "The model is acceptable but consider using with caution for critical decisions."
else:
    rating = "⭐⭐ NEEDS IMPROVEMENT"
    recommendation = "Consider collecting more data or trying advanced techniques."

print(f"\nBest Model Performance: {rating}")
print(f"MAPE: {best_mape:.2f}%")
print(f"\nRecommendation: {recommendation}")

print(f"\nNext Steps:")
print(f"  1. Use {best_model} model for future forecasting")
print(f"  2. Monitor actual vs predicted values monthly")
print(f"  3. Retrain model with new data quarterly")
print(f"  4. Consider ensemble methods if higher accuracy needed")

# ============================================================================
# 8.10 SUMMARY
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON COMPLETE!")
print("="*80)

print(f"\nModels Compared: {len(models_data)}")
print(f"Best Model: {best_model}")
print(f"Best MAPE: {best_mape:.2f}%")

print(f"\nPerformance Rankings (by MAPE):")
for idx, row in performance_df.iterrows():
    print(f"  {idx+1}. {row['Model']}: {row['MAPE']:.2f}%")

print(f"\nFiles Created:")
print(f"  • model_comparison_performance.csv")
print(f"  • model_comparison_monthly.csv")
print(f"  • best_model_selection.csv")
print(f"  • visualizations/24_performance_comparison.png")
print(f"  • visualizations/25_all_models_comparison.png")
print(f"  • visualizations/26_error_distributions.png")
print(f"  • visualizations/27_accuracy_over_time.png")

print(f"\nReady for next step: Future Forecasting")
print("="*80)
