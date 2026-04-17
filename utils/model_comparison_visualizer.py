import numpy as np
import joblib # For loading models saved with sklearn's joblib/pickle
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

DATA_PATH = './data/'
MODELS_PATH = './models/'
OUTPUTS_PATH = './outputs/updated/'

# Define the models to load
model_names = ['gradientboosting', 'randomforest', 'svm', 'votingclassifier']
model_filenames = {
    'gradientboosting': 'gradientboosting_model.pkl',
    'randomforest': 'randomforest_model.pkl',
    'svm': 'svm_model.pkl',
    'votingclassifier': 'votingclassifier_model.pkl'
}

# --- 1. Load Data ---
print("Loading test data...")
try:
    X_test_scaled = np.load(os.path.join(DATA_PATH, 'X_test_scaled.npy'))
    y_test = np.load(os.path.join(DATA_PATH, 'y_test.npy'))
    print(f"X_test_scaled shape: {X_test_scaled.shape}")
    print(f"y_test shape: {y_test.shape}")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure 'X_test_scaled.npy' and 'y_test.npy' are in the '{DATA_PATH}' directory.")
    exit()

# --- 2. Load Models and Evaluate ---
print("\nLoading and evaluating models...")
model_accuracies = {}
model_reports = {} # To store classification reports for more detailed metrics

for name, filename in model_filenames.items():
    model_path = os.path.join(MODELS_PATH, filename)
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) # output_dict for easier parsing
        
        model_accuracies[name] = accuracy
        model_reports[name] = report
        print(f"  {name.capitalize()} Accuracy: {accuracy:.4f}")
    except FileNotFoundError:
        print(f"  Warning: Model file '{filename}' not found. Skipping {name}.")
    except Exception as e:
        print(f"  Error loading or evaluating {name}: {e}")

if not model_accuracies:
    print("No models were successfully loaded or evaluated. Exiting.")
    exit()

# --- 3. Visualization ---

# Visualization 1: Bar Chart of Model Accuracies
plt.figure(figsize=(10, 6))
sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()), palette='viridis')
plt.title('Comparison of Model Accuracies on Test Set')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(min(model_accuracies.values()) - 0.05, max(model_accuracies.values()) + 0.02) # Adjust y-axis for better visibility
for i, accuracy in enumerate(model_accuracies.values()):
    plt.text(i, accuracy + 0.005, f'{accuracy:.2f}', ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
accuracy_bar_chart_path = os.path.join(OUTPUTS_PATH, 'model_accuracy_bar_chart.png')
plt.savefig(accuracy_bar_chart_path)
print(f"\nSaved Model Accuracy Bar Chart to {accuracy_bar_chart_path}")

# Visualization 2: Radar Chart for Multi-Metric Comparison

# Extract metrics for radar chart
metrics_for_radar = ['accuracy', 'weighted avg_precision', 'weighted avg_recall', 'weighted avg_f1-score']
# We'll need the number of classes to get actual precision/recall/f1 per class,
# but for a general overview, weighted avg is good. If you have only 2 classes,
# you could use '0_precision', '1_precision' etc.
# For simplicity, we'll use overall 'accuracy' and 'weighted avg' for precision, recall, f1.

radar_data = {}
for model_name, report in model_reports.items():
    data = []
    # Note: 'accuracy' is directly available in the report output_dict at the top level
    data.append(report.get('accuracy', 0)) # Get overall accuracy
    data.append(report.get('weighted avg', {}).get('precision', 0))
    data.append(report.get('weighted avg', {}).get('recall', 0))
    data.append(report.get('weighted avg', {}).get('f1-score', 0))
    radar_data[model_name] = data

if radar_data:
    df_radar = pd.DataFrame.from_dict(radar_data, orient='index', columns=['Accuracy', 'Precision', 'Recall', 'F1-score'])
    
    # --- Plotting the Radar Chart ---
    labels = list(df_radar.columns)
    num_vars = len(labels)

    # Calculate angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop for plotting

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot each model's data
    for i, (model_name, row) in enumerate(df_radar.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1] # Complete the loop
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=model_name.capitalize(), color=sns.color_palette('tab10')[i])
        ax.fill(angles, values, color=sns.color_palette('tab10')[i], alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Set y-axis limits and ticks
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)

    plt.title('Multi-Metric Comparison of Classifiers (Radar Chart)', size=16, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    radar_chart_path = os.path.join(OUTPUTS_PATH, 'model_radar_chart.png')
    plt.savefig(radar_chart_path)
    print(f"Saved Multi-Metric Radar Chart to {radar_chart_path}")

else:
    print("Could not generate radar chart as no model reports were available.")

print("\nScript finished.")