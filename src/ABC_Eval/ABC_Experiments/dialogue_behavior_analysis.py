import pandas as pd
import os
import sys
import itertools

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DB/prompts')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DB/evaluation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_processing')))

from data_preprocessing import import_and_process_data
from prompts_abc_fs import *
from evaluation import eval_accuracy, eval_f1, eval_precision, eval_mse, eval_recall, calculate_jensenshannon

# Define the path to the combined dataset relative to the script's location
script_dir = os.path.dirname(__file__)
dataset_path = os.path.join(script_dir, "datasets_abc", "combined_dataset.csv")

# Load the combined dataset
combined = pd.read_csv(dataset_path, sep=",")
print(f"Total rows in combined dataset: {len(combined)}")

# Remove the 'empathetic' column if it exists
if 'empathetic' in combined.columns:
    combined = combined.drop(columns=['empathetic'])
    print("Removed 'empathetic' column.")

# Convert all relevant values in the dataset
for col in combined.columns:
    combined[col] = combined[col].apply(lambda x: 0 if str(x).strip().lower().startswith("no") else 1)

# Load datasets
train_data, val_data, test_data, train_breakdown_llms, val_breakdown_llms, test_breakdown_llms, train_original_rows, test_original_rows = import_and_process_data()

# Extract labels from test data
y = [x['label'] for x in test_data]

# Extract column names, excluding the first and last column
column_names = combined.columns[1:-1].tolist()

# Convert the columns in the combined dataset to lists
dct = {col: combined[col].tolist() for col in column_names}

def eval_metrics(preds, y):
    f1 = eval_f1(preds, y)
    accuracy = eval_accuracy(y, preds)
    precision = eval_precision(y, preds)
    recall = eval_recall(y, preds)
    mse = eval_mse(y, preds)
    jsd = calculate_jensenshannon(y, preds)
    return accuracy, precision, recall, f1, mse, jsd

# Calculate evaluation metrics for each behavior
result_dct = {behavior: eval_f1(preds, y) for behavior, preds in dct.items()}
sorted_individual_results = dict(sorted(result_dct.items(), key=lambda item: item[1], reverse=True))

print("Individual results sorted by F1-score:")
for behavior, score in sorted_individual_results.items():
    print(f"{behavior}: {score}")

def evaluate_combinations(combinations, combined, y):
    result_dct = {}
    for combi in combinations:
        result = [1 if any(pred == 1 for pred in preds) else 0 for preds in zip(*(combined[col] for col in combi))]
        result_dct["_".join(combi)] = eval_f1(result, y)
    return dict(sorted(result_dct.items(), key=lambda item: item[1], reverse=True))

# Evaluate and sort combinations of columns from 2 to 8 columns
for i in range(2, 9):
    combi_dct = evaluate_combinations(itertools.combinations(column_names, i), combined, y)
    best_combination = next(iter(combi_dct.items()))
    best_combination_name = best_combination[0]
    best_combination_preds = [1 if any(pred == 1 for pred in preds) else 0 for preds in zip(*(combined[col] for col in best_combination_name.split('_')))]
    best_combination_metrics = eval_metrics(best_combination_preds, y)
    
    print(f"Best combination of {i} columns sorted by F1-score:")
    print(f"{best_combination_name}: Accuracy={best_combination_metrics[0]}, Precision={best_combination_metrics[1]}, Recall={best_combination_metrics[2]}, F1={best_combination_metrics[3]}, MSE={best_combination_metrics[4]}, JSD={best_combination_metrics[5]}")
