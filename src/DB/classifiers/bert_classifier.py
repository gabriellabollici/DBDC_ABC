import sys 
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, IntervalStrategy, set_seed
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from datasets import Dataset, DatasetDict
import scipy.stats
from typing import Optional, Tuple, List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
from data_preprocessing import import_and_process_data
from evaluation.evaluation import eval_accuracy, eval_f1, eval_precision, eval_mse, eval_recall, calculate_jensenshannon

set_seed(42)

def load_results_file(file_name):
    results_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results', file_name)
    with open(results_path, 'r') as f:
        return json.load(f)
    
def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    try:
        absolute_config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'results', config_path)
        with open(absolute_config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None


def save_cv_results(output_path: str, results: List[Dict[str, Any]]) -> None:
    """
    Save cross-validation results to a specified output file.

    Args:
        output_path (str): Path to the output JSON file.
        results (list): List of cross-validation results to save.
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def save_results(config_path: str, results: Dict[str, Any]) -> None:
    """
    Save the evaluation results to the configuration file.

    Args:
        config_path (str): Path to the configuration file.
        results (dict): Dictionary containing the results to be saved.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['results'] = results
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

def calculate_jsd(labels: np.ndarray, probs: np.ndarray) -> float:
    """
    Calculate Jensen-Shannon Divergence between labels and predicted probabilities.

    Args:
        labels (np.ndarray): True labels in one-hot encoded format.
        probs (np.ndarray): Predicted probabilities.

    Returns:
        float: Jensen-Shannon Divergence value.
    """
    labels_one_hot = np.eye(probs.shape[1])[labels]
    m = 0.5 * (labels_one_hot + probs)
    jsd = 0.5 * (scipy.stats.entropy(labels_one_hot.T, m.T) + scipy.stats.entropy(probs.T, m.T))
    return jsd.mean()

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = eval_accuracy(labels, predictions)
    precision = eval_precision(labels, predictions)
    recall = eval_recall(labels, predictions)
    f1 = eval_f1(labels, predictions)
    
    mse = eval_mse(labels, predictions)
    js_divergence = calculate_jensenshannon(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'jsd': js_divergence
    }

def train_and_evaluate(
    model_name: str, 
    learning_rate: float, 
    batch_size: int, 
    num_epochs: int, 
    weight_decay: float, 
    train_dataset: Dataset, 
    val_dataset: Dataset, 
    test_dataset: Optional[Dataset] = None
) -> Tuple[Dict[str, float], List[int], List[int], Trainer, torch.nn.Module, Optional[Dataset]]:
    """
    Train and evaluate a model using the specified parameters and datasets.

    Args:
        model_name (str): Name of the pre-trained model.
        learning_rate (float): Learning rate for training.
        batch_size (int): Batch size for training and evaluation.
        num_epochs (int): Number of epochs to train.
        weight_decay (float): Weight decay value.
        train_dataset (Dataset): Dataset for training.
        val_dataset (Dataset): Dataset for validation.
        test_dataset (Optional[Dataset]): Dataset for testing.

    Returns:
        tuple: Evaluation metrics, true labels, predicted labels, trainer, model, and test dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    tokenized_datasets = DatasetDict({
        'train': train_dataset.map(tokenize_function, batched=True),
        'val': val_dataset.map(tokenize_function, batched=True)
    })
    
    if test_dataset is not None:
        tokenized_datasets['test'] = test_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="test-trainer",
        evaluation_strategy=IntervalStrategy.EPOCH,
        save_total_limit=5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        save_strategy=IntervalStrategy.EPOCH,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

    val_results = trainer.evaluate(eval_dataset=tokenized_datasets["val"])
    val_predictions = trainer.predict(test_dataset=tokenized_datasets["val"])
    val_pred_labels = np.argmax(val_predictions.predictions, axis=-1).tolist()
    val_true_labels = list(tokenized_datasets["val"]['label'])

    if test_dataset is not None:
        test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
        test_predictions = trainer.predict(test_dataset=tokenized_datasets["test"])
        test_pred_labels = np.argmax(test_predictions.predictions, axis=-1).tolist()
        test_true_labels = list(tokenized_datasets["test"]['label'])
        results = compute_metrics((test_predictions.predictions, test_true_labels))
        return results, test_true_labels, test_pred_labels, trainer, model, test_dataset
    else:
        results = compute_metrics((val_predictions.predictions, val_true_labels))
        return results, val_true_labels, val_pred_labels

def generate_heatmap(true_labels: List[int], pred_labels: List[int], output_dir: str) -> None:
    """
    Generate and save a confusion matrix heatmap of true vs. predicted labels.

    Args:
        true_labels (list): List of true labels.
        pred_labels (list): List of predicted labels.
        output_dir (str): Path to save the generated heatmap image.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=["negative", "positive"], columns=["negative", "positive"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix Heatmap - BERT Best Configuration')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    heatmap_file = os.path.join(output_dir, 'best_config_heatmap.png')
    plt.savefig(heatmap_file)
    plt.close()

    print(f'Confusion Matrix Heatmap saved as {heatmap_file}')

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'bert'))


    output_dir = os.path.join(base_dir)
    config_path = os.path.join(base_dir, 'best_config_bert.json')
    output_path = os.path.join(base_dir, 'cv_bert.json')

    config = load_config(config_path)
    if config is None:
        print("Failed to load configuration.")
        return

    checkpoint = config.get('checkpoint', 'bert-base-cased')
    learning_rate = config.get('learning_rate', 1e-05)
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 10)
    weight_decay = config.get('weight_decay', 0.01)
    output_dir = config.get('output_dir', os.path.join(os.getcwd(), "src", "DB", "results", "bert"))

    train_data, val_data, test_data, train_breakdown_llms, val_breakdown_llms, test_breakdown_llms, train_original_rows, test_original_rows = import_and_process_data()
    train_breakdown = pd.DataFrame(train_data)
    val_breakdown = pd.DataFrame(val_data)
    test_breakdown = pd.DataFrame(test_data)

    train_breakdown.replace({"breakdown": 1, "no breakdown": 0}, inplace=True)
    val_breakdown.replace({"breakdown": 1, "no breakdown": 0}, inplace=True)
    test_breakdown.replace({"breakdown": 1, "no breakdown": 0}, inplace=True)

    full_dataset = pd.concat([train_breakdown, val_breakdown], ignore_index=True)

    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_results = []

    for train_index, val_index in kf.split(full_dataset):
        train_subset = full_dataset.iloc[train_index]
        val_subset = full_dataset.iloc[val_index]

        train_dataset = Dataset.from_pandas(train_subset)
        val_dataset = Dataset.from_pandas(val_subset)

        results, _, _ = train_and_evaluate(checkpoint, learning_rate, batch_size, num_epochs, weight_decay, train_dataset, val_dataset)
        cv_results.append(results)

    save_cv_results(output_path, cv_results)
    print(f'Cross-validation results saved in: {output_path}')

    train_dataset = Dataset.from_pandas(train_breakdown)
    val_dataset = Dataset.from_pandas(val_breakdown)
    test_dataset = Dataset.from_pandas(test_breakdown)
    results, true_labels, pred_labels, trainer, model, _ = train_and_evaluate(checkpoint, learning_rate, batch_size, num_epochs, weight_decay, train_dataset, val_dataset, test_dataset)

    save_results(config_path, results)

    generate_heatmap(true_labels, pred_labels, output_dir)
    classified_csv_path = os.path.join(output_dir, 'bert_classified.csv')

    df_classified = pd.DataFrame({
        'index': test_dataset.index,
        'text': test_dataset['text'],
        'true_label': true_labels,
        'pred_label': pred_labels
    })

    df_classified.to_csv(classified_csv_path, index=False)
    print(f'Classified results saved as {classified_csv_path}')

    model.save_pretrained(output_dir)
    trainer.save_model(output_dir)

if __name__ == '__main__':
    main()