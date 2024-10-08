import json
import os
import sys
import torch
import random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback, IntervalStrategy, set_seed
from transformers import TrainerCallback, TrainerState, TrainerControl
from datasets import Dataset, DatasetDict
from itertools import product
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
#from evaluation.evaluation import evaluation
from evaluation.evaluation import eval_accuracy, eval_f1, eval_precision, eval_mse, eval_recall, calculate_jensenshannon
from data_preprocessing import import_and_process_data

# Set the seed value
SEED_VALUE = 42
set_seed(SEED_VALUE)

def load_config(config_path, model_type):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config.get(model_type, {})

def load_previous_results(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def find_best_configuration(results):
    best_result = max(results, key=lambda x: x['results']['eval_Accuracy'])
    return best_result

class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, filename='metrics.log'):
        self.filename = filename

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.log_history:
            metrics = state.log_history[-1]  # Get the last logged metrics
            with open(self.filename, 'a') as f:
                f.write(json.dumps(metrics) + '\n')


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = eval_accuracy(labels.tolist(), predictions.tolist())
    precision = eval_precision(labels.tolist(), predictions.tolist())
    recall = eval_recall(labels.tolist(), predictions.tolist())
    f1 = eval_f1(labels.tolist(), predictions.tolist())
    
    mse = eval_mse(labels.tolist(), predictions.tolist())
    js_divergence = calculate_jensenshannon(labels.tolist(), predictions.tolist())
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MSE': mse,
        'Jensen-Shannon': js_divergence
    }

def train_and_evaluate(model_name, learning_rate, batch_size, num_epochs, weight_decay, seed=42):
    set_seed(seed)
    train_data, val_data, test_data, train_breakdown_llms, val_breakdown_llms, test_breakdown_llms, train_original_rows, test_original_rows = import_and_process_data()

    train_breakdown = pd.DataFrame(train_data)
    val_breakdown = pd.DataFrame(val_data)
    test_breakdown = pd.DataFrame(test_data)


    train_breakdown.replace({"breakdown": 1, "no breakdown": 0}, inplace=True)
    train_breakdown = train_breakdown.infer_objects(copy=False)
    val_breakdown.replace({"breakdown": 1, "no breakdown": 0}, inplace=True)
    val_breakdown = val_breakdown.infer_objects(copy=False)
    test_breakdown.replace({"breakdown": 1, "no breakdown": 0}, inplace=True)
    test_breakdown = test_breakdown.infer_objects(copy=False)

    train_breakdown = Dataset.from_pandas(train_breakdown)
    val_breakdown = Dataset.from_pandas(val_breakdown)
    test_breakdown = Dataset.from_pandas(test_breakdown)
    dataset = DatasetDict({"train": train_breakdown, "val": val_breakdown, "test": test_breakdown})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=seed).select(range(1000))
    small_eval_dataset = tokenized_datasets["val"].shuffle(seed=seed).select(range(50))

    data_collator = DataCollatorWithPadding(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        "test-trainer",
        evaluation_strategy = IntervalStrategy.EPOCH,
        eval_steps=50,
        save_total_limit = 5, # Only last 5 models are saved. Older ones are deleted.
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        metric_for_best_model = "F1",
        load_best_model_at_end = True,
        save_strategy=IntervalStrategy.EPOCH,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=2),  MetricsLoggerCallback(filename='metrics_callback.log')]
    )
    trainer.train()

    tokenized_test_dataset = tokenized_datasets["test"]
    test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset)

    print("Test Results:")
    for metric_name, metric_value in test_results.items():
        print(f"{metric_name}: {metric_value}")

    predictions = trainer.predict(test_dataset=tokenized_test_dataset)
    print("Predictions:", predictions.predictions)
    print("Labels:", predictions.label_ids)
    print("Metrics:", predictions.metrics)
    
    return test_results, trainer, tokenized_test_dataset, trainer.state.log_history

def main(config_path=os.path.join(os.getcwd(), "src", "DB", "config.json"), model_type='BERT', results_path='results/bert/BERT_results.json'):
    config = load_config(config_path, model_type)
    checkpoint = config.get('checkpoint', 'bert-base-cased')
    learning_rates = config.get('learning_rates', [2e-5])
    batch_sizes = config.get('batch_sizes', [16])
    num_epochs_list = config.get('num_epochs', [5])
    weight_decay_list = config.get('weight_decay', [0.01])
    output_dir = config.get('output_dir', 'results/bert')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_list = []
    for lr, batch_size, num_epochs, weight_decay in product(learning_rates, batch_sizes, num_epochs_list, weight_decay_list):
        print(f'Training {checkpoint} with learning rate {lr}, batch size {batch_size}, num epochs {num_epochs}, weight decay {weight_decay}')
        results, trainer, tokenized_test_dataset, log_history = train_and_evaluate(checkpoint, lr, batch_size, num_epochs, weight_decay)
        results_list.append({
            'learning_rate': lr,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'weight_decay': weight_decay,
            'results': results
        })

    results_file = os.path.join(output_dir, f'{model_type}_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_list, f, indent=4)

    print(f'Results stored in {results_file}')

    previous_results = load_previous_results(results_path)
    best_config = find_best_configuration(previous_results)
    
    print("Best Configuration:")
    print(json.dumps(best_config, indent=4))


if __name__ == '__main__':
    main()
