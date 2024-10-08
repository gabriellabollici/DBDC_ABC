import json
import os
import sys
import numpy as np
import transformers
import torch
import pandas as pd
import string
import scipy.stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_processing')))

import prompts.prompts as prompts
from data_preprocessing import import_and_process_data
from evaluation.evaluation import eval_accuracy, eval_f1, eval_precision, eval_mse, eval_recall, calculate_jensenshannon

torch.cuda.empty_cache()

# Load dataset
train_data, val_data, test_data, train_breakdown_llms, val_breakdown_llms, test_breakdown_llms, train_original_rows, test_original_rows = import_and_process_data()

# Verify first rows and lengths
def print_dataset_info(dataset, name):
    print(f"{name} (first rows):")
    print(pd.DataFrame(dataset).head())
    print(f"Length: {len(dataset)}\n")


print_dataset_info(train_breakdown_llms, "Train Breakdown LLMS (breakdown/no breakdown labels)")
print_dataset_info(val_breakdown_llms, "Validation Breakdown LLMS (breakdown/no breakdown labels)")
print_dataset_info(test_breakdown_llms, "Test Breakdown LLMS (breakdown/no breakdown labels)")

data_breakdown = train_breakdown_llms + val_breakdown_llms
labels_breakdown = [entry['label'] for entry in data_breakdown]

# Convert to lists of dictionaries
data_breakdown = [{'utterance': entry['text'], 'label': entry['label']} for entry in data_breakdown]

# Convert test data to lists of dictionaries
test_data = [{'utterance': entry['text'], 'label': entry['label']} for entry in test_breakdown_llms]
test_labels = [entry['label'] for entry in test_breakdown_llms]

# Load breakdown prompts
breakdown_prompt_fs_10 = prompts.Few_Shot_10
breakdown_prompt_fs_20 = prompts.Few_Shot_20
breakdown_prompt_zs = prompts.Zero_Shot

def load_config(filename):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config.json')
    with open(config_path, "r") as f:
        return json.load(f)

def setup_pipeline(model_id, quantization):
    if quantization:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            token=os.getenv('HF_TOKEN'),
            device_map="auto",
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )
    else:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )
    return pipeline

def calculate_jsd(p, q):
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    m = 0.5 * (p + q)
    return 0.5 * (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m))

def classify_with_prompt(prompt_name, prompt, data, labels, pipeline, temperature, top_p_value, quantization):
    predictions = []
    problematic_instances = []

    translator = str.maketrans('', '', string.punctuation)

    for idx, entry in enumerate(data):
        utterance = entry['utterance']
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": utterance},
        ]

        prompt_text = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        eos_token_id = pipeline.tokenizer.eos_token_id
        empty_token_id = pipeline.tokenizer.convert_tokens_to_ids("")

        terminators = [token for token in [eos_token_id, empty_token_id] if token is not None]

        print(f"Evaluating with temperature: {temperature}, top_p: {top_p_value}, quantization: {quantization}")

        outputs = pipeline(
            prompt_text,
            max_new_tokens=20,
            eos_token_id=terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p_value,
        )

        output = outputs[0]["generated_text"][len(prompt_text):]
        
        predicted_label = output.strip().lower().translate(translator)

        if predicted_label not in ['breakdown', 'no breakdown', 'unknown']:
            print(f"Warning: The predicted label {predicted_label} is not in possible labels. Assigning 'unknown'.")
            predicted_label = "unknown"
            problematic_instances.append(idx)
        
        predictions.append(predicted_label)

    # Ensure predictions and test_labels lengths match
    test_labels = labels[:len(predictions)]

    for i, response in enumerate(predictions):
        if response == "not a breakdown":
            predictions[i] = "no breakdown"
        if predictions[i] not in ['breakdown', 'no breakdown', 'unknown']:
            print(f"Warning: Unexpected response: {response}")
            predictions[i] = "unknown"
            problematic_instances.append(i)

    accuracy_result = eval_accuracy(test_labels, predictions)
    print(f'{prompt_name} Accuracy:', accuracy_result)

    precision_result = eval_precision(test_labels, predictions)
    print(f'{prompt_name} Precision:', precision_result)

    recall_result = eval_recall(test_labels, predictions)
    print(f'{prompt_name} Recall:', recall_result)

    mse_result = eval_mse(test_labels, predictions)
    print(f'{prompt_name} MSE:', mse_result)

    f1_result = eval_f1(test_labels, predictions)
    print(f'{prompt_name} F1:', f1_result)

    def labels_to_prob(labels):
        label_dict = {'breakdown': 0, 'no breakdown': 1, 'unknown': 2}
        prob = np.zeros((len(labels), 3))  # 3 classes
        for i, label in enumerate(labels):
            prob[i, label_dict[label]] = 1
        return prob.tolist()  # Convert to list

    # In classify_with_prompt, before calling calculate_jsd
    test_label_probs = labels_to_prob(test_labels)
    prediction_probs = labels_to_prob(predictions)

    # Calculate Jensen-Shannon Divergence
    jsd_result = calculate_jsd(np.array(test_label_probs), np.array(prediction_probs))

    if np.isnan(jsd_result).any():
        jsd_result = np.nanmean(jsd_result)  # Handle NaN values if they occur
    else:
        jsd_result = float(jsd_result)  # Ensure single result

    print(f'{prompt_name} Jensen-Shannon Divergence:', jsd_result)

    results = {
        "prompt_name": prompt_name,
        "temperature": float(temperature),
        "p_value": float(top_p_value),
        "quantization": quantization,
        "results": {
            "Accuracy": float(accuracy_result),
            "Precision": float(precision_result),
            "Recall": float(recall_result),
            "F1": float(f1_result),
            "MSE": float(mse_result),
            "Jensen-Shannon Divergence": float(jsd_result)
        }
    }

    return results

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

if __name__ == "__main__":
    llama_config = load_config("config.json")['LLAMA']
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    output_dir = llama_config.get("output_dir", "results/LLAMA")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    best_f1 = 0
    best_config = None

    # Test all combinations of hyperparameters for all prompts
    for prompt_name, prompt in [
        ("breakdown_fs_10", breakdown_prompt_fs_10),
        ("breakdown_fs_20", breakdown_prompt_fs_20),
        ("breakdown_zs", breakdown_prompt_zs)
    ]:
        for quantization in llama_config['quantization']:
            for temp in llama_config['temperature']:
                for top_p_value in llama_config['top_p']:
                    pipeline = setup_pipeline(model_id, quantization)
                    result = classify_with_prompt(prompt_name, prompt, test_data, test_labels, pipeline, temp, top_p_value, quantization)
                    all_results.append(result)

                    # Print results to verify
                    serializable_result = convert_to_serializable(result)
                    print(json.dumps(serializable_result, indent=4))

                    if result["results"]["F1"] > best_f1:
                        best_f1 = result["results"]["F1"]
                        best_config = {
                            'prompt_name': prompt_name,
                            'prompt': prompt,
                            'temperature': temp,
                            'p_value': top_p_value,
                            'quantization': quantization
                        }

    json_filename = os.path.join(output_dir, "llama_results_abc.json")
    with open(json_filename, 'w') as f:
        serializable_results = convert_to_serializable(all_results)
        json.dump(serializable_results, f, indent=4)

    print(f'All configurations results stored in {json_filename}')

    if best_config:
        print(f'Best configuration: {best_config}')
