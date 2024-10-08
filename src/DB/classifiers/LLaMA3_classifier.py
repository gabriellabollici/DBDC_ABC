import sys 
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import transformers
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import string
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import scipy.stats
from typing import List, Dict, Union
#from dotenv import load_dotenv
#load_dotenv()


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
import prompts.prompts as prompts
from data_preprocessing import import_and_process_data
from evaluation.evaluation import eval_accuracy, eval_f1, eval_precision, eval_mse, eval_recall

# Set random seed for reproducibility
torch.cuda.empty_cache()
SEED_VALUE = 42

# Load dataset
train_data, val_data, test_data, train_breakdown_llms, val_breakdown_llms, test_breakdown_llms, train_original_rows, test_original_rows = import_and_process_data()

data_breakdown = train_breakdown_llms + val_breakdown_llms
labels_breakdown = [entry['label'] for entry in data_breakdown]

data_breakdown = [{'utterance': entry['text'], 'label': entry['label']} for entry in data_breakdown]

test_data = [{'utterance': entry['text'], 'label': entry['label']} for entry in test_breakdown_llms]
test_labels = [entry['label'] for entry in test_breakdown_llms]

# Load breakdown prompts
breakdown_prompt_fs_10 = prompts.Few_Shot_10
breakdown_prompt_fs_20 = prompts.Few_Shot_20
breakdown_prompt_zs = prompts.Zero_Shot

def load_config(filename: str) -> dict:
    """
    Load configuration settings from a JSON file.
    
    Args:
        filename (str): Path to the JSON configuration file.
        
    Returns:
        dict: Configuration data.
    """
    with open(filename, "r") as f:
        return json.load(f)

def setup_pipeline(model_id: str, quantization: bool) -> transformers.Pipeline:
    """
    Set up a text generation pipeline using a specified model ID with optional quantization.
    
    Args:
        model_id (str): Model identifier for the LLaMA model.
        quantization (bool): Whether to use quantization for memory efficiency.
        
    Returns:
        transformers.Pipeline: Text generation pipeline.
    """
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

def calculate_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the Jensen-Shannon Divergence between two probability distributions.
    
    Args:
        p (np.ndarray): Probability distribution 1.
        q (np.ndarray): Probability distribution 2.
        
    Returns:
        float: Jensen-Shannon Divergence value.
    """
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    m = 0.5 * (p + q)
    return 0.5 * (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m))

def classify_with_prompt(prompt_name: str, prompt: str, data: List[Dict[str, str]], labels: List[str], pipeline: transformers.Pipeline, 
                         temperature: float, top_p_value: float, quantization: bool, output_dir: str) -> Dict[str, Union[str, float]]:
    """
    Classify a list of data points using a provided prompt and a pre-trained language model pipeline.
    
    Args:
        prompt_name (str): Name of the prompt being used.
        prompt (str): Prompt text.
        data (List[Dict[str, str]]): List of data points with utterances and labels.
        labels (List[str]): List of true labels corresponding to the data points.
        pipeline (transformers.Pipeline): Language model pipeline for generating predictions.
        temperature (float): Temperature setting for text generation.
        top_p_value (float): Top-p value for nucleus sampling.
        quantization (bool): Whether the model is quantized.
        output_dir (str): Directory to save the output results.
    
    Returns:
        Dict[str, Union[str, float]]: A dictionary containing classification metrics.
    """
    predictions = []
    problematic_instances = []

    translator = str.maketrans('', '', string.punctuation)

    for idx, entry in enumerate(data):
        text = entry['utterance']
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
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

    # Generate and save confusion matrix heatmap
    cm = confusion_matrix(test_labels, predictions, labels=["breakdown", "no breakdown", "unknown"])
    cm_df = pd.DataFrame(cm, index=["breakdown", "no breakdown", "unknown"], columns=["breakdown", "no breakdown", "unknown"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
    # Change name depending on the prompt
    if prompt_name == "breakdown_fs_10":
        title_name = "Few Shot 10 Best Configuration"
    elif prompt_name == "breakdown_fs_20":
        title_name = "Few Shot 20 Best Configuration"
    elif prompt_name == "breakdown_zs":
        title_name = "Zero Shot Best Configuration"
    else:
        title_name = prompt_name
    plt.title(f'Confusion Matrix Heatmap - {title_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    heatmap_file = os.path.join(output_dir, f'{title_name}_heatmap.png')
    plt.savefig(heatmap_file)
    plt.close()

    results = {
        "prompt_name": title_name,
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

    # Save classified data
    classified_df = pd.DataFrame({
        'index': range(len(predictions)),
        'text': [entry['utterance'] for entry in data],
        'true_label': test_labels,
        'pred_label': predictions
    })

    classified_csv_path = os.path.join(output_dir, f'{title_name.replace(" ", "_")}-classified.csv')
    classified_df.to_csv(classified_csv_path, index=False)
    print(f'Classified results saved as {classified_csv_path}')

    return results


def convert_to_serializable(obj: Union[dict, list, np.ndarray, np.float32, np.float64, np.int32, np.int64]) -> Union[dict, list, float, int]:
    """
    Convert objects to a JSON-serializable format.
    
    Args:
        obj (Union[dict, list, np.ndarray, np.float32, np.float64, np.int32, np.int64]): Object to convert.
        
    Returns:
        Union[dict, list, float, int]: JSON-serializable object.
    """
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
    
def main():
    llama_config = load_config(os.path.join(os.getcwd(), "src", "DB", "results", "llama", "best_config_llama.json"))
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    output_dir = os.path.join(os.getcwd(), "src", "DB", "results", "llama")
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    cv_results = []
    
    for config in llama_config:
        prompt_name = config['prompt_name']
        temperature = config['temperature']
        top_p_value = config['p_value']
        quantization = config['quantization']
        
        if prompt_name == "breakdown_fs_10":
            prompt = breakdown_prompt_fs_10
        elif prompt_name == "breakdown_fs_20":
            prompt = breakdown_prompt_fs_20
        elif prompt_name == "breakdown_zs":
            prompt = breakdown_prompt_zs
        else:
            raise ValueError(f"Unknown prompt name: {prompt_name}")
        
        pipeline = setup_pipeline(model_id, quantization)
        
        k = 10
        kf = KFold(n_splits=k, shuffle=True, random_state=SEED_VALUE)
        
        prompt_cv_results = []
        
        for train_index, val_index in kf.split(data_breakdown):
            val_subset = [data_breakdown[i] for i in val_index]
            val_labels = [labels_breakdown[i] for i in val_index]
            
            result = classify_with_prompt(prompt_name, prompt, val_subset, val_labels, pipeline, temperature, top_p_value, quantization, output_dir)
            prompt_cv_results.append(result)
        
        cv_results.append({prompt_name: prompt_cv_results})
        
        result = classify_with_prompt(prompt_name, prompt, test_data, test_labels, pipeline, temperature, top_p_value, quantization, output_dir)
        all_results.append(result)
        
        serializable_result = convert_to_serializable(result)
        print(json.dumps(serializable_result, indent=4))
    
    cv_json_filename = os.path.join(output_dir, "cv_results_llama.json")
    with open(cv_json_filename, 'w') as f:
        serializable_cv_results = convert_to_serializable(cv_results)
        json.dump(serializable_cv_results, f, indent=4)
    
    json_filename = os.path.join(output_dir, "best_config_llama_results.json")
    with open(json_filename, 'w') as f:
        serializable_results = convert_to_serializable(all_results)
        json.dump(serializable_results, f, indent=4)
    
    print(f'All configurations results stored in {json_filename}')
    print(f'Cross-validation results stored in {cv_json_filename}')


if __name__ == "__main__":
    main()