import json
import os
import sys
import numpy as np
import transformers
import torch
import pandas as pd
import string
#from dotenv import load_dotenv
#load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../DB/prompts')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_processing')))

from data_preprocessing import import_and_process_data
from prompts_abc_fs import *

torch.cuda.empty_cache()

# Define the output directory relative to the script's location
script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, "datasets_abc")
os.makedirs(output_dir, exist_ok=True)

# Load dataset
train_data, val_data, test_data, train_breakdown_llms, val_breakdown_llms, test_breakdown_llms, train_original_rows, test_original_rows = import_and_process_data()

# Define prompts
prompts_dict = {
    'commonsense_contradiction': commonsense_contradiction,
    'ignore': ignore,
    'incorrect_fact': incorrect_fact,
    'irrelevant': irrelevant,
    'lack_of_empathy': lack_of_empathy,
    'partner_contradiction': partner_contradiction,
    'self_contradiction': self_contradiction,
    'redundant': redundant,
    'empathetic': empathetic
}

# Convert test data to lists of dictionaries
test_data = [{'utterance': entry['text'], 'label': entry['label']} for entry in test_breakdown_llms]
test_labels = [entry['label'] for entry in test_breakdown_llms]

def load_config(filename):
    with open(filename, "r") as f:
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

def clean_label(label):
    # Remove punctuation and convert into lowercase but keep spaces 
    label = label.strip().lower().translate(str.maketrans('', '', string.punctuation))
    return label

def classify_with_prompt(prompt_name, prompt, data, labels, pipeline, temperature, top_p_value, quantization):
    predictions = []
    problematic_instances = []

    # Replace underscores with spaces
    positive_label = prompt_name.replace('_', ' ')
    negative_label = f"no {positive_label}"

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
        
        predicted_label = clean_label(output)

        # Print details to diagnose the issue 
        print(f"Utterance: {utterance}")
        print(f"Generated Output: {output}")
        print(f"Cleaned Predicted Label: {predicted_label}")
        print(f"Positive Label: {positive_label}, Negative Label: {negative_label}")
        print(f"Predicted Label Length: {len(predicted_label)}, Positive Label Length: {len(positive_label)}, Negative Label Length: {len(negative_label)}")

        if predicted_label not in [positive_label, negative_label, 'unknown']:
            print(f"Warning: The predicted label '{predicted_label}' is not in possible labels. Assigning 'unknown'.")
            predicted_label = "unknown"
            problematic_instances.append(idx)
        
        predictions.append(predicted_label)

    # Ensure predictions and test_labels lengths match
    test_labels = labels[:len(predictions)]

    result_df = pd.DataFrame({
        'utterance': [entry['utterance'] for entry in data],
        'predicted_label': predictions
    })

    return result_df

# Model configured with the best configuration from Few-Shot 
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
quantization = False
temperature = 0.7
top_p_value = 0.6

pipeline = setup_pipeline(model_id, quantization)

# Create and store datasets
for prompt_name, prompt in prompts_dict.items():
    result_df = classify_with_prompt(prompt_name, prompt, test_data, test_labels, pipeline, temperature, top_p_value, quantization)
    output_file = os.path.join(output_dir, f"{prompt_name}_dataset.csv")
    result_df.to_csv(output_file, index=False)
    print(f"Saved dataset: {output_file}")
