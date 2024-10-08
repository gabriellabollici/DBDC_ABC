import json
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_results_from_json(file_path):
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

def extract_metrics_lstm(results):
    df = pd.DataFrame(results)
    
    df['Accuracy'] = df['results'].apply(lambda x: x.get('accuracy', 0))
    df['Precision'] = df['results'].apply(lambda x: x.get('precision', 0))
    df['Recall'] = df['results'].apply(lambda x: x.get('recall', 0))
    df['F1'] = df.apply(lambda row: 2 * (row['Precision'] * row['Recall']) / (row['Precision'] + row['Recall']) if (row['Precision'] + row['Recall']) > 0 else 0, axis=1)
    df['MSE'] = df['results'].apply(lambda x: x.get('mse', 0))
    df['JSD'] = df['results'].apply(lambda x: x.get('jsd', 0))
    
    return df

def extract_metrics_bert(results):
    df = pd.DataFrame(results)
    
    df['Accuracy'] = df['results'].apply(lambda x: x.get('eval_Accuracy', 0))
    df['Precision'] = df['results'].apply(lambda x: x.get('eval_Precision', 0))
    df['Recall'] = df['results'].apply(lambda x: x.get('eval_Recall', 0))
    df['F1'] = df['results'].apply(lambda x: x.get('eval_F1', 0))
    df['MSE'] = df['results'].apply(lambda x: x.get('eval_MSE', 0))
    df['JSD'] = df['results'].apply(lambda x: x.get('eval_Jensen-Shannon', 0))
    
    return df

def extract_metrics_llama(results):
    df = pd.DataFrame(results)
    
    df['prompt_name'] = df['prompt_name']
    df['Accuracy'] = df['results'].apply(lambda x: x.get('Accuracy', 0))
    df['Precision'] = df['results'].apply(lambda x: x.get('Precision', 0))
    df['Recall'] = df['results'].apply(lambda x: x.get('Recall', 0))
    df['F1'] = df['results'].apply(lambda x: x.get('F1', 0))
    df['MSE'] = df['results'].apply(lambda x: x.get('MSE', 0))
    df['JSD'] = df['results'].apply(lambda x: x.get('Jensen-Shannon Divergence', 0))
    df['Cohen\'s Kappa'] = df['results'].apply(lambda x: x.get('Cohen\'s Kappa', 0))
    
    return df

def select_best_configuration(df):
    best_config = df.loc[df['Accuracy'].idxmax()]
    return best_config

def plot_metrics(df, title, save_path=None, figsize=(10, 6)):
    epochs = range(1, len(df) + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, df['Accuracy'], marker='o', linestyle='-', color='blue', label='Accuracy')
    ax.plot(epochs, df['Precision'], marker='s', linestyle='--', color='green', label='Precision')
    ax.plot(epochs, df['Recall'], marker='^', linestyle='-.', color='orange', label='Recall')
    ax.plot(epochs, df['F1'], marker='*', linestyle=':', color='red', label='F1')
    
    ax.set_xticks(range(0, len(epochs), max(1, len(epochs)//20)))
    ax.set_xticklabels(range(1, len(epochs) + 1, max(1, len(epochs)//20)), fontsize=8)
    ax.set_xlabel('Number of Configurations')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

def plot_mse_jensen_shannon(df, title, save_path=None, figsize=(10, 6)):
    epochs = range(1, len(df) + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, df['MSE'], marker='o', linestyle='-', color='blue', label='MSE')
    ax.plot(epochs, df['JSD'], marker='s', linestyle='--', color='green', label='Jensen-Shannon')
    
    ax.set_xticks(range(0, len(epochs), max(1, len(epochs)//20)))
    ax.set_xticklabels(range(1, len(epochs) + 1, max(1, len(epochs)//20)), fontsize=8)
    ax.set_xlabel('Number of Configurations')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

def process_results(model_name, results_file, extract_metrics_fn, metrics_plot_title, mse_js_plot_title, metrics_save_path, mse_js_save_path, figsize=(10, 6)):
    results = load_results_from_json(results_file)
    df = extract_metrics_fn(results)
    best_config = select_best_configuration(df)

    print(f"Best configuration for {model_name} based on Accuracy:")
    print(best_config)
    
    plot_metrics(df, metrics_plot_title, metrics_save_path, figsize)
    plot_mse_jensen_shannon(df, mse_js_plot_title, mse_js_save_path, figsize)

def process_llama_results(results_file, figsize=(20, 10)):
    results = load_results_from_json(results_file)
    df = extract_metrics_llama(results)
    prompt_names = df['prompt_name'].unique()
    
    for prompt_name in prompt_names:
        prompt_df = df[df['prompt_name'] == prompt_name]
        metrics_plot_title = f'LLAMA Model Evaluation Metrics for {prompt_name}'
        mse_js_plot_title = f'LLAMA Model MSE and Jensen-Shannon for {prompt_name}'
        metrics_save_path = f'results/llama/{prompt_name}_metrics_plot.png'
        mse_js_save_path = f'results/llama/{prompt_name}_mse_js_plot.png'
        
        best_config = select_best_configuration(prompt_df)

        print(f"Best configuration for LLAMA with prompt '{prompt_name}' based on Accuracy:")
        print(best_config)
        
        plot_metrics(prompt_df, metrics_plot_title, metrics_save_path, figsize)
        plot_mse_jensen_shannon(prompt_df, mse_js_plot_title, mse_js_save_path, figsize)

def main():
    lstm_results_file = 'results/lstm/lstm_results.json'
    process_results(
        model_name='LSTM',
        results_file=lstm_results_file,
        extract_metrics_fn=extract_metrics_lstm,
        metrics_plot_title='LSTM Model Evaluation Metrics over Different Configurations',
        mse_js_plot_title='LSTM Model MSE and Jensen-Shannon over Different Configurations',
        metrics_save_path='results/lstm/LSTM_metrics_plot.png',
        mse_js_save_path='results/lstm/LSTM_mse_js_plot.png'
    )
    
    # BERT
    bert_results_file = 'results/bert/BERT_results.json'
    process_results(
        model_name='BERT',
        results_file=bert_results_file,
        extract_metrics_fn=extract_metrics_bert,
        metrics_plot_title='BERT Model Evaluation Metrics over Different Configurations',
        mse_js_plot_title='BERT Model MSE and Jensen-Shannon over Different Configurations',
        metrics_save_path='results/bert/BERT_metrics_plot.png',
        mse_js_save_path='results/bert/BERT_mse_js_plot.png'
    )

    # LLAMA
    llama_results_file = 'results/llama/llama_results.json'
    process_llama_results(llama_results_file, figsize=(20, 10)) 

if __name__ == '__main__':
    main()
