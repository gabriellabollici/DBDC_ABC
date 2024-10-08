import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_results_from_json(file_path):
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

def extract_metrics_lstm(results):
    df = pd.DataFrame(results)
    
    df['Accuracy'] = df['results'].apply(lambda x: x.get('accuracy', 0))
    df['Precision'] = df['results'].apply(lambda x: x.get('precision', 0))
    df['Recall'] = df['results'].apply(lambda x: x.get('recall', 0))
    df['F1'] = df['results'].apply(lambda x: x.get('f1', 0))

    
    return df

def extract_metrics_bert(results):
    df = pd.DataFrame(results)
    
    df['Accuracy'] = df['results'].apply(lambda x: x.get('eval_Accuracy', 0))
    df['Precision'] = df['results'].apply(lambda x: x.get('eval_Precision', 0))
    df['Recall'] = df['results'].apply(lambda x: x.get('eval_Recall', 0))
    df['F1'] = df['results'].apply(lambda x: x.get('eval_F1', 0))

    return df

def select_best_configuration(df):
    best_config = df.loc[df['Accuracy'].idxmax()]
    return best_config

def plot_comparison(best_lstm_config, best_bert_config, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    lstm_values = [best_lstm_config[metric] for metric in metrics]
    bert_values = [best_bert_config[metric] for metric in metrics]
    
    x = range(len(metrics))
    
    ax.bar(x, lstm_values, width=0.4, label='LSTM', align='center')
    ax.bar([i + 0.4 for i in x], bert_values, width=0.4, label='BERT', align='center')
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Comparison of Best Configuration Classification Metrics: LSTM vs BERT')
    ax.set_xticks([i + 0.2 for i in x])
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()

def main():
    lstm_results_file = 'results/lstm/lstm_results.json'
    bert_results_file = 'results/bert/BERT_results.json'
    
    lstm_results = load_results_from_json(lstm_results_file)
    bert_results = load_results_from_json(bert_results_file)
    
    lstm_df = extract_metrics_lstm(lstm_results)
    bert_df = extract_metrics_bert(bert_results)
    
    best_lstm_config = select_best_configuration(lstm_df)
    best_bert_config = select_best_configuration(bert_df)

    print("Best LSTM configuration based on Accuracy:")
    print(best_lstm_config)
    print("\nBest BERT configuration based on Accuracy:")
    print(best_bert_config)
    
    comparison_save_path = 'results/comparison/LSTM_vs_BERT_metrics_comparison.png'

    plot_comparison(best_lstm_config, best_bert_config, comparison_save_path)

if __name__ == '__main__':
    main()

