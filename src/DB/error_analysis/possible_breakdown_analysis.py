import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uuid

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_columns(row):
    breakdowns = [annotation.get('breakdown') for annotation in row]
    return pd.Series({'breakdown': breakdowns})

def common_label(lst):
    if not lst:
        return None
    return pd.Series(lst).mode().iloc[0]

def load_and_process_data(dataset_dst):
    file_names = [file_name for file_name in os.listdir(dataset_dst) if file_name.endswith('.json')]
    dfs = []

    for file_name in file_names:
        rute_file = os.path.join(dataset_dst, file_name)
        content = load_json_file(rute_file)
        df = pd.json_normalize(content['turns'], meta=['turn-index','speaker','time', 'annotation-id','utterance', 'comment','annotations'])
        df['file_name'] = file_name  
        df['uuid'] = [str(uuid.uuid4()) for _ in range(len(df))]  
        dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)

    new_columns = result_df['annotations'].apply(extract_columns)

    df_combined = pd.concat([result_df, new_columns], axis=1)
    df_combined.drop(['annotations', 'time'], axis=1, inplace=True)

    df_combined['common_label'] = df_combined['breakdown'].apply(common_label)

    df_combined['Group'] = df_combined['common_label'].notnull().cumsum()
    df_combined = df_combined.groupby('Group').agg({'common_label': 'last',
                                                    'utterance': lambda x: [f"{tipo}: {text}" for tipo, text in zip(df_combined.loc[x.index, 'speaker'], x)],
                                                    'file_name': 'first',
                                                    'uuid': 'first'})

    df_combined['utterance'] = df_combined['utterance'].apply(lambda lista: [interaction for interaction in lista if interaction.startswith('U')] +
                                                                        [interaction for interaction in lista if interaction.startswith('S')])
    df_combined = df_combined.reset_index(drop=True)

    df_combined = df_combined.rename(columns={'common_label': 'label', 'utterance': 'text'})
    df_combined = df_combined[['text', 'label', 'file_name', 'uuid']]
    df_combined = df_combined.dropna(subset=['label'])

    df_combined['text'] = df_combined['text'].apply(lambda x: ' '.join(x))

    return df_combined

def import_dbdc_data():
    train_dst = "release-v3-distrib/release-v3-distrib/dialogue_breakdown_detection/dbdc4_en_dev_labeled"
    test_dst = "release-v3-distrib/release-v3-distrib/dialogue_breakdown_detection/dbdc4_en_eval_labeled"
    
    train_breakdown = load_and_process_data(train_dst)
    test_breakdown = load_and_process_data(test_dst)

    return train_breakdown, test_breakdown

def save_processed_data():
    train_breakdown, test_breakdown = import_dbdc_data()
    combined_data = pd.concat([train_breakdown, test_breakdown], ignore_index=True)
    combined_data.to_csv('combined_breakdown_with_T.csv', index=False)

save_processed_data()

def load_combined_data():
    return pd.read_csv('combined_breakdown_with_T.csv')

def process_test_with_T(test_breakdown):
    combined_data = load_combined_data()
    
    test_data_filtered = test_breakdown[test_breakdown['text'].isin(combined_data['text'])]

    merged_data = test_data_filtered.merge(combined_data[['text', 'label']], on='text', how='left', suffixes=('', '_with_T'))

    merged_data = merged_data.rename(columns={'label': 'Label', 'label_with_T': 'Label with T'})
    merged_data['Label'] = merged_data['Label'].replace({'breakdown': 1, 'no breakdown': 0})

    def convert_label_with_T(label):
        if label == 'O':
            return 0
        elif label == 'X':
            return 1
        else:
            return label

    merged_data['Label with T'] = merged_data['Label with T'].apply(convert_label_with_T)

    final_data = merged_data[['text', 'Label', 'Label with T']]
    final_data = final_data.rename(columns={'text': 'Text'})

    test_with_T = final_data
    return test_with_T

train_breakdown, test_breakdown = import_dbdc_data()
test_breakdown = pd.DataFrame(test_breakdown)

test_with_T = process_test_with_T(test_breakdown)

classified_bert = pd.read_csv('results/bert/bert_classified.csv')
classified_lstm = pd.read_csv('results/lstm/lstm_classified.csv')
classified_llama_zero_shot = pd.read_csv('results/llama/Zero_Shot_Best_Configuration-classified.csv')
classified_llama_few_shot_10 = pd.read_csv('results/llama/Few_Shot_10_Best_Configuration-classified.csv')
classified_llama_few_shot_20 = pd.read_csv('results/llama/Few_Shot_20_Best_Configuration-classified.csv')

def add_label_with_T(df, label_with_T_df):
    df_with_T = df.merge(label_with_T_df[['Text', 'Label with T']], left_on='text', right_on='Text', how='left').drop(columns=['Text'])
    if 'true_label' in df_with_T.columns:
        df_with_T = df_with_T.drop(columns=['true_label'])
    return df_with_T

def adjust_pred_label(df):
    if 'pred_label' in df.columns:
        df['pred_label'] = df['pred_label'].replace({'breakdown': 1, 'no breakdown': 0})
    return df

classified_bert_withT = add_label_with_T(classified_bert, test_with_T)
classified_lstm_withT = add_label_with_T(classified_lstm, test_with_T)
classified_llama_zero_shot_withT = adjust_pred_label(add_label_with_T(classified_llama_zero_shot, test_with_T))
classified_llama_few_shot_10_withT = adjust_pred_label(add_label_with_T(classified_llama_few_shot_10, test_with_T))
classified_llama_few_shot_20_withT = adjust_pred_label(add_label_with_T(classified_llama_few_shot_20, test_with_T))

output_dir = 'Error_Analysis'
os.makedirs(output_dir, exist_ok=True)

graph_names = {
    "classified_bert_withT": "BERT",
    "classified_lstm_withT": "LSTM",
    "classified_llama_zero_shot_withT": "Zero-Shot LLaMA",
    "classified_llama_few_shot_10_withT": "Few-Shot 10 LLaMA",
    "classified_llama_few_shot_20_withT": "Few-Shot 20 LLaMA"
}

def create_bar_chart(df, model_name):
    t_samples = df[df['Label with T'] == 'T']
    y_pred = t_samples['pred_label'] if 'pred_label' in t_samples.columns else t_samples['Label']
    
    counts = y_pred.value_counts().sort_index()
    
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.xlabel('Predicted Label')
    plt.ylabel('Count')
    plt.title(graph_names[model_name])
    plt.close()

fig, axs = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle('Classification of the T Label for Each Model', fontsize=16)

def plot_bar_chart_in_grid(ax, df, model_name):
    t_samples = df[df['Label with T'] == 'T']
    y_pred = t_samples['pred_label'] if 'pred_label' in t_samples.columns else t_samples['Label']
    
    counts = y_pred.value_counts().sort_index()
    
    sns.barplot(x=counts.index, y=counts.values, palette='viridis', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Count')
    ax.set_title(graph_names[model_name])

plot_bar_chart_in_grid(axs[0, 0], classified_bert_withT, "classified_bert_withT")
plot_bar_chart_in_grid(axs[0, 1], classified_lstm_withT, "classified_lstm_withT")
plot_bar_chart_in_grid(axs[1, 0], classified_llama_zero_shot_withT, "classified_llama_zero_shot_withT")
plot_bar_chart_in_grid(axs[1, 1], classified_llama_few_shot_10_withT, "classified_llama_few_shot_10_withT")
plot_bar_chart_in_grid(axs[2, 0], classified_llama_few_shot_20_withT, "classified_llama_few_shot_20_withT")

fig.delaxes(axs[2, 1])

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(output_dir, 'combined_bar_charts.png'))
plt.show()

