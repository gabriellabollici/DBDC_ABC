import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_columns(dataset_row):
    breakdowns = [annotation.get('breakdown') for annotation in dataset_row]
    return pd.Series({'breakdown': breakdowns})

def common_label(lst):
    if not lst:
        return None
    return pd.Series(lst).mode().iloc[0]

def load_and_process_data(breakdown_dst):
    file_names = [file_name for file_name in os.listdir(breakdown_dst) if file_name.endswith('.json')]
    dfs = []

    for file_name in file_names:
        rute_file = os.path.join(breakdown_dst, file_name)
        content = load_json_file(rute_file)
        df = pd.json_normalize(content['turns'], meta=['turn-index', 'speaker', 'time', 'annotation-id', 'utterance', 'comment', 'annotations'])
        dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)

    new_columns = result_df['annotations'].apply(extract_columns)

    df_combined = pd.concat([result_df, new_columns], axis=1)
    df_combined.drop(['annotations', 'time'], axis=1, inplace=True)

    df_combined['common_label'] = df_combined['breakdown'].apply(common_label)

    df_combined['Group'] = df_combined['common_label'].notnull().cumsum()
    df_combined = df_combined.groupby('Group').agg({'common_label': 'last',
                                                    'utterance': lambda x: [f"{tipo}: {text}" for tipo, text in zip(df_combined.loc[x.index, 'speaker'], x)]})

    df_combined['utterance'] = df_combined['utterance'].apply(lambda lista: [interaction for interaction in lista if interaction.startswith('U')] +
                                                                        [interaction for interaction in lista if interaction.startswith('S')])
    df_combined = df_combined.reset_index(drop=True)

    df_combined = df_combined.rename(columns={'common_label': 'label', 'utterance': 'text'})
    df_combined = df_combined[['text', 'label']]
    df_combined = df_combined.dropna(subset=['label'])

    df_combined['text'] = df_combined['text'].apply(lambda x: ' '.join(x))

    df_combined.loc[df_combined['label'].str.contains('T'), 'label'] = 'X'

    return df_combined, result_df.shape[0]

def import_dbdc_data(path_to_train_data: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'dialogue_breakdown_detection', 'dbdc4_en_dev_labeled'),
                     path_to_test_data: str = os.path.join(os.path.dirname(__file__), '..', 'data', 'dialogue_breakdown_detection', 'dbdc4_en_eval_labeled')):
    train_dst = path_to_train_data
    test_dst = path_to_test_data
    
    train_breakdown, train_original_rows = load_and_process_data(train_dst)
    train_breakdown.replace({"X": 'breakdown', 'O': 'no breakdown'}, inplace=True)
    
    test_breakdown, test_original_rows = load_and_process_data(test_dst)
    test_breakdown.replace({"X": 'breakdown', 'O': 'no breakdown'}, inplace=True)

    return train_breakdown, test_breakdown, train_original_rows, test_original_rows

def divide_data(train_data, test_data):
    total_files = pd.concat([train_data, test_data], axis=0, ignore_index=True).drop_duplicates()
    train_data, temp_data = train_test_split(total_files, test_size=0.2, random_state=42, stratify=total_files['label'])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['label'])

    return train_data, val_data, test_data

def import_and_process_data():
    train_breakdown, test_breakdown, train_original_rows, test_original_rows = import_dbdc_data()

    train_data, val_data, test_data = divide_data(train_breakdown, test_breakdown)

    train_breakdown_llms = train_data.copy()
    val_breakdown_llms = val_data.copy()
    test_breakdown_llms = test_data.copy()

    train_data.replace({"breakdown": 1, "no breakdown": 0}, inplace=True)
    val_data.replace({"breakdown": 1, "no breakdown": 0}, inplace=True)
    test_data.replace({"breakdown": 1, "no breakdown": 0}, inplace=True)

    train_data = train_data.to_dict(orient='records')
    val_data = val_data.to_dict(orient='records')
    test_data = test_data.to_dict(orient='records')
    
    train_breakdown_llms = train_breakdown_llms.to_dict(orient='records')
    val_breakdown_llms = val_breakdown_llms.to_dict(orient='records')
    test_breakdown_llms = test_breakdown_llms.to_dict(orient='records')

    return train_data, val_data, test_data, train_breakdown_llms, val_breakdown_llms, test_breakdown_llms, train_original_rows, test_original_rows
