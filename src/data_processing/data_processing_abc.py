import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

pd.set_option('display.max_colwidth', None)

script_dir = os.path.dirname(os.path.abspath(__file__))
abc_eval = os.path.join(script_dir, '..', 'ABC_Eval', 'ABC_Experiments', 'behavior-classification-results.json')

with open(abc_eval, 'r') as f:
    data = json.load(f)

dfs = []

for key, value in data.items():
    df = pd.json_normalize(value, 
                           meta=['user', 'system', ['behaviors', 'self contradiction'], ['behaviors', 'empathetic'], ['behaviors', 'lack of empathy'], ['behaviors', 'irrelevant'], 
                                 ['behaviors','ignore'], ['behaviors','incorrect fact'],['behaviors','partner contradiction'], ['behaviors','redundant']])
    
    df.rename(columns=lambda x: x.replace('behaviors', '').strip(), inplace=True)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df.columns = ['user', 'system', 'self_contradiction_human',
              'self_contradiction_gpt', 'self_contradiction_specialized',
              'empathetic_human', 'empathetic_gpt', 'empathetic_specialized',
              'lack_of_empathy_human', 'lack_of_empathy_gpt', 'lack_of_empathy_specialized',
              'irrelevant_human', 'irrelevant_gpt', 'irrelevant_specialized',
              'ignore_human', 'ignore_gpt', 'ignore_specialized',
              'incorrect_fact_human', 'incorrect_fact_gpt', 'incorrect_fact_specialized',
              'commonsense_contradiction_human', 'commonsense_contradiction_gpt',
              'partner_contradiction_human', 'partner_contradiction_gpt',
              'redundant_human', 'redundant_gpt']

pd.set_option('display.max_colwidth', 1000)

def combine_columns(row, prefix):
    column_names = [column for column in df.columns if column.startswith(prefix)]
    values = [row[column] for column in column_names]
    return 1 if values.count(1) >= values.count(0) else 0

df['self_contradiction'] = df.apply(lambda row: combine_columns(row, 'self_contradiction'), axis=1)
df['empathetic'] = df.apply(lambda row: combine_columns(row, 'empathetic'), axis=1)
df['lack_of_empathy'] = df.apply(lambda row: combine_columns(row, 'lack_of_empathy'), axis=1)
df['irrelevant'] = df.apply(lambda row: combine_columns(row, 'irrelevant'), axis=1)
df['incorrect_fact'] = df.apply(lambda row: combine_columns(row, 'incorrect_fact'), axis=1)
df['ignore'] = df.apply(lambda row: combine_columns(row, 'ignore'), axis=1)
df['commonsense_contradiction'] = df.apply(lambda row: combine_columns(row, 'commonsense_contradiction'), axis=1)
df['partner_contradiction'] = df.apply(lambda row: combine_columns(row, 'partner_contradiction'), axis=1)
df['redundant'] = df.apply(lambda row: combine_columns(row, 'redundant'), axis=1)

columns_to_drop = [column for column in df.columns if any(substring in column for substring in ['human', 'specialized', 'gpt'])]
df.drop(columns=columns_to_drop, inplace=True)

df['utterance'] = 'U:' + df['user'].astype(str) + '. S:' + df['system'].astype(str)
columns = df.columns.tolist()
columns = ['utterance'] + [column for column in columns if column != 'utterance']
df = df[columns]
df.drop(columns=['user', 'system'], inplace=True)

def create_dataframe_by_label(df, label):
    label_df = df[['utterance', label]].copy()  
    label_df[label] = label_df[label].astype('object') 
    label_df.loc[:, label] = label_df[label].replace({0: 'no ' + label, 1: label})
    new_df = label_df.rename(columns={'utterance': 'text', label: 'label'})
    return new_df

self_contradiction_df = create_dataframe_by_label(df, 'self_contradiction')
irrelevant_df = create_dataframe_by_label(df, 'irrelevant')
empathetic_df = create_dataframe_by_label(df, 'empathetic')
lack_of_empathy_df = create_dataframe_by_label(df, 'lack_of_empathy')
incorrect_fact_df = create_dataframe_by_label(df, 'incorrect_fact')
ignore_df = create_dataframe_by_label(df, 'ignore')
commonsense_contradiction_df = create_dataframe_by_label(df, 'commonsense_contradiction')
partner_contradiction_df = create_dataframe_by_label(df, 'partner_contradiction')
redundant_df = create_dataframe_by_label(df, 'redundant')

def print_sample_rows(dfs_list):
    start_index = 0
    for df in dfs_list:
        label_column = 'label'
        print(f"Dataset: {df['label'].iloc[0].replace('no ', '')}")
        no_label_rows = df[df[label_column].str.startswith('no')].iloc[start_index:start_index + 5]
        label_rows = df[~df[label_column].str.startswith('no')].iloc[start_index:start_index + 5]
        
        print("Rows with 'no' label:")
        print(no_label_rows)
        print("\nRows without 'no' label:")
        print(label_rows)
        print("\n" + "-"*80 + "\n")
        
        start_index += 10  # Increment start index to ensure different rows for next dataset

datasets = [self_contradiction_df, irrelevant_df, empathetic_df, lack_of_empathy_df, incorrect_fact_df, ignore_df, commonsense_contradiction_df, partner_contradiction_df, redundant_df]
print_sample_rows(datasets)

