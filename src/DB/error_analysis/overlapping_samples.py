import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'results')

classified_bert = pd.read_csv(os.path.join(results_dir, 'bert', 'bert_classified.csv'))
classified_lstm = pd.read_csv(os.path.join(results_dir, 'lstm', 'lstm_classified.csv'))
classified_llama_zero_shot = pd.read_csv(os.path.join(results_dir, 'llama', 'Zero_Shot_Best_Configuration-classified.csv'))
classified_llama_few_shot_10 = pd.read_csv(os.path.join(results_dir, 'llama', 'Few_Shot_10_Best_Configuration-classified.csv'))
classified_llama_few_shot_20 = pd.read_csv(os.path.join(results_dir, 'llama', 'Few_Shot_20_Best_Configuration-classified.csv'))

classified_bert['misclassified'] = classified_bert['true_label'] != classified_bert['pred_label']
classified_lstm['misclassified'] = classified_lstm['true_label'] != classified_lstm['pred_label']
classified_llama_zero_shot['misclassified'] = classified_llama_zero_shot['true_label'] != classified_llama_zero_shot['pred_label']
classified_llama_few_shot_10['misclassified'] = classified_llama_few_shot_10['true_label'] != classified_llama_few_shot_10['pred_label']
classified_llama_few_shot_20['misclassified'] = classified_llama_few_shot_20['true_label'] != classified_llama_few_shot_20['pred_label']

classified_bert = classified_bert.rename(columns={'misclassified': 'misclassified_bert', 'index': 'id', 'pred_label': 'pred_label_bert'})
classified_lstm = classified_lstm.rename(columns={'misclassified': 'misclassified_lstm', 'index': 'id', 'pred_label': 'pred_label_lstm'})
classified_llama_zero_shot = classified_llama_zero_shot.rename(columns={'misclassified': 'misclassified_llama_zero_shot', 'index': 'id', 'pred_label': 'pred_label_llama_zero_shot'})
classified_llama_few_shot_10 = classified_llama_few_shot_10.rename(columns={'misclassified': 'misclassified_llama_few_shot_10', 'index': 'id', 'pred_label': 'pred_label_llama_few_shot_10'})
classified_llama_few_shot_20 = classified_llama_few_shot_20.rename(columns={'misclassified': 'misclassified_llama_few_shot_20', 'index': 'id', 'pred_label': 'pred_label_llama_few_shot_20'})

merged_df = classified_bert[['id', 'text', 'true_label', 'pred_label_bert', 'misclassified_bert']]
merged_df = merged_df.merge(classified_lstm[['id', 'pred_label_lstm', 'misclassified_lstm']], on='id')
merged_df = merged_df.merge(classified_llama_zero_shot[['id', 'pred_label_llama_zero_shot', 'misclassified_llama_zero_shot']], on='id')
merged_df = merged_df.merge(classified_llama_few_shot_10[['id', 'pred_label_llama_few_shot_10', 'misclassified_llama_few_shot_10']], on='id')
merged_df = merged_df.merge(classified_llama_few_shot_20[['id', 'pred_label_llama_few_shot_20', 'misclassified_llama_few_shot_20']], on='id')

overlapping_misclassifications = merged_df[
    (merged_df['misclassified_bert'] == True) &
    (merged_df['misclassified_lstm'] == True) &
    (merged_df['misclassified_llama_zero_shot'] == True) &
    (merged_df['misclassified_llama_few_shot_10'] == True) &
    (merged_df['misclassified_llama_few_shot_20'] == True)
]

pd.set_option('display.max_colwidth', None)
print(overlapping_misclassifications['text'])
