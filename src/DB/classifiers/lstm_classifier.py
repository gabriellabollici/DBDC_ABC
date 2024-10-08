import sys 
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation')))


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from keras.layers import LSTM, Activation, Dense, Input, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix
import scipy.stats
import random

from evaluation import eval_accuracy, eval_precision, eval_recall, eval_f1, eval_mse, calculate_jensenshannon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
from data_preprocessing import import_and_process_data

SEED_VALUE = 42

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def save_cv_results(output_path, results):
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def save_results(config_path, results):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['results'] = results
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

def generate_heatmap(true_labels, pred_labels, output_dir):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=["negative", "positive"], columns=["negative", "positive"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix Heatmap - LSTM Best Configuration')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    heatmap_file = os.path.join(output_dir, 'best_config_heatmap.png')
    plt.savefig(heatmap_file)
    plt.close()

    print(f'Confusion Matrix Heatmap saved as {heatmap_file}')

def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

def calculate_jsd(p, q):
    p = np.asarray(p) + 1e-10
    q = np.asarray(q) + 1e-10
    m = 0.5 * (p + q)
    return 0.5 * (scipy.stats.entropy(p, m) + scipy.stats.entropy(q, m)).sum()

def train_and_evaluate(config, X_train, Y_train, X_val, Y_val):
    max_words = 1000
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

    def RNN():
        inputs = Input(name='inputs', shape=[max_len])
        layer = Embedding(max_words, 50)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(256, name='FC1')(layer)
        layer = Dense(1, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)

        return model

    model = RNN()
    model.summary()

    learning_rate = config.get('learning_rate', 0.001)
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 20)

    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=learning_rate),
    )

    model.fit(
        sequences_matrix,
        Y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(sequence.pad_sequences(tok.texts_to_sequences(X_val), maxlen=max_len), Y_val),
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)]
    )

    val_sequences = tok.texts_to_sequences(X_val)
    val_sequences_matrix = sequence.pad_sequences(val_sequences, maxlen=max_len)

    loss = model.evaluate(val_sequences_matrix, Y_val, verbose=0)

    Y_pred = model.predict(val_sequences_matrix)
    Y_pred_binary = (Y_pred > 0.5).astype(int).flatten()

    accuracy = eval_accuracy(Y_val.flatten(), Y_pred_binary)
    precision = eval_precision(Y_val.flatten(), Y_pred_binary)
    recall = eval_recall(Y_val.flatten(), Y_pred_binary)
    f1 = eval_f1(Y_val.flatten(), Y_pred_binary)
    mse = eval_mse(Y_val.flatten(), Y_pred_binary)
    jsd = calculate_jensenshannon(Y_val.flatten(), Y_pred_binary)

    results_dict = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'jsd': jsd
    }

    return results_dict, Y_val, Y_pred_binary, model

def main(config_path=os.path.join(os.getcwd(), "src", "DB", "results", "lstm", "best_config_lstm.json"), output_path=os.path.join(os.getcwd(), "src", "DB", "results", "lstm", "cv_lstm.json")):
    set_seed(SEED_VALUE)
    
    config = load_config(config_path)
    if config is None:
        print("Failed to load configuration.")
        return

    output_dir = config.get('output_dir', os.path.join(os.getcwd(), "src", 'DB', 'results', 'lstm'))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_data, val_data, test_data, train_breakdown_llms, val_breakdown_llms, test_breakdown_llms, train_original_rows, test_original_rows = import_and_process_data()
    train_breakdown = pd.DataFrame(train_data)
    val_breakdown = pd.DataFrame(val_data)
    test_breakdown = pd.DataFrame(test_data)

    X_train_val = pd.concat([train_breakdown, val_breakdown], axis=0, ignore_index=True)['text']
    Y_train_val = pd.concat([train_breakdown, val_breakdown], axis=0, ignore_index=True)['label']
    X_test = test_breakdown['text']
    Y_test = test_breakdown['label']

    le = LabelEncoder()
    Y_train_val = le.fit_transform(Y_train_val)
    Y_test = le.transform(Y_test)
    Y_train_val = Y_train_val.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED_VALUE)
    cv_results = []

    for train_index, val_index in kf.split(X_train_val):
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]

        results, _, _, _ = train_and_evaluate(config, X_train, Y_train, X_val, Y_val)
        cv_results.append(results)


    save_cv_results(output_path, cv_results)
    print(f'Cross-validation results stored in {output_path}')

    results, true_labels, pred_labels, model = train_and_evaluate(config, X_train_val, Y_train_val, X_test, Y_test)
    save_results(config_path, results)


    generate_heatmap(true_labels, pred_labels, output_dir)

    df_classified = pd.DataFrame({
        'index': test_breakdown.index,
        'text': test_breakdown['text'],
        'true_label': true_labels.flatten(),
        'pred_label': pred_labels.flatten()
    })

    print(df_classified)

    classified_csv_path = os.path.join(output_dir, 'lstm_classified.csv')
    df_classified.to_csv(classified_csv_path, index=False)
    print(f'Classified results saved as {classified_csv_path}')

    model_save_path = os.path.join(output_dir, 'model.h5')
    model.save(model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    main()
