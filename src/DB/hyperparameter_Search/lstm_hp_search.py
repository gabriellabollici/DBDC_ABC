import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'evaluation')))

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import RMSprop
from keras.layers import LSTM, Activation, Dense, Input, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
import scipy.stats
from itertools import product

from evaluation import eval_accuracy, eval_precision, eval_recall, eval_f1, eval_mse, calculate_jensenshannon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data_processing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'results')))
from data_preprocessing import import_and_process_data

def load_config(config_path, model_type):
    config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    return config.get(model_type, {})

def train_and_evaluate(dropout_rate, batch_size, num_epochs, learning_rate, X_train, Y_train, X_test, Y_test):
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

    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=learning_rate),
    )

    model.fit(
        sequences_matrix,
        Y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5)]
    )

    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

    loss = model.evaluate(test_sequences_matrix, Y_test, verbose=0)

    Y_pred = model.predict(test_sequences_matrix)
    Y_pred_binary = (Y_pred > 0.5).astype(int)

    accuracy = eval_accuracy(Y_test.flatten(), Y_pred_binary.flatten())
    precision = eval_precision(Y_test.flatten(), Y_pred_binary.flatten())
    recall = eval_recall(Y_test.flatten(), Y_pred_binary.flatten())
    f1 = eval_f1(Y_test.flatten(), Y_pred_binary.flatten())
    mse = eval_mse(Y_test.flatten(), Y_pred_binary.flatten())
    jsd = calculate_jensenshannon(Y_test.flatten(), Y_pred_binary.flatten())

    print(f'Test set\n  Loss: {loss:.3f}\n  Accuracy: {accuracy:.3f}\n  Precision: {precision:.3f}\n  Recall: {recall:.3f}\n  MSE: {mse:.3f}\n  F1: {f1:.3f}\n  Jensen-Shannon Divergence: {jsd:.3f}')

    results_dict = {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mse': mse,
        'jensen_shannon': jsd
    }

    return results_dict


def main(config_path='../config.json', model_type='LSTM'):
    config = load_config(config_path, model_type)
    dropout_rates = config.get('dropout_rate', [0.5])
    batch_sizes = config.get('batch_sizes', [128])
    num_epochs_list = config.get('num_epochs', [10])
    learning_rates = config.get('learning_rates', [1e-3])
    output_dir = config.get('output_dir', 'results/lstm')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_list = []

    all_combinations = product(dropout_rates, batch_sizes, num_epochs_list, learning_rates)

    train_data, val_data, test_data, train_breakdown_llms, val_breakdown_llms, test_breakdown_llms, train_original_rows, test_original_rows = import_and_process_data()
    train_breakdown = pd.DataFrame(train_data)
    val_breakdown = pd.DataFrame(val_data)
    test_breakdown = pd.DataFrame(test_data)

    df = pd.concat([train_breakdown, val_breakdown, test_breakdown], axis=0, ignore_index=True)

    X = df['text']
    Y = df['label']
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1, 1)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    for dropout_rate, batch_size, num_epochs, learning_rate in all_combinations:
        print(f'Training LSTM with dropout rate {dropout_rate}, batch size {batch_size}, num epochs {num_epochs}, learning rate {learning_rate}')
        
        results_dict = train_and_evaluate(dropout_rate, batch_size, num_epochs, learning_rate, X_train, Y_train, X_val, Y_val)

        results_list.append({
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'results': results_dict
        })

    results_file = os.path.join(output_dir, 'lstm_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_list, f, indent=4)

    print(f'All configurations results stored in {results_file}')

if __name__ == "__main__":
    main()