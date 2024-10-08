from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from scipy.spatial import distance
import numpy as np
from sklearn.preprocessing import LabelEncoder

# CLASSIFICATION METRICS

def eval_accuracy(test: list, preds: list) -> float:
    """Calculates the accuracy between two lists
    Args:
        test (list): ground truth
        preds (list): predicted
    Returns:
        float: accuracy score
    """
    acc = accuracy_score(test, preds)  
    return acc

def eval_precision(test: list, preds: list) -> float:
    """Calculates the precision between two lists
    Args:
        test (list): ground truth
        preds (list): predicted
    Returns:
        float: precision score
    """
    precision = precision_score(test, preds, average='weighted')  
    return precision

def eval_recall(test: list, preds: list) -> float:
    """Calculates the recall between two lists
    Args:
        test (list): ground truth
        preds (list): predicted
    Returns:
        float: recall score
    """
    recall = recall_score(test, preds, average='weighted')  
    return recall

def eval_f1(test: list, preds: list) -> float:
    """Calculates the f1 score between two lists
    Args:
        test (list): ground truth
        preds (list): predicted
    Returns:
        float: f1 score
    """
    f1 = f1_score(test, preds, average='weighted')  
    return f1

# DISTRIBUTION-RELATED METRICS

def eval_mse(test: list, preds: list) -> float:
    """Calculates the mean squared error between two lists of labels.
    
    Args:
        test (list): Lista de etiquetas verdaderas.
        preds (list): Lista de etiquetas predichas.
    
    Returns:
        float: Error cuadrático medio.
    """
    label_encoder = LabelEncoder()
    all_labels = test + preds
    label_encoder.fit(all_labels)
    
    true_labels_encoded = label_encoder.transform(test)
    predictions_encoded = label_encoder.transform(preds)

    mse = mean_squared_error(true_labels_encoded, predictions_encoded)

    return mse

def calculate_jensenshannon(labels1, labels2):
    """Calculates the Jensen-Shannon distance between two sets of labels.
    Args:
        labels1 (list or np.ndarray): ground truth labels
        labels2 (list or np.ndarray): predicted labels
    Returns:
        float: The Jensen-Shannon distance between the two sets of labels.
    """
    unique_labels = np.unique(np.concatenate((labels1, labels2)))
    dist1 = np.histogram(labels1, bins=len(unique_labels), density=True)[0]
    dist2 = np.histogram(labels2, bins=len(unique_labels), density=True)[0]

    dist1 = dist1 + 1e-10 
    dist2 = dist2 + 1e-10

    dist1 /= np.sum(dist1)  
    dist2 /= np.sum(dist2)

    return distance.jensenshannon(dist1, dist2)


