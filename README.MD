## Introduction
This repository is created for the thesis project of Gabriella Bollici and is named:
ENHANCING HUMAN-COMPUTER INTERACTION LEVERAGING MISCOMMUNICATION DETECTION IN CHATBOT DIALOGUES FOR CONVERSATION BREAKDOWN

## Task definition
Generating conversational dialogue poses challenges and frequently results in scenarios where the agent produces a response that the user struggles to address adequately or that creates friction between the user and the agent. These system failures are known as **breakdowns**, and may cause users to abandon the conversation. Therefore, identifying and solving these breakdowns is considered a key element for chatbot development. Dialogue breakdown analysis, a subfield of Natural Language Processing, focuses on identifying points in conversations where users struggle to continue. 

To address this issue, conversational features extracted from users' utterances are utilized to train models capable of detecting problematic situations, also known as **miscommunications**. This study specifically targets the classification of these miscommunications, defined as "a situation when a dialogue system gives a user an inappropriate reply" (Hendriksen et al., 2021). To achieve this, we incorporate the ABC-Eval dataset (Finch et al., 2023), which contains labeled data related to 9 dialogue behaviors: empathetic responses, lack of empathy, common sense understanding, contradiction, incorrect factual information, self-contradiction, contradiction by the partner, redundancy, and instances of ignoring or providing irrelevant information such as empathetic responses, contradictions, and irrelevant information. These dialogue behaviors are used to enrich the classification process, thereby enhancing the model's ability to accurately identify breakdown-inducing utterances. In this way, beyond determining whether a breakdown has occurred or not, we delve deeper by identifying the type of dialogue behavior or miscommunication that has caused it, contributing to improving the understanding of breakdown causes.

### Format
Input: sentence
Output: 1/0 (breakdown, no breakdown)

### Example:
Input: U: Okey how is it going? S: i am just a person who is a vegan. 

Output: 1 (breakdown)

## Architecture (or link to poster?)
Link to the poster in the repository

## Repository overview
**Data** English subset of the Dialogue Breakdown Detection Challenge (DBDC) Dataset. It contains train and evaluation data. 

**Data Processing:** This module process the dialogue breakdown detection dataset and the abc_eval dataset. For the former, it loads JSON files containing annotated dialogue turns, normalizes and extracts relevant columns, and identifies the most common breakdown label in each group of interactions. The dialogues are then grouped, formatted as text, and cleaned by removing rows without labels. Afterward, the data is split into training, validation, and test sets, ensuring label balance. The breakdown labels are converted into numerical values (1 for "breakdown," 0 for "no breakdown") for classification models, while the other version is kept for language models (LLMs). Finally, the processed datasets are returned in both formats along with metadata about the original data. For the abc_eval dataset, it creates a dataframe with the miscommunications and displays it. 

**DB (dialogue breakdown):** This module represents the performance segment, where three models are compared: BERT, LSTM, and LLaMA3. This module involves hyperparameter tuning, displaying graphs, and evaluating the models using accuracy, precision, recall, F1-score, mean squared error (MSE), and Jensen-Shannon divergence (JSD). In the error analysis section, the samples that all models have misclassified are examined, while tracking how each model classified instances originally labeled as "possibly a breakdown." Additionally, the prompts used for classification with LLaMA3 are included, as well as a results folder that stores the outcomes of hyperparameter tuning, cross-validation results, and the best configuration results for each model.

**ABC_Eval:** This module represents the explainability segment, where miscommunication classification is performed using LLaMA3 and the results are analyzed. The dataset with the miscommunication labels is stored in the `behavior-classification-results.json`. The file `abc_icl.py` handles the classification for each miscommunication, and saves the results of each in the dataset_abc folder, while `dialogue_behavior_datasets.py` processes again these results and creates a dataframe with all dialogue behaviors, transforming the labels into zeros and ones (1 if the behavior is present, 0 if not). Finally, the file `dialogue_behavior_analysis.py` establishes the miscommunication-breakdown equivalence, first for each individual behavior, then for various combinations, and ultimately by combining all of them. An evaluation of all these results is conducted, using the same metrics that were used for the explainability segment. 


## Installation

Create a conda environment and install the necessary requirements:
```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
pip install -r requirements.txt
```

## Huggingface API key
For the classifier/LLaMA3_classifier.py and ABC_experiments/abc_icl.py scripts a Huggingface API key is required.
- Generate a Huggingface API key by creating an account in the access tokens tab [here](https://huggingface.co/settings/tokens).
- Create .env file and add the following line:
HF_TOKEN="<your_HF_token>"

## How to run best classification models
Run BERT classifier using the best hyperparameter config:
`python classifiers/bert_classifier.py`

Run Llama3 ICL classifier using the best hyperparameter config:
`python classifiers/LLaMA3_classifier.py`

Run LSTM classifier using the best hyperparameter config:
`python classifiers/lstm_classifier.py`

## Acknowledgement
I would like to express my gratitude to my supervisor, Frédéric Blain, for his guidance in the development of this thesis, and to Marijn van Wingerden, who subsequently took over. Both have provided invaluable perspectives and insights for the advancement of this work. 

I wish also to extend my thanks to Daan Vos, who has guided and accompanied me throughout this journey; his ideas, advice and support have been crucial in bringing this project to fruition. 