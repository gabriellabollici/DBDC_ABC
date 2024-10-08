import matplotlib.pyplot as plt
import pandas as pd

# import LSTM log
df = pd.read_csv("lstm_training_log.csv", sep=";")
print(len(df))


# Sample DataFrame
# Step 2: Create a grouping key that increments every time 'Counter' is 0
df['group_key'] = (df['epoch'] == 0).cumsum()

# Step 3: Split the DataFrame into a dictionary of DataFrames
groups = {k: v for k, v in df.groupby('group_key')}

# Now `groups` is a dictionary with each group as a DataFrame
for key, group in groups.items():
    print("Group Key:", key)
    print(group)
    print()

epochs = list(group['epoch'])
# change this for another metric if you want to show that
performance_metric = list(group['accuracy'])
performance_metric_val = list(group['val_accuracy'])

plt.plot(epochs, performance_metric, performance_metric_val)
# ensure that epochs are 0, 1, 2, 3 instead of 0, 0.5, 1, 1.5
plt.xticks(epochs)
# start at 0 on y-axis
plt.ylim(bottom=0, top=1)
# labels
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Performance per epoch')
plt.savefig("lstm_plot.png")


# BERT
from collections import defaultdict
from ast import literal_eval
with open("metrics_callback.log", "r", encoding='UTF-8') as f:
    raw_text = f.read()

lst = raw_text.split("\n")
# the last item gave errors
lst = lst[:-1]

bert_dct = defaultdict(list)
cntr = 0
# loop through all the items, it seems that after each epoch something like this is printed:
# {'eval_loss': 0.5854439735412598, 'eval_Accuracy': 0.7052631578947368, 'eval_Precision': 0.7044433958122998, 'eval_Recall': 0.7052631578947368, 'eval_F1': 0.7019652042925746, 'eval_MSE': 0.29473684210526313, 'eval_Jensen-Shannon': 0.04535189442373448, 'eval_runtime': 0.4946, 'eval_samples_per_second': 576.212, 'eval_steps_per_second': 10.109, 'epoch': 1.0, 'step': 40}
# and then after early stopping cancels the run some sort of summary:
#{'train_runtime': 47.6929, 'train_samples_per_second': 375.297, 'train_steps_per_second': 5.871, 'total_flos': 2691099874222080.0, 'train_loss': 0.5434435367584228, 'epoch': 4.0, 'step': 160}
# after every summary we split
# note: only the training performance is calculated at the end so we cannot make a similar plot like BERT

for line in lst:
    metrics = literal_eval(line)
    if "train_runtime" in metrics:
        cntr += 1
        continue
    bert_dct[cntr].append(metrics)
    

# now we can extract one specific instance:
metrics_run1 = bert_dct[1]
performance_metric_bert = [x['eval_Accuracy'] for x in metrics_run1]
epochs_bert = list(range(1, len(performance_metric_bert) + 1))

plt.plot(epochs_bert, performance_metric_bert)
plt.xticks(epochs_bert)
# start at 0 on y-axis
plt.ylim(bottom=0, top=1)
# labels
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Performance per epoch')
plt.savefig("bert_plot.png")
