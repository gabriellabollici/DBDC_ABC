import pandas as pd
import os

# Function to process the dataset and transform labels
def process_dataset(file_path, behavior):
    # Read the CSV file
    dataset = pd.read_csv(file_path)

    # Rename the label column with the behavior name
    behavior_label = behavior.replace('_', ' ')
    dataset = dataset.rename(columns={'predicted_label': behavior_label})

    return dataset

# Define the output directory relative to the script's location
script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, "datasets_abc")

# List of behaviors
behaviors = [
    'commonsense_contradiction', 'ignore', 'incorrect_fact', 'irrelevant',
    'lack_of_empathy', 'partner_contradiction', 'self_contradiction',
    'redundant', 'empathetic'
]

# List to store the processed DataFrames
dataframes = []

# Process each dataset
for behavior in behaviors:
    file_path = os.path.join(output_dir, f"{behavior}_dataset.csv")
    processed_dataset = process_dataset(file_path, behavior)
    dataframes.append(processed_dataset)

# Concatenate all DataFrames into one
final_dataset = pd.concat([df[['utterance'] + [df.columns[-1]]] for df in dataframes], axis=1)

# Remove duplicate columns of 'utterance'
final_dataset = final_dataset.loc[:, ~final_dataset.columns.duplicated()]

# Create the 'miscommunication' column
behavior_columns = [col for col in final_dataset.columns if col != 'utterance']
final_dataset['miscommunication'] = final_dataset[behavior_columns].max(axis=1)

# Count zeros and ones in the 'miscommunication' column
miscommunication_counts = final_dataset['miscommunication'].value_counts()
zeros_count = miscommunication_counts.get(0, 0)
ones_count = miscommunication_counts.get(1, 0)

# Display the resulting DataFrame
print(final_dataset)

# Display the number of zeros and ones in the 'miscommunication' column
print(f"Number of zeros in the 'miscommunication' column: {zeros_count}")
print(f"Number of ones in the 'miscommunication' column: {ones_count}")

# Save the resulting DataFrame to a CSV file
output_file = os.path.join(output_dir, "combined_dataset.csv")
final_dataset.to_csv(output_file, index=False)
