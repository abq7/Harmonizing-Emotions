import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create a mapping of chord labels to integer values
chord_mapping = {}

def extract_chords_metadata(chunk):
    def extract_num_chords(x):
        if isinstance(x, dict) and 'chordSequence' in x:
            return len(x['chordSequence'])
        else:
            return 0
    
    def extract_avg_chord_duration(x):
        if isinstance(x, dict) and 'chordSequence' in x and 'duration' in x and len(x['chordSequence']) > 0:
            return x['duration'] / len(x['chordSequence'])
        else:
            return 0
    
    def extract_chord_labels(x):
        global chord_mapping
        if isinstance(x, dict) and 'chordSequence' in x:
            labels = [chord['label'] for chord in x['chordSequence'] if 'label' in chord]
            for label in labels:
                if label not in chord_mapping:
                    chord_mapping[label] = len(chord_mapping)
            return [chord_mapping[label] for label in labels]
        else:
            return []
    
    chunk['num_chords'] = chunk['chords_metadata'].apply(extract_num_chords)
    chunk['avg_chord_duration'] = chunk['chords_metadata'].apply(extract_avg_chord_duration)
    chunk['chord_labels'] = chunk['chords_metadata'].apply(extract_chord_labels)
    return chunk

# Load the song_extracted_all_entries.json file in batches
batch_size = 1000
train_data = []
test_data = []
total_entries = 0
training_entries = 0

with pd.read_json('song_extracted_all_entries.json', lines=True, chunksize=batch_size) as reader:
    for i, chunk in enumerate(reader):
        # Check if the required columns are present in the DataFrame
        required_columns = ['_id', 'bpm', 'gain', 'length', 'arousal_predicted', 'valence_predicted', 'chords_metadata']
        missing_columns = [col for col in required_columns if col not in chunk.columns]

        if missing_columns:
            raise ValueError(f"The following required columns are missing in the DataFrame: {missing_columns}")

        # Extract chords metadata
        chunk = extract_chords_metadata(chunk)

        # Replace empty string values with np.nan for relevant columns
        chunk['bpm'] = pd.to_numeric(chunk['bpm'], errors='coerce')
        chunk['gain'] = pd.to_numeric(chunk['gain'], errors='coerce')
        chunk['length'] = pd.to_numeric(chunk['length'], errors='coerce')
        chunk['arousal_predicted'] = chunk['arousal_predicted'].apply(lambda x: np.nan if x == '' else x)
        chunk['valence_predicted'] = chunk['valence_predicted'].apply(lambda x: np.nan if x == '' else x)

        # Select only the relevant columns
        chunk = chunk[['_id', 'bpm', 'gain', 'length', 'arousal_predicted', 'valence_predicted', 'num_chords', 'avg_chord_duration', 'chord_labels']]

        # Filter rows with missing or zero values
        chunk = chunk[(chunk['bpm'].notnull()) & (chunk['gain'].notnull()) & (chunk['length'].notnull()) & (chunk['arousal_predicted'].notnull()) & (chunk['valence_predicted'].notnull()) & (chunk['num_chords'] != 0) & (chunk['avg_chord_duration'] != 0)]

        if len(chunk) >= 5:  # Adjust the threshold as needed
            # Split the current batch into train and test sets
            train_chunk, test_chunk = train_test_split(chunk, test_size=0.2, random_state=42)

            # Append the train and test data to the respective lists
            train_data.append(train_chunk)
            test_data.append(test_chunk)

            training_entries += len(train_chunk)
        else:
            # If the number of samples is below the threshold, append the entire chunk to the train data
            train_data.append(chunk)
            training_entries += len(chunk)

        total_entries += len(chunk)
        print(f"Loaded {total_entries} entries")

        # Print the last 5 rows of the chunk
        print(chunk.tail())

        # Print the column names
        print(chunk.columns)

print(f"Data loading completed. Total entries: {total_entries}, Training entries: {training_entries}")

# Concatenate the train and test data
train_df = pd.concat(train_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)

# Convert target variables to float
train_df['valence_predicted'] = train_df['valence_predicted'].astype(float)
train_df['arousal_predicted'] = train_df['arousal_predicted'].astype(float)
test_df['valence_predicted'] = test_df['valence_predicted'].astype(float)
test_df['arousal_predicted'] = test_df['arousal_predicted'].astype(float)

# Prepare the input features and target variables
X_train = train_df[['bpm', 'gain', 'length', 'num_chords', 'avg_chord_duration']]
y_train_valence = train_df['valence_predicted']
y_train_arousal = train_df['arousal_predicted']

X_test = test_df[['bpm', 'gain', 'length', 'num_chords', 'avg_chord_duration']]
y_test_valence = test_df['valence_predicted']
y_test_arousal = test_df['arousal_predicted']

# Create and train the Gradient Boosting models
valence_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
arousal_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

valence_model.fit(X_train, y_train_valence)
arousal_model.fit(X_train, y_train_arousal)

# Make predictions on the test set
valence_pred = valence_model.predict(X_test)
arousal_pred = arousal_model.predict(X_test)

# Calculate evaluation metrics
valence_mae = mean_absolute_error(y_test_valence, valence_pred)
valence_mse = mean_squared_error(y_test_valence, valence_pred)
valence_r2 = r2_score(y_test_valence, valence_pred)

arousal_mae = mean_absolute_error(y_test_arousal, arousal_pred)
arousal_mse = mean_squared_error(y_test_arousal, arousal_pred)
arousal_r2 = r2_score(y_test_arousal, arousal_pred)

print("Evaluation Results:")
print(f"Valence - MAE: {valence_mae:.4f}, MSE: {valence_mse:.4f}, R^2: {valence_r2:.4f}")
print(f"Arousal - MAE: {arousal_mae:.4f}, MSE: {arousal_mse:.4f}, R^2: {arousal_r2:.4f}")