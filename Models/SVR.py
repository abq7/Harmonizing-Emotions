import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn

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

        # Select only the relevant columns
        chunk = chunk[['_id', 'bpm', 'gain', 'length', 'arousal_predicted', 'valence_predicted', 'num_chords', 'avg_chord_duration', 'chord_labels']]
        
        # Convert empty strings to NaN
        chunk = chunk.applymap(lambda x: np.nan if x == '' else x)

        # Drop rows with missing values in relevant columns
        chunk = chunk.dropna(subset=['bpm', 'gain', 'length', 'arousal_predicted', 'valence_predicted', 'num_chords', 'avg_chord_duration'])

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

print("Data split completed")

# Print the data types of each column
print("Train data types:")
print(train_df.dtypes)
print()
print("Test data types:")
print(test_df.dtypes)
print()

# Check for missing values in relevant columns
print("Train data missing values:")
print(train_df[['bpm', 'gain', 'length', 'num_chords', 'avg_chord_duration']].isnull().sum())
print()
print("Test data missing values:")
print(test_df[['bpm', 'gain', 'length', 'num_chords', 'avg_chord_duration']].isnull().sum())
print()

# Set the maximum sequence length for chord labels
max_seq_length = 100

print("Extracting features and labels from the training data...")
train_features = train_df[['bpm', 'gain', 'length', 'num_chords', 'avg_chord_duration']].values
train_features = train_features.astype(float)  # Convert train_features to float type
train_valence = train_df['valence_predicted'].values
train_arousal = train_df['arousal_predicted'].values
print("Extraction of training features and labels completed.")

print("Extracting features and labels from the test data...")
test_features = test_df[['bpm', 'gain', 'length', 'num_chords', 'avg_chord_duration']].values
test_features = test_features.astype(float)  # Convert test_features to float type
test_valence = test_df['valence_predicted'].values
test_arousal = test_df['arousal_predicted'].values
print("Extraction of test features and labels completed.")

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert data to PyTorch tensors and move to the GPU
train_features_tensor = torch.from_numpy(train_features).float().to(device)
train_valence_tensor = torch.from_numpy(train_valence).float().to(device)
train_arousal_tensor = torch.from_numpy(train_arousal).float().to(device)
test_features_tensor = torch.from_numpy(test_features).float().to(device)
test_valence_tensor = torch.from_numpy(test_valence).float().to(device)
test_arousal_tensor = torch.from_numpy(test_arousal).float().to(device)

# Define the SVR models using PyTorch with RBF kernel
class SVRModel(nn.Module):
    def __init__(self, input_dim, gamma):
        super(SVRModel, self).__init__()
        self.gamma = gamma
        self.linear = nn.Linear(input_dim, 1)
        
    def kernel(self, x1, x2):
        # Radial Basis Function (RBF) kernel
        dist = torch.cdist(x1.unsqueeze(1), x2.unsqueeze(0), p=2)
        return torch.exp(-self.gamma * dist ** 2)
        
    def forward(self, x):
        K = self.kernel(x, self.linear.weight)
        return K @ self.linear.weight + self.linear.bias

# Define the SVR models with RBF kernel
input_dim = train_features.shape[1]
gamma = 0.1  # Adjust the gamma value as needed
valence_model = SVRModel(input_dim, gamma).to(device)
arousal_model = SVRModel(input_dim, gamma).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
valence_optimizer = torch.optim.Adam(valence_model.parameters())
arousal_optimizer = torch.optim.Adam(arousal_model.parameters())

# Train the SVR models
num_epochs = 100
batch_size = 1024

print("Training the valence model...")
for epoch in range(num_epochs):
    for i in range(0, len(train_features_tensor), batch_size):
        batch_features = train_features_tensor[i:i+batch_size]
        batch_valence = train_valence_tensor[i:i+batch_size]
        
        valence_optimizer.zero_grad()
        valence_pred = valence_model(batch_features)
        loss = criterion(valence_pred, batch_valence.view(-1, 1))
        loss.backward()
        valence_optimizer.step()
print("Valence model training completed.")

print("Training the arousal model...")
for epoch in range(num_epochs):
    for i in range(0, len(train_features_tensor), batch_size):
        batch_features = train_features_tensor[i:i+batch_size]
        batch_arousal = train_arousal_tensor[i:i+batch_size]
        
        arousal_optimizer.zero_grad()
        arousal_pred = arousal_model(batch_features)
        loss = criterion(arousal_pred, batch_arousal.view(-1, 1))
        loss.backward()
        arousal_optimizer.step()
print("Arousal model training completed.")

# Predict valence and arousal values for the test data
print("Predicting valence values for the test data...")
with torch.no_grad():
    valence_pred = valence_model(test_features_tensor).cpu().numpy().flatten()[:len(test_valence)]
print("Valence prediction completed.")

print("Predicting arousal values for the test data...")
with torch.no_grad():
    arousal_pred = arousal_model(test_features_tensor).cpu().numpy().flatten()[:len(test_arousal)]
print("Arousal prediction completed.")

# Calculate evaluation metrics
print("Calculating evaluation metrics...")
valence_mae = mean_absolute_error(test_valence, valence_pred)
valence_mse = mean_squared_error(test_valence, valence_pred)
valence_r2 = r2_score(test_valence, valence_pred)

arousal_mae = mean_absolute_error(test_arousal, arousal_pred)
arousal_mse = mean_squared_error(test_arousal, arousal_pred)
arousal_r2 = r2_score(test_arousal, arousal_pred)
print("Evaluation metrics calculation completed.")

print("Evaluation Results:")
print(f"Valence - MAE: {valence_mae:.4f}, MSE: {valence_mse:.4f}, R^2: {valence_r2:.4f}")
print(f"Arousal - MAE: {arousal_mae:.4f}, MSE: {arousal_mse:.4f}, R^2: {arousal_r2:.4f}")