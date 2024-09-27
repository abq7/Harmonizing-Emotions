import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from transformers import BertTokenizer, BertModel
from itertools import zip_longest
from keras.preprocessing.sequence import pad_sequences

# Extract additional features from the dataset
def extract_additional_features(df):
    return df[['num_chords', 'avg_chord_duration']]

def extract_chords_features(df):
    return df['chord_labels'].tolist()

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
        if isinstance(x, dict) and 'chordSequence' in x:
            return [chord['label'] for chord in x['chordSequence'] if 'label' in chord]
        else:
            return []
    
    chunk['num_chords'] = chunk['chords_metadata'].apply(extract_num_chords)
    chunk['avg_chord_duration'] = chunk['chords_metadata'].apply(extract_avg_chord_duration)
    chunk['chord_labels'] = chunk['chords_metadata'].apply(extract_chord_labels)
    return chunk

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the song_extracted_all_entries.json file in batches
batch_size = 1000
train_data = []
test_data = []

with pd.read_json('song_extracted_all_entries.json', lines=True, chunksize=batch_size) as reader:
    total_entries = 0
    for i, chunk in enumerate(reader):
        # Check if the required columns are present in the DataFrame
        required_columns = ['_id', 'lyrics', 'bpm', 'gain', 'length', 'arousal_predicted', 'valence_predicted', 'chords_metadata']
        missing_columns = [col for col in required_columns if col not in chunk.columns]

        if missing_columns:
            raise ValueError(f"The following required columns are missing in the DataFrame: {missing_columns}")

        # Fill missing values with appropriate placeholders or default values
        chunk['lyrics'] = chunk['lyrics'].fillna('')
        chunk['arousal_predicted'] = chunk['arousal_predicted'].fillna(0)
        chunk['valence_predicted'] = chunk['valence_predicted'].fillna(0)
        
        # Extract chords metadata
        chunk = extract_chords_metadata(chunk)
        
        # Convert 'arousal_predicted' and 'valence_predicted' to float
        chunk['arousal_predicted'] = chunk['arousal_predicted'].astype(float)
        chunk['valence_predicted'] = chunk['valence_predicted'].astype(float)
        
        # Select only the relevant columns
        chunk = chunk[['_id', 'lyrics', 'bpm', 'gain', 'length', 'arousal_predicted', 'valence_predicted', 'num_chords', 'avg_chord_duration', 'chord_labels']]
        
        # Tokenize the lyrics using the BERT tokenizer
        def tokenize_lyrics(lyrics):
            return tokenizer.encode(lyrics, add_special_tokens=True, max_length=512, truncation=True, padding='max_length')

        chunk['input_ids'] = chunk['lyrics'].apply(tokenize_lyrics)
        
        # Split the current batch into train and test sets
        train_chunk, test_chunk = train_test_split(chunk, test_size=0.2, random_state=42)
        
        # Append the train and test data to the respective lists
        train_data.append(train_chunk)
        test_data.append(test_chunk)
        
        total_entries += len(chunk)
        print(f"Loaded {total_entries} entries")
        
        # Print the last 5 rows of the chunk
        print(chunk.tail())
        
        # Print the column names
        print(chunk.columns)
        
        # Check if the desired number of entries has been processed
        if total_entries >= 1000000:
            break

print("Data loading completed.")

# Combine the train and test data from all batches
train_df = pd.concat(train_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)

print("Lyrics tokenization and data split completed")

# Extract additional features for train and test sets
train_additional_features = extract_additional_features(train_df)
test_additional_features = extract_additional_features(test_df)

train_chord_labels = extract_chords_features(train_df)
test_chord_labels = extract_chords_features(test_df)

print("Feature extraction completed")

# Filter out samples with missing valence or arousal values
train_df = train_df.dropna(subset=['valence_predicted', 'arousal_predicted'])
test_df = test_df.dropna(subset=['valence_predicted', 'arousal_predicted'])

# Convert 'valence_predicted' and 'arousal_predicted' columns to numeric
train_df['valence_predicted'] = pd.to_numeric(train_df['valence_predicted'], errors='coerce')
train_df['arousal_predicted'] = pd.to_numeric(train_df['arousal_predicted'], errors='coerce')
test_df['valence_predicted'] = pd.to_numeric(test_df['valence_predicted'], errors='coerce')
test_df['arousal_predicted'] = pd.to_numeric(test_df['arousal_predicted'], errors='coerce')

# Preprocess additional features
scaler = StandardScaler()
train_additional_features = scaler.fit_transform(train_additional_features)
test_additional_features = scaler.transform(test_additional_features)

# Convert chord labels to numeric values
unique_labels = set(label for labels in train_chord_labels for label in labels)
label_to_index = {label: index for index, label in enumerate(unique_labels)}
index_to_label = {index: label for label, index in label_to_index.items()}

train_chord_labels_numeric = [[label_to_index[label] for label in labels] for labels in train_chord_labels]
test_chord_labels_numeric = [[label_to_index[label] for label in labels] for labels in test_chord_labels]

# Pad chord labels to a fixed length
max_length = max(len(labels) for labels in train_chord_labels_numeric)
padding_value = len(unique_labels)
train_chord_labels_padded = pad_sequences(train_chord_labels_numeric, maxlen=max_length, padding='post', value=padding_value)
test_chord_labels_padded = pad_sequences(test_chord_labels_numeric, maxlen=max_length, padding='post', value=padding_value)

# Convert padded numeric labels back to strings
def convert_to_labels(padded_labels):
    return [[index_to_label[index] if index != padding_value else 'N/A' for index in labels] for labels in padded_labels]

# One-hot encode chord labels
encoder = OneHotEncoder(handle_unknown='ignore')
train_chord_labels = encoder.fit_transform(train_chord_labels).toarray()
test_chord_labels = encoder.transform(test_chord_labels).toarray()

# Create a PyTorch Dataset
def data_generator(dataframe, additional_features, chord_labels, batch_size, max_length, padding_value):
    num_samples = len(dataframe)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices = indices[start:end]

        batch_data = []
        for index in batch_indices:
            row = dataframe.iloc[index]
            input_ids = torch.tensor(row['input_ids'], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            additional_features_batch = torch.tensor(additional_features[index], dtype=torch.float)
            
            # Pad the chord labels for the current sample
            chord_labels_padded = np.pad(chord_labels[index], (0, max_length - len(chord_labels[index])), mode='constant', constant_values=padding_value)
            chord_labels_batch = torch.tensor(chord_labels_padded, dtype=torch.long)
            
            batch_data.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'additional_features': additional_features_batch,
                'chord_labels': chord_labels_batch,
                'valence': torch.tensor(row['valence_predicted'], dtype=torch.float),
                'arousal': torch.tensor(row['arousal_predicted'], dtype=torch.float)
            })

        yield batch_data

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        input_ids = torch.tensor(row['input_ids'], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        additional_features = torch.tensor(self.additional_features[index], dtype=torch.float)
        chord_labels = torch.tensor(self.chord_labels[index], dtype=torch.float)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'additional_features': additional_features,
            'chord_labels': chord_labels,
            'valence': torch.tensor(row['valence_predicted'], dtype=torch.float),
            'arousal': torch.tensor(row['arousal_predicted'], dtype=torch.float)
        }

# Define the model architecture
class EmotionRegressionModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, additional_features_dim, num_chord_labels):
        super(EmotionRegressionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', gradient_checkpointing=True)
        self.chord_embedding = nn.Embedding(num_chord_labels + 1, 64, padding_idx=num_chord_labels)
        self.fc1 = nn.Linear(self.bert.config.hidden_size + additional_features_dim + 64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask, additional_features, chord_labels):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden = bert_outputs.last_hidden_state[:, 0, :]
        chord_embeddings = self.chord_embedding(chord_labels)
        chord_embeddings_mean = chord_embeddings.mean(dim=1)
        concat_features = torch.cat((last_hidden, additional_features, chord_embeddings_mean), dim=1)
        hidden = self.fc1(concat_features)
        output = self.fc2(hidden)
        return output

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Instantiate the model
hidden_dim = 256
output_dim = 2
additional_features_dim = train_additional_features.shape[1]
num_chord_labels = len(unique_labels)
model = EmotionRegressionModel(hidden_dim, output_dim, additional_features_dim, num_chord_labels)
print("Model instantiation completed")

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 20
batch_size = 32
model.to(device)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()

    train_generator = data_generator(train_df, train_additional_features, train_chord_labels_numeric, batch_size, max_length, padding_value)
    for batch in train_generator:
        input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
        attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)
        additional_features = torch.stack([item['additional_features'] for item in batch]).to(device)
        chord_labels = torch.stack([item['chord_labels'] for item in batch]).to(device)
        valence = torch.stack([item['valence'] for item in batch]).to(device)
        arousal = torch.stack([item['arousal'] for item in batch]).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, additional_features, chord_labels)
        loss = criterion(outputs, torch.stack([valence, arousal], dim=1))
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
valence_true = []
arousal_true = []
valence_pred = []
arousal_pred = []

test_generator = data_generator(test_df, test_additional_features, test_chord_labels_numeric, batch_size, max_length, padding_value)
with torch.no_grad():
    for batch in test_generator:
        input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
        attention_mask = torch.stack([item['attention_mask'] for item in batch]).to(device)
        additional_features = torch.stack([item['additional_features'] for item in batch]).to(device)
        chord_labels = torch.stack([item['chord_labels'] for item in batch]).to(device)
        valence = torch.stack([item['valence'] for item in batch]).to(device)
        arousal = torch.stack([item['arousal'] for item in batch]).to(device)

        outputs = model(input_ids, attention_mask, additional_features, chord_labels)
        valence_pred.extend(outputs[:, 0].tolist())
        arousal_pred.extend(outputs[:, 1].tolist())
        valence_true.extend(valence.tolist())
        arousal_true.extend(arousal.tolist())

valence_true = np.array(valence_true)
arousal_true = np.array(arousal_true)
valence_pred = np.array(valence_pred)
arousal_pred = np.array(arousal_pred)

valence_mae = mean_absolute_error(valence_true, valence_pred)
valence_mse = mean_squared_error(valence_true, valence_pred)
valence_r2 = r2_score(valence_true, valence_pred)

arousal_mae = mean_absolute_error(arousal_true, arousal_pred)
arousal_mse = mean_squared_error(arousal_true, arousal_pred)
arousal_r2 = r2_score(arousal_true, arousal_pred)

print("Valence Evaluation:")
print("MAE:", valence_mae)
print("MSE:", valence_mse)
print("R2 Score:", valence_r2)

print("\nArousal Evaluation:")
print("MAE:", arousal_mae)
print("MSE:", arousal_mse)
print("R2 Score:", arousal_r2)