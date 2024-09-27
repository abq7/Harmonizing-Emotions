import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from transformers import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import BertTokenizer, BertModel

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
        required_columns = ['_id', 'lyrics', 'arousal_predicted', 'valence_predicted']
        missing_columns = [col for col in required_columns if col not in chunk.columns]

        if missing_columns:
            raise ValueError(f"The following required columns are missing in the DataFrame: {missing_columns}")

        # Fill missing values with appropriate placeholders or default values
        chunk['lyrics'] = chunk['lyrics'].fillna('')
        chunk['arousal_predicted'] = chunk['arousal_predicted'].fillna(0)
        chunk['valence_predicted'] = chunk['valence_predicted'].fillna(0)
        
        # Convert 'arousal_predicted' and 'valence_predicted' to float
        chunk['arousal_predicted'] = chunk['arousal_predicted'].astype(float)
        chunk['valence_predicted'] = chunk['valence_predicted'].astype(float)
        
        # Select only the relevant columns
        chunk = chunk[['_id', 'lyrics', 'arousal_predicted', 'valence_predicted']]
        
        # Filter out entries without lyrics
        chunk = chunk[chunk['lyrics'] != '']
        
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
        if total_entries >= 10000:
            break

print("Data loading completed.")

# Combine the train and test data from all batches
train_df = pd.concat(train_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)

print("Lyrics tokenization and data split completed")

# Filter out samples with missing valence or arousal values
train_df = train_df.dropna(subset=['valence_predicted', 'arousal_predicted'])
test_df = test_df.dropna(subset=['valence_predicted', 'arousal_predicted'])

# Convert 'valence_predicted' and 'arousal_predicted' columns to numeric
train_df['valence_predicted'] = pd.to_numeric(train_df['valence_predicted'], errors='coerce')
train_df['arousal_predicted'] = pd.to_numeric(train_df['arousal_predicted'], errors='coerce')
test_df['valence_predicted'] = pd.to_numeric(test_df['valence_predicted'], errors='coerce')
test_df['arousal_predicted'] = pd.to_numeric(test_df['arousal_predicted'], errors='coerce')

# Create a PyTorch Dataset
class EmotionDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        input_ids = torch.tensor(row['input_ids'], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'valence': torch.tensor(row['valence_predicted'], dtype=torch.float),
            'arousal': torch.tensor(row['arousal_predicted'], dtype=torch.float)
        }

# Define the HAN model architecture
class HANModel(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(HANModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', gradient_checkpointing=True)
        self.attention = nn.Linear(self.bert.config.hidden_size, 1)
        self.gru = nn.GRU(self.bert.config.hidden_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        print("Forward pass called")  # Add this line
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_states = bert_outputs.last_hidden_state

        # Word-level attention
        word_attention_weights = torch.softmax(self.attention(hidden_states), dim=1)
        word_level_representation = torch.sum(hidden_states * word_attention_weights, dim=1)

        # Sentence-level encoding
        _, sentence_representation = self.gru(word_level_representation.unsqueeze(0))
        sentence_representation = sentence_representation.squeeze(0)

        # Final prediction
        output = self.fc(sentence_representation)
        return output

# Set device
device = torch.device('cuda')
print(f"Device: {device}")

# Instantiate the HAN model
hidden_dim = 256
output_dim = 2
model = HANModel(hidden_dim, output_dim)
model.to(device)
print("Model instantiation completed")

# Define loss function and optimizer
criterion = nn.MSELoss().to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 10
batch_size = 32
accumulation_steps = 4

train_dataset = EmotionDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        print(f"Processing batch {batch_idx+1}")  # Add this line
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        valence = batch['valence'].to(device)
        arousal = batch['arousal'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, torch.stack([valence, arousal], dim=1))
        loss = loss / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
# Evaluation
model.eval()
valence_true = []
arousal_true = []
valence_pred = []
arousal_pred = []

test_dataset = EmotionDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=1)  # Set batch_size to 1

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        valence = batch['valence'].to(device)
        arousal = batch['arousal'].to(device)

        outputs = model(input_ids, attention_mask)
        valence_pred.append(outputs[0, 0].item())
        arousal_pred.append(outputs[0, 1].item())
        valence_true.append(valence.item())
        arousal_true.append(arousal.item())

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