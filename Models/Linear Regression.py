import pandas as pd
print("Pandas imported successfully.")

import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
print("Other libraries imported successfully.")

def process_batch(batch):
    print(f"Processing batch of size {len(batch)}...")
    
    # Function to parse the 'chords_metadata' column and extract features
    def extract_chords_features(chords_metadata):
        try:
            num_chords = len(chords_metadata['chordSequence'])
            avg_chord_duration = chords_metadata['duration'] / num_chords
            return pd.Series([num_chords, avg_chord_duration])
        except (KeyError, TypeError):
            return pd.Series([0, 0])

    # Extract features from the 'chords_metadata' column
    chords_features = batch['chords_metadata'].apply(extract_chords_features)
    chords_features.columns = ['num_chords', 'avg_chord_duration']

    # Concatenate the extracted features with the batch
    batch = pd.concat([batch, chords_features], axis=1)

    # Convert numeric columns to appropriate data types
    numeric_columns = ['bpm', 'gain', 'length', 'arousal_predicted', 'valence_predicted']
    batch[numeric_columns] = batch[numeric_columns].apply(pd.to_numeric, errors='coerce')

    print("Batch processed successfully.")
    return batch

# Initialize an empty list to store the processed batches
processed_batches = []

# Process the dataset in batches
batch_size = 1000
with open('song_extracted_all_entries.json', 'r') as file:
    print("Reading dataset file...")
    batch = []
    for line in file:
        song = json.loads(line)
        batch.append(song)
        if len(batch) == batch_size:
            print(f"Processing batch {len(processed_batches) + 1}...")
            batch_df = pd.DataFrame(batch)
            processed_batch = process_batch(batch_df)
            processed_batches.append(processed_batch)
            batch = []
            print(f"Batch {len(processed_batches)} processed successfully.")

    # Process the remaining songs if any
    if batch:
        print("Processing remaining songs...")
        batch_df = pd.DataFrame(batch)
        processed_batch = process_batch(batch_df)
        processed_batches.append(processed_batch)
        print("Remaining songs processed successfully.")

print("All batches processed successfully.")

# Concatenate the processed batches into a single DataFrame
print("Concatenating processed batches...")
songs_df = pd.concat(processed_batches, ignore_index=True)
print("Concatenation completed.")

# Select the relevant features and target variables from the songs dataset
features = ['bpm', 'gain', 'length', 'num_chords', 'avg_chord_duration']
target_valence = 'valence_predicted'
target_arousal = 'arousal_predicted'

# Extract the features and target variables
print("Extracting features and target variables...")
X = songs_df[features]
y_valence = songs_df[target_valence]
y_arousal = songs_df[target_arousal]
print("Features and target variables extracted successfully.")

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train_valence, y_test_valence, y_train_arousal, y_test_arousal, id_train, id_test = train_test_split(
    X, y_valence, y_arousal, songs_df['_id'], test_size=0.2, random_state=42)
print("Data split completed.")

# Drop samples with missing values in the training set
print("Dropping samples with missing values in the training set...")
X_train_cleaned = X_train.dropna()
y_train_valence_cleaned = y_train_valence[X_train.notna().all(axis=1)]
y_train_arousal_cleaned = y_train_arousal[X_train.notna().all(axis=1)]
print("Samples with missing values dropped from the training set.")

# Create and train the linear regression models
print("Creating and training linear regression models...")
model_valence = LinearRegression()
model_arousal = LinearRegression()

print("Training valence model...")
model_valence.fit(X_train_cleaned, y_train_valence_cleaned)
print("Valence model training completed.")

print("Training arousal model...")
model_arousal.fit(X_train_cleaned, y_train_arousal_cleaned)
print("Arousal model training completed.")

# Drop samples with missing values in the test set
print("Dropping samples with missing values in the test set...")
X_test_cleaned = X_test.dropna()
y_test_valence_cleaned = y_test_valence[X_test.notna().all(axis=1)]
y_test_arousal_cleaned = y_test_arousal[X_test.notna().all(axis=1)]
id_test_cleaned = id_test[X_test.notna().all(axis=1)]
print("Samples with missing values dropped from the test set.")

# Make predictions on the cleaned test set
print("Making predictions on the cleaned test set...")
valence_pred = model_valence.predict(X_test_cleaned)
arousal_pred = model_arousal.predict(X_test_cleaned)
print("Predictions completed.")

# Compute evaluation metrics for valence (predicted)
print("Computing evaluation metrics for valence (predicted)...")
valence_pred_mae = mean_absolute_error(y_test_valence_cleaned, valence_pred)
valence_pred_mse = mean_squared_error(y_test_valence_cleaned, valence_pred)
valence_pred_r2 = r2_score(y_test_valence_cleaned, valence_pred)
print("Valence evaluation metrics computed.")

# Compute evaluation metrics for arousal (predicted)
print("Computing evaluation metrics for arousal (predicted)...")
arousal_pred_mae = mean_absolute_error(y_test_arousal_cleaned, arousal_pred)
arousal_pred_mse = mean_squared_error(y_test_arousal_cleaned, arousal_pred)
arousal_pred_r2 = r2_score(y_test_arousal_cleaned, arousal_pred)
print("Arousal evaluation metrics computed.")

# Create a DataFrame to store the original and predicted values, along with the '_id'
print("Creating results DataFrame...")
results = pd.DataFrame({
    '_id': id_test_cleaned,
    'Valence_Predicted': y_test_valence_cleaned,
    'Valence_Pred_Model': valence_pred,
    'Arousal_Predicted': y_test_arousal_cleaned,
    'Arousal_Pred_Model': arousal_pred
})
print("Results DataFrame created.")

# Print the results
print("\nValence Metrics (Predicted):")
print("MAE:", valence_pred_mae)
print("MSE:", valence_pred_mse)
print("R-squared:", valence_pred_r2)

print("\nArousal Metrics (Predicted):")
print("MAE:", arousal_pred_mae)
print("MSE:", arousal_pred_mse)
print("R-squared:", arousal_pred_r2)

# Save the results to a JSON file
print("Saving results to JSON file...")
results.to_json('results_linear_regression.json', orient='records')
print("Results saved successfully.")

# Create scatter plots
print("Creating scatter plots...")
plt.figure(figsize=(8, 4))

# Scatter plot for valence (predicted)
plt.subplot(1, 2, 1)
plt.scatter(results['Valence_Predicted'], results['Valence_Pred_Model'], alpha=0.5)
plt.xlabel('Predicted Valence (Gradient Boosting)')
plt.ylabel('Predicted Valence (Model)')
plt.title('Valence (Predicted)')

# Scatter plot for arousal (predicted)
plt.subplot(1, 2, 2)
plt.scatter(results['Arousal_Predicted'], results['Arousal_Pred_Model'], alpha=0.5)
plt.xlabel('Predicted Arousal (Gradient Boosting)')
plt.ylabel('Predicted Arousal (Model)')
plt.title('Arousal (Predicted)')

plt.tight_layout()
plt.show()

print("Script execution completed.")