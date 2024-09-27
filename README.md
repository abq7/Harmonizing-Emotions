# Harmonizing Emotions: A Multimodal Approach for Emotion Recognition in Music

## Overview

This repository contains the code and models developed for the thesis titled **"Harmonizing Emotions: A Multimodal Approach for Emotion Recognition in Music."** The project aims to predict the emotional attributes of music, specifically **valence** and **arousal**, by analyzing both musical features and lyrics. Various machine learning and deep learning models are used to achieve this, including models leveraging traditional regression techniques and transformer-based neural networks.

### Key Features:
- **Music Feature Extraction**: Analyzes tempo (BPM), gain, song length, and chord sequences.
- **Lyrics-Based Models**: Utilizes BERT-based embeddings and other neural architectures to process song lyrics.
- **Multimodal Emotion Prediction**: Combines musical and lyrical features to predict emotional attributes.
- **Valence and Arousal Prediction**: Outputs emotion dimensions (valence, arousal) that represent the mood of the song.

## How to Use

### Prerequisites:
1. **Dataset**: The project uses a dataset of songs stored in a JSON file format (`song_extracted_all_entries.json`) containing metadata such as lyrics, BPM, gain, song length, predicted valence, predicted arousal, and chord metadata.
2. **Python Environment**: Ensure that you have Python 3.11 or a newer version installed.

### Running the Models:

1. **Install Dependencies**:
   Before running any of the models, ensure that the necessary Python packages are installed by running:
   ```bash
   pip install -r requirements.txt

2. **Running the Models**:
   Each model can be executed individually to view its predictions and evaluation. Simply open the corresponding Python script and run it:

   - `Linear Regression.py`
   - `Random Forest.py`
   - `Gradient Boosting.py`
   - `SVR.py`
   - `CNN + Attention Model.py`
   - `LSTM.py`
   - `BiLSTM + Attention Model.py`
   - `HAN.py`

   The dataset has already been preprocessed, and the models are ready to be run. Each model script includes its training and evaluation.

## Dependencies

The following dependencies are required to run the models in this repository:

- Python 3.11 or a newer version
- `seaborn`
- `numpy`
- `pandas`
- `scikit-learn`
- `torch`
- `transformers`
- `keras`
- `matplotlib`

To install these dependencies, run:

```bash
pip install -r requirements.txt

## Models

The repository contains several models used to predict valence and arousal:

1. **Linear Regression**:
   - `Linear Regression.py`: Implements a simple linear regression model to predict valence and arousal based on musical features.

2. **Random Forest Regression**:
   - `Random Forest.py`: Implements a Random Forest model to predict valence and arousal based on musical features.

3. **Gradient Boosting Regression**:
   - `Gradient Boosting.py`: A gradient boosting model that also utilizes musical metadata for predictions.

4. **SVR (Support Vector Regression) with RBF Kernel**:
   - `SVR.py`: A support vector regression model with a radial basis function kernel for emotion prediction.

5. **CNN + Attention Model**:
   - `CNN + Attention Model.py`: A convolutional neural network with an attention mechanism for emotion prediction based on musical features and lyrics.

6. **LSTM**:
   - `LSTM.py`: A long short-term memory model for sequence modeling of the lyrics for emotion prediction.

7. **BiLSTM + Attention Model**:
   - `BiLSTM + Attention Model.py`: A bidirectional LSTM with an attention mechanism to enhance emotion prediction accuracy.

8. **Hierarchical Attention Network (HAN) Model**:
   - `HAN.py`: A model that applies word-level attention followed by GRU-based sentence-level encoding for emotion prediction from lyrics.

## Dataset

The dataset is too large to be hosted directly in this repository. You can download it from Kaggle using the following link:

[Download the dataset from Kaggle](https://www.kaggle.com/datasets/ricardoalbuquerque77/music-emotion-recognition-dataset)
