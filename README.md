# Speech Sentiment Analysis

This repository contains a speech sentiment analysis project that uses a deep learning model to predict emotions from audio input.

## Project Structure

- `app.py`: Streamlit web application for audio upload and emotion prediction
- `model.py`: Neural network model definition and prediction function
- `test.py`: Script for recording audio samples
- `/models`: Directory containing trained model files

## Requirements
- streamlit
- torch
- numpy
- librosa
- joblib
- pyaudio

## Usage

1. Run the Streamlit web application:
`streamlit run app.py`

Copy code

2. To record a test audio sample:
`python test.py`


## Model

The emotion recognition model uses a parallel CNN-LSTM architecture to process mel spectrograms of audio inputs. It can predict one of eight emotions: neutral, calm, happy, sad, angry, fearful, disgust, or surprised.





### Preview
![Screenshot 2024-08-07 224552](https://github.com/user-attachments/assets/aed1af2c-cff0-4cc7-bb5a-b6f7aee9d4aa)
