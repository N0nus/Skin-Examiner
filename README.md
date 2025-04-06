# Skin Lesion Classification Using AI

This project leverages deep learning to detect and classify skin lesions using a trained EfficientNet-based model. The tool is designed to act as an early detection system to flag potential skin lesions, providing a helpful warning signal for users. This project is not meant to replace medical professionals but to assist in the early identification of concerning lesions.

## Project Overview

This web-based application is built using Flask and utilizes a PyTorch model to classify skin lesions. Users can upload an image of a skin lesion along with metadata such as age, sex, and localization. The application will then predict the lesion type with a confidence score. The goal is to raise a flag for possible skin conditions that may require further professional attention.

## Features

- **Skin Lesion Classification**: Upload an image of a skin lesion and get a prediction of its type (e.g., melanoma, basal cell carcinoma, etc.).
- **Confidence Score**: The system provides a confidence percentage to indicate the certainty of the model's prediction.
- **Metadata Input**: Users can input additional metadata (age, sex, and lesion localization) to improve prediction accuracy.
- **Dark Mode**: The application supports a modern dark mode UI for a better user experience.

## Requirements

- Python 3.7+
- Flask
- PyTorch
- EfficientNet-PyTorch
- PIL (Python Imaging Library)
- scikit-learn
- tqdm
- pandas
- numpy

## Setup

1. Clone the repository

2. Install the required Python packages

3. Either train or install skin_lesion_model_b2.pth in root directory (link to .pth file: https://drive.google.com/file/d/1GlgYeF787UHSKL9NSI1La33E4wy4FHSp/view?usp=sharing)

## Usage

1. Run the app.py file

2. Open a browser and go to http://localhost:5000 to use the app

## Model Training

To train the model run the train.py script. Requires the HAM10000 dataset and metadata csv file.

## Dataset Used

HAM10000: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data

