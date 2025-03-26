# Credit Card Fraud Detection Using CNN

## Overview
This project focuses on detecting credit card fraud using Convolutional Neural Networks (CNNs). The dataset used is a publicly available credit card transaction dataset. The primary goal is to build and compare multiple CNN models using different optimizers (Adam, SGD, and Adagrad) to evaluate their performance in classifying fraudulent transactions.

## Dataset
The dataset used is stored in `creditcard.csv`, which contains credit card transactions. It includes:
- Features representing transaction details (anonymized)
- `Amount`: The transaction amount
- `Class`: The target variable, where 0 represents legitimate transactions and 1 represents fraudulent transactions

## Preprocessing Steps
1. Load the dataset using Pandas.
2. Standardize the `Amount` column using `StandardScaler`.
3. Remove the `Amount` column after scaling.
4. Perform undersampling to handle class imbalance using `RandomUnderSampler`.
5. Split the dataset into training and testing sets.
6. Reshape the data for CNN input.

## Model Architecture
Each CNN model follows this structure:
- **Conv1D Layers**: Extract features from the input data.
- **Batch Normalization**: Normalize activations to stabilize learning.
- **Dropout**: Reduce overfitting by randomly dropping connections.
- **Flatten Layer**: Converts feature maps into a 1D vector.
- **Dense Layers**: Fully connected layers with ReLU activation.
- **Output Layer**: A single neuron with sigmoid activation for binary classification.

## Training
Three models are trained using different optimizers:
- `model`: Adam optimizer
- `model2`: SGD optimizer
- `model3`: Adagrad optimizer

Each model is trained for 20 epochs using binary cross-entropy loss and accuracy as a metric.

## Evaluation
- Training and validation accuracy/loss are visualized using `plotLearningCurve` function.
- Performance comparison is done based on accuracy and loss curves.

## Requirements
To run this project, install the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn imbalanced-learn
```

## Running the Project
Execute the script in a Python environment:
```bash
python fraud_detection.py
```
Ensure the dataset `creditcard.csv` is placed in the correct path as referenced in the script.

## Results
The project provides a comparative analysis of different optimization algorithms in CNN-based fraud detection. The best-performing model can be selected based on the plotted accuracy and loss curves.
