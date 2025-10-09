# Loan Default Prediction Project

## Overview

This project provides an end-to-end demonstration of a binary classification workflow for predicting loan defaults. The entire process, from data generation to model evaluation and visualization, is contained within a single Python script (`Main.py`).

The script simulates a dataset of loan applicants with various features, preprocesses the data, trains two popular classification models (Logistic Regression and Random Forest), and then evaluates their performance using standard metrics and visualizations. This repository is an excellent example for understanding and implementing a complete machine learning pipeline.

## Features

- **Synthetic Data Generation**: Creates a realistic, albeit artificial, dataset for loan applications using `numpy` and `pandas`.
- **Data Preprocessing**: Implements standard preprocessing techniques, including one-hot encoding for categorical variables and feature scaling for numerical variables using `StandardScaler`.
- **Model Training**: Trains and compares two distinct classification models:
    - **Logistic Regression**: A linear model known for its interpretability.
    - **Random Forest**: An ensemble model known for its high accuracy.
- **Comprehensive Evaluation**: Measures model performance using multiple metrics:
    - ROC AUC Score
    - Classification Report (Precision, Recall, F1-Score)
    - Confusion Matrix
- **Rich Visualization**: Generates several plots with `matplotlib` to interpret the results:
    - **ROC Curve**: To compare the trade-off between true positive and false positive rates for both models.
    - **Histogram of Predicted Probabilities**: To show the distribution of the model's confidence scores.
    - **Pie Chart**: To visualize the class distribution in the test set.
    - **Scatter Plot**: To explore the relationship between features, colored by the predicted probability of default.

## How to Run

### 1. Prerequisites

Make sure you have Python 3 installed. You will need the following libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

### 2. Installation

numpy
pandas
scikit-learn
matplotlib



**Install the packages using pip:**

```bash
pip install -r requirements.txt
3. Execution
To run the entire pipeline, simply execute the main script from your terminal:

Bash

python Main.py
The script will print the model evaluation metrics to the console and display the generated plots.

Project Workflow
Data Simulation: The script begins by generating a dataset of 2000 samples. Features like age, income, credit_score, and loan_amount are created. The target variable, default, is synthetically generated based on a logistic function, creating a clear (but not perfect) relationship between the features and the outcome.

Data Preprocessing:

Categorical features (marital_status, purpose) are converted to numerical format using one-hot encoding (pd.get_dummies).

The dataset is split into training (80%) and testing (20%) sets.

Numerical features are scaled using StandardScaler to ensure that all features contribute equally to the model's training.

Model Training & Evaluation:

A Logistic Regression model and a Random Forest Classifier are trained on the preprocessed training data.

Each model is then used to make predictions on the unseen test data.

The performance of each model is printed to the console, showing the AUC score, a detailed classification report, and a confusion matrix.

Visualization:

The script automatically generates and displays four plots to help visualize the model's performance and the data's characteristics, providing deeper insights than metrics alone.

Example Output
When you run the script, you will see console output similar to this:

=== Logistic Regression ===
AUC: 0.99...
Classification Report:
              precision    recall  f1-score   support
           0       0.99      0.97      0.98       253
           1       0.95      0.98      0.96       147
...
Confusion Matrix:
[[245   8]
 [  3 144]]

=== Random Forest ===
AUC: 0.99...
Classification Report:
              precision    recall  f1-score   support
           0       0.98      0.97      0.98       253
           1       0.95      0.97      0.96       147
...
Confusion Matrix:
[[246   7]
 [  5 142]]
