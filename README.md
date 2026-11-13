# Loan Default Prediction — ML Workflow Demonstration

A small machine-learning project demonstrating a binary classification workflow for loan-default prediction using a synthetically generated dataset.

> **Data disclaimer:** The dataset is generated in code using an assumed probability formula. Results measure how well models learn that simulated process; they are not evidence of real-world credit-risk performance and must not be used to make lending decisions.

## Features

- Generate a reproducible synthetic applicant dataset.
- Encode categorical features.
- Split data into stratified training and test sets.
- Train Logistic Regression and Random Forest classifiers.
- Compare ROC-AUC, precision, recall, F1, and confusion matrices.
- Plot ROC curves and prediction distributions.

## Technology stack

- Python
- NumPy and Pandas
- scikit-learn
- Matplotlib

## Run locally

Install dependencies:

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

Run the script:

```bash
python Main.py
```

The script generates its dataset at runtime, trains the models, prints evaluation metrics, and opens plots.

## Workflow

1. Generate synthetic applicant features and a binary default label.
2. Separate predictors from the target.
3. One-hot encode categorical variables.
4. Create a stratified train/test split.
5. Fit preprocessing on training data and transform the test data.
6. Train Logistic Regression and Random Forest.
7. Compare classification metrics and visualize results.

## Interpretation cautions

- Synthetic labels are created from a known formula using the generated features, so high scores can reflect recovery of the simulation rule.
- The generated distributions and relationships are assumptions, not representative evidence about a lender or population.
- Accuracy alone is insufficient for imbalanced classification; inspect class-wise precision/recall, calibration, and threshold trade-offs.
- Real credit-risk work requires lawful, representative data, careful leakage checks, fairness analysis, governance, and domain validation.

## Recommended improvements

- Move preprocessing into a Pipeline and ColumnTransformer.
- Add cross-validation and probability calibration.
- Add tests for data generation and training.
- Save model artifacts with explicit versions and metadata.
- Add a requirements file and reproducible random seeds.
- Evaluate on a suitable, documented public dataset before making claims about generalization.

## Author

Piyush Rai — [GitHub](https://github.com/Piyushupc)
