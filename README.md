# Loan Default Prediction — Machine Learning Demonstration

An educational binary-classification project that generates a synthetic loan-applicant dataset, trains classification models, and visualizes evaluation results.

> **Important:** The labels are generated from an artificial rule in the script. High model scores show how well a model recovers that simulated rule; they do not demonstrate real-world credit-risk accuracy.

## Current workflow

1. Generate synthetic applicant records and default labels.
2. Prepare the feature data.
3. Split the data into training and test sets.
4. Train Logistic Regression and Random Forest classifiers.
5. Evaluate and visualize model performance.

## Technology

Python, NumPy, Pandas, scikit-learn, Matplotlib, and Seaborn.

## Run locally

1. Clone the repository and create a virtual environment.
2. Install the dependencies imported by `Main.py`.
3. Run:

```bash
python Main.py
```

## Evaluation and interpretation

Use reported metrics only in the context of this generated dataset. Document the synthetic label-generation rule, random seed, class balance, and train/test methodology. Do not present the results as evidence that the model is suitable for lending decisions.

## Recommended improvements

- Move preprocessing and model fitting into scikit-learn Pipelines.
- Add cross-validation, confusion matrix, precision, recall, F1, ROC-AUC, and calibration analysis.
- Separate data generation, training, evaluation, and visualization.
- Add tests and reproducible configuration.
- For any future real dataset, document source, license, and usage conditions. Evaluate fairness and avoid sensitive or proxy attributes without a justified, legally reviewed purpose.

## Author

Piyush Rai — [GitHub](https://github.com/Piyushupc)
