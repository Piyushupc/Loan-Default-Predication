import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Dummy dataset
np.random.seed(42)
n = 2000

# Feature generation
age = np.random.randint(21, 70, size=n)
income = np.round(np.random.normal(60000, 25000, size=n)).clip(8000, 30000)
loan_amount = np.round(np.random.normal(15000, 10000, size=n)).clip(1000, 10000)
loan_term_month = np.random.choice([12, 24, 36, 48, 60], size=n, p=[0.05, 0.15, 0.45, 0.07, 0.28])
credit_score = np.round(np.random.normal(658, 70, size=n)).clip(300, 850)
employment_years = np.round(np.random.exponential(scale=3, size=n)).astype(int)
prior_defaults = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
marital_status = np.random.choice(['single', 'married', 'divorced', 'widowed'], size=n, p=[0.45, 0.45, 0.07, 0.03])
purpose = np.random.choice(['debt_consolidation', 'home_improvement', 'education', 'car', 'other'], size=n, p=[0.5, 0.15, 0.1, 0.15, 0.1])

# Probability of default formula
loan_income_ratio = loan_amount / (income + 1)
score = (
    -3.0 * ((credit_score - 650) / 100.0)
    + 6.0 * loan_income_ratio
    - 0.2 * employment_years
    + 2.5 * prior_defaults
    + np.where(purpose == 'debt_consolidation', 0.5, 0.0)
    + np.where(marital_status == 'single', 0.1, 0.0)
)

prob_default = 1 / (1 + np.exp(-score))
default = (np.random.rand(n) < prob_default).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'age': age,
    'income': income,
    'loan_amount': loan_amount,
    'loan_term_month': loan_term_month,
    'credit_score': credit_score,
    'employment_years': employment_years,
    'prior_defaults': prior_defaults,
    'marital_status': marital_status,
    'purpose': purpose,
    'default': default
})

print("Dataset created successfully")
print(df.head())

# Data Preprocessing
X = df.drop(columns=['default'])
y = df['default']

# Categorical encoding
X = pd.get_dummies(X, columns=['marital_status', 'purpose'], drop_first=True)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric columns
num_cols = ['age', 'income', 'loan_amount', 'loan_term_month', 'credit_score', 'employment_years', 'prior_defaults']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train Models
lr = LogisticRegression(max_iter=1000, solver='liblinear')
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Evaluate Models
models = {'Logistic Regression': lr, 'Random Forest': rf}
results = {}

for name, model in models.items():  # <-- changed 'models' to 'model' in loop variable
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    results[name] = {'auc': auc, 'report': report, 'cm': cm}

    print(f"\n=== {name} ===")
    print(f"AUC: {auc}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

# plot ROC curve
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Hisotogram 
plt.figure(figsize=(8, 6))
plt.hist(y_prob, bins=30, alpha=0.7, label='Predicted Probabilities')
plt.xlabel('Predicted Probability of Default')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities')
plt.legend()
plt.show()

# Pie Chart
plt.figure(figsize=(6, 6))
labels = ['No Default', 'Default']
sizes = [np.sum(y_test == 0), np.sum(y_test == 1)]
colors = ['#66b3ff', '#ff9999']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Loan Default in Test Set')
plt.axis('equal')
plt.show()

# Scater plot
plt.figure(figsize=(8, 6))
plt.scatter(X_test['loan_amount'], X_test['income'], c=y_prob, cmap='viridis', alpha=0.7)
plt.colorbar(label='Predicted Probability of Default')
plt.xlabel('Loan Amount')
plt.ylabel('Income')
plt.title('Loan Amount vs Income Colored by Predicted Probability of Default')
plt.show()

