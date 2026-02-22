# Bank Marketing Prediction — Logistic Regression

## Overview

This project builds a machine learning pipeline to predict whether a bank client will subscribe to a term deposit, based on data collected from a Portuguese bank's direct marketing campaigns. Using the `bank-full.csv` dataset, the notebook covers the full data science workflow: exploratory data analysis, feature encoding, outlier handling, and binary classification with Logistic Regression.

## Motivation and Importance

Direct marketing campaigns are costly and resource-intensive. Predicting which clients are likely to subscribe to a term deposit allows banks to:

- **Optimize Campaign Targeting:** Focus outreach efforts on high-probability clients, reducing costs and improving conversion rates.
- **Understand Client Behavior:** Uncover the demographic and behavioral patterns that distinguish subscribers from non-subscribers.
- **Support Data-Driven Strategy:** Replace intuition-based decisions with evidence-based targeting using historical campaign data.

## Dataset

The dataset (`bank-full.csv`, semicolon-separated) contains records of direct marketing contacts with the following features:

**Bank Client Data:**
- `age` — Age of the client (numeric)
- `job` — Type of job (categorical: admin., blue-collar, technician, management, etc.)
- `marital` — Marital status (married, single, divorced)
- `education` — Education level (primary, secondary, tertiary, unknown)
- `default` — Has credit in default? (yes/no)
- `balance` — Average yearly balance in euros (numeric)
- `housing` — Has housing loan? (yes/no)
- `loan` — Has personal loan? (yes/no)

**Last Contact Data:**
- `contact` — Contact communication type (cellular, telephone, unknown)
- `day` — Last contact day of the month
- `month` — Last contact month
- `duration` — Last contact duration in seconds

**Campaign Data:**
- `campaign` — Number of contacts during this campaign
- `pdays` — Days since last contact from a previous campaign (-1 if not previously contacted)
- `previous` — Number of contacts before this campaign
- `poutcome` — Outcome of the previous campaign

**Target Variable:**
- `y` — Has the client subscribed to a term deposit? (yes/no)

## Methodology

### 1. Exploratory Data Analysis (EDA)

The notebook conducts a thorough EDA before any modeling:

- **Target Distribution:** Analyzes the class imbalance between subscribers (`yes`) and non-subscribers (`no`).
- **Categorical Variables vs. Target:** Bar plots and frequency charts reveal which job types, marital statuses, education levels, and contact methods are more associated with subscription. Key findings include that retired and student clients subscribe at higher rates, and clients with tertiary education are more likely to say yes.
- **Numerical Variables vs. Target:** Box plots and histograms compare distributions of age, balance, duration, campaign count, pdays, and previous contacts between the two target classes. Notably, call duration is consistently higher for subscribers.
- **Variable vs. Variable Analysis:** Explores relationships between features, such as how `default` status relates to other variables like `duration` and `campaign`. Pairplots are generated for numerical variables, colored by target and by `default` and `housing` status.
- **Correlation Heatmap:** Computed for all numerical variables to detect multicollinearity.

### 2. Outlier Detection and Handling

During EDA, significant outliers are identified in `balance`, `pdays`, `previous`, `duration`, `campaign`, and `age`. The notebook applies the **IQR (Interquartile Range) method** to detect and remove outliers from the `balance` column as a first step, then re-evaluates model performance on the cleaned dataset. Missing values introduced during outlier removal are filled with the **column median**.

### 3. Feature Encoding

All categorical variables are encoded before modeling using two distinct strategies:

- **One-Hot Encoding (`pd.get_dummies`):** Applied to nominal variables without inherent order — `job`, `marital`, `housing`, `loan`, `poutcome`, `contact`, `default`. The resulting binary columns are cast to integers.
- **Label Encoding (`LabelEncoder`):** Applied to ordinal or cyclical variables:
  - `education`: `unknown` values are replaced with `other` before encoding to preserve the ordinal nature.
  - `month`: Encoded as numeric values representing calendar months.

### 4. Modeling — Logistic Regression

- **Train/Test Split:** 80% training, 20% testing.
- **Model:** `LogisticRegression` from Scikit-learn, trained on the encoded feature matrix.
- **Two Runs:** The model is trained and evaluated both on the original data and on the outlier-cleaned version to compare the impact of outlier removal.
- **Reusable Functions:** Encoding and training steps are refactored into clean, reusable functions (`one_hot_encode_columns`, `encode_education_column`, `encode_month_column`, `train_logistic_regression`) for better code organization.

### 5. Evaluation

Model performance is assessed using:

- **Confusion Matrix** — Visualized as a heatmap to clearly show true/false positives and negatives.
- **Classification Report** — Includes precision, recall, F1-score, and support for both classes, giving a full picture of performance on the imbalanced target.

## Technology Stack

- **Core Libraries:** NumPy, Pandas, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (`LogisticRegression`, `LabelEncoder`, `train_test_split`, `confusion_matrix`, `classification_report`)
- **Development Environment:** Jupyter Notebook

## Getting Started

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Then open the notebook in Jupyter and run all cells sequentially. Make sure `bank-full.csv` is placed in the same directory as the notebook.

## Limitations and Future Work

- **Class Imbalance:** The dataset is heavily imbalanced toward `no` responses. Future work should explore techniques such as SMOTE, class weighting, or threshold tuning to improve recall for the minority class.
- **Outlier Strategy:** Currently only `balance` outliers are removed. A more comprehensive strategy applied to all affected columns (`pdays`, `duration`, `campaign`, etc.) could further improve model quality.
- **Model Diversity:** Logistic Regression serves as a solid baseline. Comparing with more powerful classifiers like Random Forest, Gradient Boosting, or XGBoost would provide a clearer benchmark.
- **Feature Selection:** Given the large number of one-hot encoded columns, feature selection techniques (e.g., RFE or feature importance) could help reduce dimensionality and improve generalization.
- **Hyperparameter Tuning:** Regularization strength (`C`) and solver choice for Logistic Regression have not been optimized. `GridSearchCV` could yield better results.


