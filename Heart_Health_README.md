# Heart Health Prediction — Machine Learning

## Overview

This project presents a complete machine learning pipeline for predicting the risk of heart attacks based on patient health data. Using a structured clinical dataset (`Heart_health.csv`), the project walks through data preprocessing, feature engineering, class imbalance handling, and model training and evaluation with Support Vector Machines (SVM). Two notebook versions are provided, each reflecting an increasingly refined approach to feature representation and preprocessing.

## Motivation and Importance

Cardiovascular disease remains one of the leading causes of death worldwide. Early and accurate prediction of heart attack risk can enable timely intervention and significantly improve patient outcomes. This project aims to:

- **Support Clinical Decision-Making:** Provide a data-driven tool to assist healthcare professionals in identifying high-risk patients.
- **Explore Feature Engineering Strategies:** Investigate how different representations of clinical variables (raw vs. categorized) affect model performance.
- **Address Class Imbalance:** Apply resampling techniques to ensure fair model training when heart attack cases are underrepresented.
- **Benchmark SVM Variants:** Compare multiple SVM kernel configurations to identify the best-performing approach.

## Dataset

The dataset (`Heart_health.csv`) contains patient records with the following raw features:

- **Demographics:** Age, Gender
- **Biometrics:** Height (cm), Weight (kg), Blood Pressure (mmHg)
- **Lab Results:** Cholesterol (mg/dL), Glucose (mg/dL)
- **Lifestyle:** Smoker status, Exercise (hours/week)
- **Target:** Heart Attack (binary — Yes/No)

Identifiers (`ID`, `Name`) are dropped before any processing.

## Project Versions

### Version 1 — Baseline Preprocessing & SVM (`Heart_Health__ML_V1_AMGHAR_HIND.ipynb`)

This version establishes the baseline pipeline with straightforward numerical preprocessing:

1. **Data Cleaning & Encoding:**
   - Encodes `Smoker` as binary (Yes → 1, No → 0).
   - Converts `Gender` into a binary `Female` indicator column.
   - Computes **BMI** from height and weight, then drops the original columns.
   - Splits the `Blood Pressure` column into separate `Systolic Pressure` and `Diastolic Pressure` numeric features.

2. **Exploratory Data Analysis:**
   - Correlation heatmaps of numerical features.
   - Descriptive statistics to understand distributions and potential outliers.

3. **Class Imbalance Handling:**
   - Detects the minority class (heart attack positive cases).
   - Applies **manual oversampling** by duplicating minority samples to balance the training set.

4. **Modeling — Support Vector Machine:**
   - Scales features using `StandardScaler`.
   - Trains and evaluates SVM with both **linear** and **polynomial** kernels.
   - Reports accuracy, precision, recall, F1-score, and ROC-AUC.
   - Visualizes the **confusion matrix** and **learning curves** to assess generalization.

---

### Version 2 — Advanced Feature Engineering (`Heart_Health__ML_V2_AMGHAR_HIND.ipynb`)

This version significantly extends V1 by replacing raw numerical features with clinically meaningful **categorical variables**, resulting in a richer and more interpretable feature space:

1. **BMI Categorization (two strategies explored):**
   - *Strategy 1 (7 bins):* Severely Underweight → Extremely Obese. Dropped due to empty categories.
   - *Strategy 2 (4 bins — retained):* Underweight / Healthy Weight / Overweight / Obesity — better distribution across the dataset.

2. **Blood Pressure Categorization:**
   - A custom function classifies each patient's blood pressure reading into clinical categories: `Normal`, `Elevated`, `High Blood Pressure Stage 1`, `High Blood Pressure Stage 2`, `Hypertensive Crisis`.
   - The raw systolic and diastolic columns are then dropped.

3. **Age Categorization:**
   - Multiple binning strategies are explored.
   - Final bins: `30–39`, `40–49`, `50–59`, `60 and above` (the "Under 30" bin was discarded due to zero instances).

4. **Preprocessing Pipeline:**
   - Uses `ColumnTransformer` to apply `StandardScaler` to remaining numerical features (Cholesterol, Glucose, Exercise) and `OneHotEncoder` to all categorical features (BMI category, BP category, Age category).
   - Wrapped in a clean, reusable `Pipeline` structure.

5. **Modeling — Support Vector Machine:**
   - Trains SVM (default and polynomial kernel) on the transformed feature set.
   - Evaluates with the same metrics as V1: accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and learning curves.

---

## Key Differences Between V1 and V2

| Aspect | V1 | V2 |
|---|---|---|
| BMI | Raw numerical value | 4-category ordinal variable |
| Blood Pressure | Raw systolic/diastolic values | Clinical BP category |
| Age | Raw numerical value | Binned age group |
| Preprocessing | Manual `StandardScaler` | `ColumnTransformer` + `Pipeline` |
| Imbalance Handling | Manual oversampling | Explored within pipeline |

## Technology Stack

- **Core Libraries:** NumPy, Pandas, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn (`SVC`, `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `Pipeline`, `train_test_split`, `learning_curve`)
- **Imbalance Handling:** `imbalanced-learn`
- **Development Environment:** Jupyter Notebook

## Project Structure

```
heart-health-ml/
├── README.md                             # This file
├── Heart_Health__ML_V1_AMGHAR_HIND.ipynb # Baseline preprocessing and SVM
├── Heart_Health__ML_V2_AMGHAR_HIND.ipynb # Advanced feature engineering and pipeline
└── Heart_health.csv                      # Source dataset
```

## Getting Started

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

Then open either notebook in Jupyter and run all cells sequentially. Make sure `Heart_health.csv` is placed in the same directory as the notebooks.

## Evaluation Metrics Used

Both versions evaluate model performance using a consistent set of metrics to allow fair comparison: **Accuracy**, **Precision**, **Recall**, **F1-Score**, **ROC-AUC**, **Confusion Matrix**, and **Learning Curves** (to assess overfitting/underfitting behavior).

## Limitations and Future Work

- **Dataset Size:** Performance and generalization depend heavily on the size and representativeness of the dataset. A larger clinical dataset would yield more reliable results.
- **Model Diversity:** Only SVM is explored in these versions. Future iterations could benchmark against Random Forests, Gradient Boosting (XGBoost, LightGBM), or deep learning approaches.
- **Hyperparameter Tuning:** Systematic tuning via `GridSearchCV` or `RandomizedSearchCV` is not yet implemented and could improve results significantly.
- **Feature Selection:** Correlation analysis hints at redundancy between some features. Applying dimensionality reduction (PCA) or feature importance scores could streamline the pipeline.
- **Clinical Validation:** Model outputs should be validated with medical professionals before any real-world deployment.

## Acknowledgement

I would like to express my sincere gratitude to my supervisor, **EL HABIB Ben Lahmar**, for his support and valuable guidance throughout this project.
