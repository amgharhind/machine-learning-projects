# Avito Cars Price Prediction — Machine Learning

## Overview

This project builds a machine learning pipeline to predict the price of used cars listed on Avito Morocco, based on a scraped dataset (`Cars_Datasets.csv`). The notebook covers the complete data science workflow: exploratory data analysis, feature engineering, missing value imputation, outlier detection and handling, categorical encoding, dimensionality reduction, and regression modeling with multiple algorithms including Decision Trees, Random Forests, and Gradient Boosting.

## Motivation and Importance

The used car market in Morocco is highly active, with platforms like Avito listing thousands of vehicles across all price ranges. Accurately predicting a car's price based on its characteristics can:

- **Help Buyers and Sellers:** Provide a data-driven reference for fair pricing, reducing information asymmetry in the market.
- **Identify Key Price Drivers:** Uncover which features — brand, mileage, fuel type, transmission, or year — most strongly influence a car's value.
- **Automate Valuation:** Enable automated price estimation tools for listing platforms or dealerships.
- **Reveal Market Insights:** Understand regional preferences and trends in the Moroccan used car market (e.g., fuel type popularity, dominant brands).

## Dataset

The dataset (`Cars_Datasets.csv`, semicolon-separated) contains used car listings scraped from Avito Morocco with the following features:

**Car Identity:**
- `Marque` — Car brand (e.g., Dacia, Renault, Toyota, Land Rover)
- `Modèle` — Car model
- `Année-Modèle` — Year of the model

**Technical Specifications:**
- `Kilométrage` — Mileage range (formatted as "min - max", split into `min_kilometrage` and `max_kilometrage`)
- `Puissance fiscale` — Fiscal horsepower (numeric)
- `Type de carburant` — Fuel type (Diesel, Essence, Hybride, Electrique)
- `Boite de vitesses` — Gearbox type (Manuelle, Automatique)

**Listing Attributes:**
- `Première main` — First owner? (Oui/Non)
- `État` — Condition (Endommagé, Pour Pièces, Correct, Bon, Très bon, Excellent)

**Target Variable:**
- `Prix` — Listed price in Moroccan Dirhams (MAD)

## Methodology

### 1. Exploratory Data Analysis (EDA)

The notebook performs an extensive EDA to understand the data and uncover relationships between features and the target price:

**Target Analysis:**
- Distribution of `Prix` reveals a heavily right-skewed distribution with significant outliers at the high end.

**Categorical Features vs. Price:**
- `Type de carburant`: Diesel is the most common fuel type on Avito, far outnumbering others. Electric cars have the highest average price, while Diesel and Essence show the most price outliers.
- `Marque`: Land Rover is by far the most expensive brand on average. Dacia, Volkswagen, and Renault are the most frequently listed brands. Suzuki and Mini are rarely listed, suggesting low market preference in Morocco.
- `Boite de vitesses`: Automatic transmission cars are significantly more expensive on average than manual ones, consistent with global automotive trends.
- `Première main`: First-owner cars command higher prices on average across almost all brands (with a noted exception for Suzuki, attributed to dataset sparsity).
- `État`: Car condition follows an expected price gradient, with better-condition cars priced higher.
- `Année-Modèle`: 2022 models are the most expensive on average but represent only ~0.08% of listings. 2017 is the most common model year.

**Numerical Features vs. Price:**
- `max_kilometrage` and `min_kilometrage`: A linear negative relationship exists between mileage and price — higher mileage cars tend to be cheaper.
- `Année-Modèle`: Newer models correlate with higher prices.
- A **3D scatter plot** visualizes the joint relationship between `max_kilometrage`, `Année-Modèle`, and `Prix`.

**Correlation Heatmap:**
- Computed among `Prix`, `min_kilometrage`, `max_kilometrage`, and `Puissance fiscale`. No strong linear correlation is found between any single numerical feature and price, confirming the need for a non-linear model.

**Variable vs. Variable Analysis:**
- Cross-analysis of `default` status, `Marque`, `Boite de vitesses`, and `Type de carburant` using grouped bar plots and pairplots to explore interaction effects.

### 2. Feature Engineering

- **Mileage Parsing:** The `Kilométrage` column (formatted as a range string like "50 000 - 100 000 km") is split into two separate numeric columns: `min_kilometrage` and `max_kilometrage`. Non-digit characters are stripped and values are converted to integers.
- **Year Type Conversion:** `Année-Modèle` is cast to `object` type for categorical encoding, preserving its discrete, non-continuous nature.

### 3. Missing Value Imputation

Missing values are handled using `SimpleImputer` with a `most_frequent` strategy for the two affected columns:
- `Première main` — imputed with the most frequent category.
- `État` — imputed with the most frequent condition category.

### 4. Normalization (Explored but Discarded)

`StandardScaler` was tested on numerical features (`Puissance fiscale`, `Prix`, `min_kilometrage`, `max_kilometrage`). This approach was **deliberately abandoned** due to the heavy presence of outliers, which would distort scaled values and degrade model performance. The raw numerical values are retained for modeling.

### 5. Feature Encoding

A multi-strategy encoding approach is applied to all categorical variables:

- **Ordinal Encoding (`OrdinalEncoder`):**
  - `État` — Mapped with a custom domain-specific ordering: Endommagé (0) → Pour Pièces (1) → Correct (2) → Bon (3) → Très bon (4) → Excellent (5), respecting the natural quality hierarchy.
  - `Année-Modèle` — Encoded ordinally to reflect chronological ordering.

- **One-Hot Encoding (`OneHotEncoder` with `drop='first'`):**
  - `Type de carburant` — Nominal fuel type categories encoded as binary columns.
  - `Boite de vitesses` — Gearbox type (invalid `'--'` entries replaced with `'Other'` before encoding).
  - `Première main` — Binary ownership flag.
  - `Marque` — Brand encoded with full one-hot expansion.
  - `Modèle` — Car model encoded with full one-hot expansion.

### 6. Modeling — Regression Algorithms

The notebook trains and evaluates multiple regression models, progressively refining the approach:

**Baseline Models:**
- **Decision Tree Regressor** — Trained as a baseline; noted to suffer from overfitting/underfitting.
- **Random Forest Regressor** — Ensemble model that improves over the Decision Tree. R² of ~0.38 is achieved but indicates moderate explanatory power, attributed to the large number of outliers in `Prix`.

**Main Model — Gradient Boosting Regressor:**
- Identified as the best-performing model.
- **Hyperparameter Tuning with `GridSearchCV`:** Systematic search over `n_estimators` [50, 100, 150], `learning_rate` [0.01, 0.1, 0.2], and `max_depth` [3, 5, 7]. Best configuration: `learning_rate=0.2`, `max_depth=5`, `n_estimators=150`.
- **Learning Curves** plotted to assess convergence and detect overfitting.

**Additional Experiments:**
- **AdaBoost Regressor** — Tested as an alternative ensemble method.
- **Robust Scaler** — Applied before modeling as an alternative to StandardScaler, designed to be more resistant to outliers.
- **PCA (Principal Component Analysis)** — Explored for dimensionality reduction, with explained variance plots guiding component selection. Also tested inside a `Pipeline` combined with `SelectKBest` for feature selection.

**Outlier Removal on Target:**
- After initial experiments, price outliers are removed from `Prix` using the IQR method. The filtered dataset is used to retrain the Gradient Boosting model, yielding improved and more stable results.

### 7. Evaluation Metrics

All models are evaluated using:
- **MAE (Mean Absolute Error)** — Average absolute difference between predicted and actual prices.
- **MSE (Mean Squared Error)** — Penalizes large errors more heavily.
- **R² Score** — Proportion of price variance explained by the model.

## Technology Stack

- **Core Libraries:** NumPy, Pandas, Matplotlib, Seaborn
- **Interactive Visualization:** Plotly
- **Machine Learning:** Scikit-learn (`DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `AdaBoostRegressor`, `LinearRegression`, `Ridge`, `Lasso`, `SVR`, `GridSearchCV`, `PCA`, `SelectKBest`, `Pipeline`, `SimpleImputer`, `OrdinalEncoder`, `OneHotEncoder`, `StandardScaler`, `RobustScaler`)
- **Development Environment:** Jupyter Notebook

## Getting Started

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn plotly
```

Then open the notebook in Jupyter and run all cells sequentially. Make sure `Cars_Datasets.csv` is placed in the same directory as the notebook.

## Limitations and Future Work

- **Outlier Impact:** The dataset contains extreme price values that significantly affect model training. A more systematic outlier removal strategy applied to all numerical columns (not just `Prix`) could improve performance further.
- **Mileage Representation:** Using the midpoint of the mileage range instead of separate min/max columns could provide a cleaner single feature for the model.
- **High Cardinality Encoding:** One-hot encoding `Modèle` produces a very high number of binary columns. Techniques like target encoding or embeddings could reduce dimensionality while preserving the signal.
- **Scraping Recency:** The dataset reflects a specific snapshot of Avito listings. Regular data refreshes would be necessary for a production-level pricing tool.
- **Deep Learning:** A neural network approach could potentially capture complex non-linear interactions between features more effectively than gradient boosting.

