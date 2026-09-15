# 💰 BudgetPlanner

**BudgetPlanner** is a Python-based personal finance project that combines **budget planning, data analysis, and machine learning** to analyze financial patterns and generate savings-related predictions.

## 🚀 Features

* 📊 Financial data preprocessing
* 🔍 Exploratory Data Analysis (EDA)
* 🤖 Machine Learning regression
* 📈 Model evaluation using MAE, RMSE, and R²
* 💾 Model saving with Joblib
* 🔮 Financial prediction
* 💰 Priority-based budget planning

## 🧠 ML Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
EDA
   ↓
Feature Selection
   ↓
ML Training
   ↓
Model Evaluation
   ↓
Best Model
   ↓
Prediction
```

The project compares:

* Linear Regression
* Random Forest Regression

Numerical and categorical features are processed using a Scikit-learn preprocessing pipeline.

## 📊 Dataset

**Indian Personal Finance and Spending Habits**

The dataset contains approximately **20,000 financial records** covering income, expenses, savings, demographics, and spending patterns.

🔗 [Kaggle Dataset](https://www.kaggle.com/datasets/shriyashjagtap/indian-personal-finance-and-spending-habits)

## 📁 Project Structure

```text
BudgetPlanner/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── budget_model.joblib
│
├── reports/
│   └── figures/
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── model_training.py
│   └── prediction.py
│
├── budget_planner.py
├── BudgetPlanner.ipynb
├── requirements.txt
└── README.md
```

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

## ▶️ Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python src/data_preprocessing.py
```

Run EDA:

```bash
python src/eda.py
```

Train the model:

```bash
python src/model_training.py
```

Run prediction:

```bash
python src/prediction.py
```

## 📈 Model Evaluation

Models are evaluated using:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **R²** — Coefficient of Determination

The best-performing model is saved as:

```text
models/budget_model.joblib
```

# 💻 Example Prediction Input

Example financial information:

```text
Income: 60000
Age: 30
Dependents: 2
Occupation: Salaried
City Tier: 1

Rent: 15000
Loan Repayment: 5000
Insurance: 2000
Groceries: 6000
Transport: 3000
Eating Out: 2000
Entertainment: 1500
Utilities: 2500
Healthcare: 1000
Education: 2000
Miscellaneous: 1500
```

The model processes these inputs and generates the selected financial prediction.

> The exact input fields depend on the feature set defined during model training.

# 🔮 Future Improvements

### 1. Advanced Machine Learning Models

* Gradient Boosting
* HistGradientBoosting
* XGBoost
* LightGBM
* Neural Networks

### 2. Hyperparameter Optimization

* GridSearchCV
* RandomizedSearchCV
* Optuna

### 3. Cross Validation

Implement **K-Fold Cross Validation** for more reliable model evaluation.

### 4. Feature Importance

Add:

* Random Forest feature importance
* Permutation importance
* SHAP explainability

This will help identify which financial factors have the greatest influence on predictions.

### 5. Interactive Dashboard

Build a user-friendly dashboard using:

* Streamlit
* Flask
* FastAPI

The dashboard can visualize:

* Income
* Expenses
* Savings
* Spending distribution
* ML predictions
* Budget recommendations

### 6. Personalized Budget Recommendations

Future versions can combine ML predictions with the rule-based budget planner.

```text
Financial Data
      ↓
Expense Analysis
      ↓
ML Prediction
      ↓
Spending Pattern Analysis
      ↓
Budget Recommendation
      ↓
Savings Recommendation
```

# 📌 Project Status

🚧 **Under Development**

### Current Implementation

* ✅ Dataset preprocessing
* ✅ EDA
* ✅ Financial feature analysis
* ✅ ML preprocessing pipeline
* ✅ Regression model training
* ✅ Model comparison
* ✅ Model evaluation
* ✅ Model persistence
* ✅ Prediction pipeline
* ✅ Rule-based budgeting baseline

### Planned Improvements

* 🔄 Improved feature selection
* 🔄 Better target definition
* 🔄 Hyperparameter optimization
* 🔄 Model explainability
* 🔄 Interactive dashboard
* 🔄 Personalized budget recommendations

---

## 👩‍💻 Author

**Harshitha CS**

AI/ML | Computer Vision | Python | Machine Learning

⭐ If you find this project useful, consider giving the repository a star.
