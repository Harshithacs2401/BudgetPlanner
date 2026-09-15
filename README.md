# 💰 BudgetPlanner

**BudgetPlanner** is a Python-based personal finance optimization system that helps users allocate their monthly budget between essential **needs** and discretionary **wants** based on user-defined priorities and available savings.

The project focuses on **algorithmic decision-making, priority-based resource allocation, constraint handling, and financial data processing**, providing a foundation that can be extended into an AI/ML-based personal finance recommendation system.

---

## 🧠 Project Overview

The system takes a user's:

* Monthly budget
* Desired savings
* Essential needs
* Discretionary wants
* Priority scores
* Cost associated with each item

and determines which items can be accommodated within the user's **spendable budget**.

The core decision-making process prioritizes higher-priority items first while respecting the user's savings constraint.

### High-Level Workflow

```text
User Financial Inputs
        │
        ▼
Monthly Budget + Savings Goal
        │
        ▼
Calculate Available Spending Amount
        │
        ▼
Collect Needs & Wants
        │
        ▼
Assign Priority Scores
        │
        ▼
Priority-Based Sorting
        │
        ▼
Budget Constraint Evaluation
        │
        ▼
Select Affordable Items
        │
        ▼
Calculate Total Spending
        │
        ▼
Calculate Remaining Savings
        │
        ▼
Generate Recommendation
```

---

# 🎯 Problem Statement

Personal budgeting often requires deciding which expenses should be prioritized when available funds are limited.

For example, if a user has:

```text
Monthly Budget = ₹50,000
Target Savings = ₹10,000

Available Spending Budget = ₹40,000
```

and multiple expenses compete for the available amount, the system prioritizes expenses according to their assigned importance.

This transforms the budgeting problem into a **constraint-based resource allocation problem**.

---

# ⚙️ Current Approach

The current implementation uses a **priority-based greedy allocation strategy**.

Each requirement is represented using:

```text
Item = Name + Priority + Cost
```

Items are sorted by priority in descending order:

```text
Priority 10
    ↓
Priority 9
    ↓
Priority 8
    ↓
...
Priority 1
```

The system then evaluates whether the cost of each item can be accommodated within the remaining spending budget.

### Decision Rule

```text
If Remaining Budget - Item Cost >= 0

        ↓

Accept Item
        ↓
Update Remaining Budget
```

Otherwise:

```text
Reject / Defer Item
```

This approach provides a simple and interpretable decision-making mechanism.

---

# 🧮 Core Algorithm

The available spending amount is calculated as:

```text
Available Budget = Monthly Budget - Target Savings
```

For every item ordered by priority:

```text
if item_cost <= remaining_budget:

    select item

    remaining_budget =
        remaining_budget - item_cost

else:

    do not select item
```

Finally:

```text
Total Spending =
Sum of selected item costs
```

and:

```text
Final Savings =
Target Savings + Remaining Budget
```

---

# 📊 Example

Consider the following input:

```text
Monthly Budget = ₹500
Target Savings = ₹100
```

Therefore:

```text
Available Spending = ₹400
```

### Needs

| Item   | Priority | Cost |
| ------ | -------: | ---: |
| Water  |       10 |  ₹50 |
| Food   |        9 | ₹150 |
| Drinks |        8 |  ₹50 |

### Wants

| Item  | Priority | Cost |
| ----- | -------: | ---: |
| Party |       10 |  ₹80 |
| Pizza |        8 |  ₹40 |

The system processes items according to priority.

```text
Initial available budget = ₹400

Water      → ₹350 remaining
Food       → ₹200 remaining
Drinks     → ₹150 remaining
Party      → ₹70 remaining
Pizza      → ₹30 remaining
```

Final result:

```text
Total Spending = ₹370
Final Savings   = ₹130
```

---

# 🧠 Why This Project Is Relevant to AI/ML

Although the current implementation does **not use a trained machine learning model**, it demonstrates several concepts that are relevant to intelligent decision systems:

### 1. Decision Making

The system makes automated decisions about which expenses can be fulfilled.

### 2. Priority-Based Optimization

Items are ranked according to importance before allocation.

### 3. Constraint Handling

The system must satisfy:

```text
Spending <= Budget - Target Savings
```

### 4. Structured Data Processing

User-provided financial information is transformed into structured dictionaries and processed algorithmically.

### 5. Recommendation Logic

The system produces a list of items that can be purchased within the available budget.

These concepts provide a natural foundation for introducing machine learning into the project.

---

# 🤖 AI/ML Extension

The current rule-based system can be extended into a data-driven financial intelligence system.

## 1. Expense Prediction

Historical spending data could be used to predict future expenses.

Potential models:

* Linear Regression
* Random Forest
* Gradient Boosting
* XGBoost
* Time-Series models

```text
Historical Transactions
          ↓
Data Preprocessing
          ↓
Feature Engineering
          ↓
ML Model
          ↓
Predicted Expenses
```

---

## 2. Automatic Priority Prediction

Currently, the user manually provides a priority score between 1 and 10.

A future ML model could automatically estimate priority based on historical behavior.

Possible features:

```text
Item Category
Transaction Frequency
Historical Spending
Monthly Income
Previous User Decisions
Seasonality
Essential/Non-essential Indicator
```

Instead of:

```text
User → Priority = 8
```

the system could predict:

```text
ML Model → Priority Score = 8.4
```

---

## 3. Expense Classification

Transaction descriptions can be automatically categorized.

Example:

```text
"Electricity Bill"
        ↓
     Utilities

"Swiggy"
        ↓
       Food

"Uber"
        ↓
 Transportation
```

Possible ML approaches:

* Logistic Regression
* Random Forest
* NLP classification
* Neural Networks

---

## 4. Spending Anomaly Detection

The system could identify unusual spending behavior.

Example:

```text
Average Monthly Shopping
₹4,000

Current Month
₹9,500

        ↓

Potential Spending Anomaly
```

Possible techniques:

* Isolation Forest
* Local Outlier Factor
* Statistical anomaly detection

---

## 5. Personalized Recommendations

The rule-based recommendation engine can eventually be enhanced using user spending history.

For example:

```text
Historical Spending
        ↓
User Behavior Analysis
        ↓
ML Prediction
        ↓
Budget Optimization
        ↓
Personalized Recommendation
```

Possible recommendation:

> Your entertainment spending has increased significantly compared with previous months. Consider reducing discretionary expenses to maintain your savings target.

---

# 🔬 Potential ML Pipeline

A future version can follow this pipeline:

```text
             Financial Data
                   │
                   ▼
          Data Preprocessing
                   │
                   ▼
             Data Validation
                   │
                   ▼
                 EDA
                   │
                   ▼
          Feature Engineering
                   │
                   ▼
        ┌──────────┴──────────┐
        ▼                     ▼
 Expense Prediction      Classification
        │                     │
        └──────────┬──────────┘
                   ▼
           Anomaly Detection
                   │
                   ▼
          Budget Optimization
                   │
                   ▼
       Personalized Recommendation
```

---

# 🛠️ Technology

### Current Implementation

* **Python**
* Python Dictionaries
* Functions
* Sorting algorithms
* Conditional logic
* SMTP email integration
* Jupyter Notebook

### Potential AI/ML Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* XGBoost
* Jupyter Notebook

---

# 📂 Repository Structure

The current repository is lightweight and primarily contains Python source code and a Jupyter Notebook.

```text
BudgetPlanner/
│
├── budget_planner.py
│
├── BudgetPlanner.ipynb
│
├── README.md
│
└── .gitignore
```

The project can later be expanded into:

```text
BudgetPlanner/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   └── model_training.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── prediction.py
│   └── recommendation.py
│
├── models/
│
├── budget_planner.py
├── requirements.txt
└── README.md
```

---

# ▶️ Running the Project

### Clone the repository

```bash
git clone https://github.com/Harshithacs2401/BudgetPlanner.git
```

### Navigate to the project

```bash
cd BudgetPlanner
```

### Run the Python application

```bash
python budget_planner.py
```

The application accepts:

1. Monthly budget
2. Target savings
3. Number of needs
4. Need name, priority and cost
5. Number of wants
6. Want name, priority and cost

It then generates the optimized budget allocation.

---

# 📓 Jupyter Notebook

The project also contains a Jupyter Notebook containing the original implementation and sample execution.

The notebook demonstrates:

* User input processing
* Priority assignment
* Budget calculation
* Needs and wants allocation
* Spending calculation
* Savings calculation
* Email notification

---

# 📈 Current Output

For a sample budget of ₹500:

```text
Monthly Budget       : ₹500
Target Savings       : ₹100
Available Spending   : ₹400

Total Spending       : ₹370
Final Savings        : ₹130
```

The system successfully allocates available funds based on item priority while respecting the target savings amount.

---

# 🔮 Future Scope

The project can evolve from a rule-based budget planner into an **AI-driven personal finance decision-support system**.

### Planned improvements

* 📊 Historical transaction dataset
* 🤖 ML-based expense prediction
* 🧠 Automatic priority prediction
* 🏷️ Automatic expense categorization
* 🚨 Spending anomaly detection
* 📈 Time-series forecasting
* 🎯 Savings goal prediction
* 🧮 ML-assisted budget optimization
* 💡 Personalized financial recommendations
* 🔍 Explainable AI for recommendations

---

# ⚠️ Security Note

The original implementation contains SMTP authentication credentials directly inside the Python source code.

**Credentials should never be stored in source code or committed to GitHub.**

Use environment variables instead:

```python
import os

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
```

and store credentials in a local `.env` file that is excluded through `.gitignore`.

If the credential currently present in the repository is real, **rotate/revoke it before using the repository publicly.**

---

# 📌 Project Status

### Current

**Rule-Based Budget Optimization System**

The current implementation uses priority-based algorithmic decision-making and does not contain a trained ML model.

### Future

**AI/ML-Based Personal Finance Intelligence System**

The planned ML components will introduce predictive analytics, automated categorization, anomaly detection, and personalized recommendations.

---

# 👨‍💻 Author

## Harshitha CS

GitHub:
[Harshithacs2401](https://github.com/Harshithacs2401?utm_source=chatgpt.com)

Project Repository:
[BudgetPlanner](https://github.com/Harshithacs2401/BudgetPlanner?utm_source=chatgpt.com)

---

# ⭐ Project Summary

**BudgetPlanner** is a Python-based budget optimization project that applies priority-based decision-making and budget constraints to help users allocate limited financial resources.

The project provides a foundation for developing an **AI/ML-powered personal finance system** by combining financial data processing, predictive modeling, optimization, anomaly detection, and personalized recommendations.

> **Current focus:** Algorithmic budget optimization
> **Future focus:** AI/ML-driven financial prediction and recommendation
