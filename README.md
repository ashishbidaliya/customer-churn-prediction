# Customer Churn Prediction

A machine learning project to predict whether a bank customer will churn. I compared 5 different classification algorithms and picked the best one based on accuracy and performance on the churn class.

---

## Problem Statement

Banks lose revenue when customers leave. The goal here is to build a model that can flag customers likely to churn so the business can take action before it's too late. I used a real-world dataset from Kaggle with 10,000 customer records.

---

## Dataset

- **Source:** [Kaggle — Churn Modelling Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling)
- **Records:** 10,000 customers | 14 features
- **Target column:** `Exited` (1 = Churned, 0 = Stayed)

---

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

---

## Feature Engineering

Beyond the raw features, I created a few additional ones that improved model performance:

- `BalanceZero` — customers with zero balance showed higher churn tendency
- `BalanceToSalaryRatio` — balance relative to salary as a financial health indicator
- `ProductUsage` — combined number of products with active membership status
- `AgeGroup` and `TenureGroup` — bucketed age and tenure into ranges
- `Male_Germany`, `Male_Spain` — gender and geography interaction terms

---

## Results

I trained 5 models and compared them. Gradient Boosting came out on top.

| Model                | Accuracy   | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
| -------------------- | ---------- | ----------------- | -------------- | ---------------- |
| Gradient Boosting ✅ | **86.75%** | 0.75              | 0.49           | 0.59             |
| Random Forest        | 86.65%     | 0.76              | 0.46           | 0.58             |
| KNN                  | 83.00%     | 0.61              | 0.37           | 0.46             |
| Logistic Regression  | 81.10%     | 0.55              | 0.20           | 0.29             |
| SVM                  | 80.35%     | 0.00              | 0.00           | 0.00             |

---

## Key Findings

- Age turned out to be the strongest predictor — older customers churn significantly more
- Customers who are not active members are far more likely to leave
- Germany had a noticeably higher churn rate compared to France and Spain
- Having zero balance was a strong churn signal
- Customers with only one product churned more than those with multiple products

---

## How to Run

Clone the repo and set up the environment:

```bash
git clone https://github.com/ashishbidaliya/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

Download `Churn_Modelling.csv` from [Kaggle](https://www.kaggle.com/datasets/shubh0799/churn-modelling) and place it inside the `data/` folder.

To explore the full analysis:

```bash
jupyter notebook notebooks/churn_prediction.ipynb
```

To run just the final model:

```bash
python src/model.py
```

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   └── Churn_Modelling.csv       # not pushed to GitHub, download from Kaggle
├── notebooks/
│   └── churn_prediction.ipynb    # full EDA, feature engineering and model comparison
├── src/
│   └── model.py                  # clean final pipeline using best model
├── README.md
├── requirements.txt
└── .gitignore
```
