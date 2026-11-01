# Customer Churn Prediction (AI / Machine Learning Project)

**Goal:** Predict whether a telecom customer will leave (churn) so the company can offer retention deals in time.

**Type of problem:** Binary classification (Churn = Yes / No)
**Models compared:** Logistic Regression, Random Forest, Gradient Boosting
**Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC

## Project structure
```
churn_project/
├── generate_data.py    # creates the dataset (data/telco_churn.csv)
├── train.py            # cleaning, EDA, training, evaluation, saves best model
├── predict.py          # predict for a sample customer or a CSV file
├── app.py              # Streamlit web app (optional demo)
├── requirements.txt
├── data/               # dataset (auto-created)
├── models/             # saved model (auto-created)
└── outputs/            # charts + results (auto-created)
```

## How to run (step by step)

1. Install Python 3.9+ (check with `python --version`).
2. Open a terminal inside the `churn_project` folder.
3. (Recommended) create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
4. Install libraries: `pip install -r requirements.txt`
5. Create the dataset: `python generate_data.py`
6. Train and evaluate the models: `python train.py`
   Charts and a comparison table appear in `outputs/`.
7. Try a prediction: `python predict.py`
8. (Optional) Launch the web app: `streamlit run app.py`

## Using the real Kaggle dataset (recommended for submission)
Download "Telco Customer Churn" from Kaggle, rename the file to `telco_churn.csv`,
put it in `data/` (replace the synthetic one), and run `python train.py`. No code changes needed.

## How it works
1. **Data cleaning:** drop customerID, fix TotalCharges, convert Churn to 0/1.
2. **EDA:** churn distribution, churn by contract type, tenure vs churn.
3. **Preprocessing:** scale numeric columns, one-hot encode categorical columns.
4. **Training:** 80/20 stratified split, three models, class weights to handle imbalance.
5. **Evaluation:** metrics, confusion matrix, ROC curves. Best model (by ROC-AUC) is saved.
6. **Insights:** feature importance shows what drives churn (short tenure,
   month-to-month contract, fiber optic, electronic check, high monthly charges).

## Possible improvements
SMOTE for imbalance, hyperparameter tuning (GridSearchCV), XGBoost, SHAP explanations.
