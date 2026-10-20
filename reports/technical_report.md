# Technical Report: Customer Churn Prediction and Business Intelligence Platform

## 1. Summary
This project predicts which telecom customers are likely to churn, explains the predictions,
and turns the findings into testable retention ideas. It covers data validation, SQL analytics,
a leakage-safe machine learning pipeline, model comparison, explainability, a Streamlit dashboard,
and a FastAPI service. The final XGBoost model reaches ROC-AUC 0.844 and PR-AUC 0.667 on a held-out
test set, catching 88.5% of churners at the cost of many false alarms (precision 44.5%).
All results come from the IBM Telco public sample dataset and are not claims about a real business.

## 2. Problem and data
- **Business question:** which customers will churn, why, and where is a retention effort worth trying?
- **Data:** IBM Telco sample dataset, 7,043 customers, 21 columns. Target `Churn` (Yes/No).
  26.5% of customers churned, a moderate imbalance of about 2.8 to 1.
- **Data quality:** no duplicate customers; 11 blank `TotalCharges` values, all for customers with
  `tenure` 0 (new, not yet billed). The pipeline fills them with 0 and keeps a missing-value flag.

## 3. Method
1. **Split first:** stratified 60/20/20 train/validation/test (4,225 / 1,409 / 1,409), seed 42.
2. **Leakage control:** feature engineering uses only single-row operations; imputation, encoding and
   scaling live inside a scikit-learn Pipeline, so they are fitted on training data only (and on
   training folds during cross-validation). The test set is used once.
3. **Models:** a do-nothing baseline plus Logistic Regression, Decision Tree, Random Forest,
   Gradient Boosting and XGBoost. Each was tuned with 3-fold cross-validation on the training set,
   scored by PR-AUC (average precision).
4. **Selection:** models compared on the validation set by PR-AUC, which does not depend on a
   threshold and focuses on the minority (churn) class.
5. **Threshold:** chosen on validation data by maximising F2 (recall weighted twice as much as
   precision), giving 0.34. This encodes the assumption that a missed churner costs more than a
   wasted retention offer.

## 4. Results
| Model (validation) | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| XGBoost | 0.507 | 0.791 | 0.839 | 0.648 |
| Logistic Regression | 0.669 | 0.524 | 0.837 | 0.648 |
| Gradient Boosting | 0.662 | 0.503 | 0.838 | 0.645 |
| Random Forest | 0.660 | 0.436 | 0.839 | 0.643 |
| Decision Tree | 0.499 | 0.746 | 0.816 | 0.598 |
| Baseline (no skill) | 0.000 | 0.000 | 0.500 | 0.265 |

(Precision and recall at a 0.5 threshold so models compare fairly.)

**Final test results (XGBoost, threshold 0.34):** accuracy 0.677, precision 0.445, recall 0.885,
F1 0.593, ROC-AUC 0.844, PR-AUC 0.667. Confusion matrix on 1,409 customers: 331 churners caught,
43 missed, 412 false alarms, 623 correctly left alone.

**Reading the results honestly.**
- The four leading models are within about 0.005 PR-AUC of each other on validation, which is inside
  noise. XGBoost was selected by a fixed rule, but a simpler Logistic Regression is nearly as good.
- Accuracy is misleading: the baseline scores 73.5% accuracy while catching no churners. The final
  model's accuracy is lower on purpose because the threshold trades false alarms for recall.
- At the chosen threshold, about 55% of flagged customers are false alarms. Whether that is
  acceptable depends on the cost of an offer relative to the value of a customer.
- Contacting only flagged customers reaches a churner about every 2.2 contacts, against about
  every 3.8 when contacting everyone. Reaching a churner is not the same as keeping them.

## 5. Explainability
SHAP (global and per-customer) and permutation importance were computed on validation data.
Contract type, tenure, internet service, monthly charges and online security rank highest. The two
methods disagree on order (SHAP ranks Contract first, permutation ranks tenure first) because the
features are correlated, so importance is treated as a rough guide. These are associations the model
learned, not causes of churn.

## 6. Application
- **Dashboard (Streamlit, 6 pages):** overview, EDA, model performance with a threshold explorer,
  single-customer risk prediction with factor chart, batch CSV prediction, business insights.
- **API (FastAPI):** `/health`, `/model-info`, `/predict` (optional SHAP reasons), `/predict-batch`.
  Inputs are validated strictly: unknown categories, out-of-range values, impossible combinations and
  unexpected fields are rejected with clear messages. The model is loaded once at startup.
- **Quality:** automated tests for preprocessing, metrics, validation, API and dashboard pages,
  including a test that production-path predictions equal evaluation-path predictions.

## 7. Business insights
Segment analysis with 95% confidence intervals and five retention ideas, each checked within
subgroups. Example of an inconvenient finding reported rather than hidden: the "first-year customers
churn more" pattern holds for month-to-month contracts (51.4% vs 33.5%) but not for one-year
contracts (10.5% vs 11.3%). Each idea lists what the evidence does not show and an experiment to test
it. A what-if calculator turns assumptions the user supplies into a net-value estimate; it is not a
forecast, because the data cannot measure how well an offer would work.

## 8. Limitations
- Public sample data, a single snapshot: no time-based validation.
- Association, not causation; effects of interventions are unmeasured.
- Probabilities not calibrated; risk bands are a presentation convention.
- The threshold reflects an assumed cost ratio.
- Demographic columns are used by the model and no fairness audit was done.
- API is not production-hardened (no authentication or rate limiting).

## 9. Next steps
Calibration; fairness audit; time-based validation; randomised retention experiments to measure real
effects; monitoring for drift; authentication and deployment if the service leaves local use.