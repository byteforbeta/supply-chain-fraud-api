import pandas as pd
import numpy as np
import xgboost as xgb
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# 1. Initialize Weights & Biases
wandb.init(
    project="supply-chain-fraud-detection",
    config={
        "test_size": 0.2,
        "random_state": 42,
        "max_depth": 3,
        "learning_rate": 0.01,
        "n_estimators": 200,
    }
)
config = wandb.config

print("Loading and cleaning data")
df = pd.read_csv('data/DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')

# --- 2. THE AUTOMATED CLEANING PIPELINE ---
# Drop Leakage and Noise
drop_cols = ['Days for shipping (real)', 'Delivery Status', 'Late_delivery_risk', 
             'Shipping date (DateOrders)', 'Customer Email', 'Customer Password', 
             'Customer Fname', 'Customer Lname', 'Product Description', 
             'Product Image', 'Latitude', 'Longitude', 'Customer Zipcode', 
             'Order Zipcode', 'Order Item Cardprod Id']
df = df.drop(columns=drop_cols, errors='ignore')

# Target Variable
df['is_fraud'] = np.where(df['Order Status'] == 'SUSPECTED_FRAUD', 1, 0)
df = df.drop(columns=['Order Status'])

# Frequency Encoding for Customer ID
df['customer_order_frequency'] = df['Customer Id'].map(df['Customer Id'].value_counts())
df = df.drop(columns=['Customer Id', 'Order Id', 'Order Customer Id'], errors='ignore')

# Convert remaining strings to Categorical for XGBoost
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].astype('category')



# --- 3. TRAIN/TEST SPLIT ---
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
)

# --- 4. MODEL TRAINING (Handling the Imbalance) ---
imbalance_ratio = (len(y_train) - y_train.sum()) / y_train.sum() # This will be used to penalize the error on less fraud cases in order to force model to learn fraud cases

print("Training XGBoost model")
model = xgb.XGBClassifier(
    enable_categorical=True,
    tree_method='hist',
    scale_pos_weight=imbalance_ratio,
    max_depth=config.max_depth,
    learning_rate=config.learning_rate,
    n_estimators=config.n_estimators,
    eval_metric='aucpr' 
)

model.fit(X_train, y_train)

# --- 5. EVALUATION & W&B LOGGING ---
print("Evaluating model")
preds = model.predict(X_test)

# W&B needs the raw array, not the Pandas Series with shuffled indices
y_test_arr = y_test.to_numpy() 
probas = model.predict_proba(X_test)
probs_positive = probas[:, 1] # Probabilities for the PR-AUC calculation

metrics = {
    "precision": precision_score(y_test_arr, preds),
    "recall": recall_score(y_test_arr, preds),
    "f1_score": f1_score(y_test_arr, preds),
    "pr_auc": average_precision_score(y_test_arr, probs_positive)
}

wandb.log(metrics)

# Log the interactive curves (using the clean y_test_arr)
wandb.log({"PR_Curve": wandb.plot.pr_curve(y_test_arr, probas, labels=["Normal", "Fraud"])})
wandb.log({"ROC_Curve": wandb.plot.roc_curve(y_test_arr, probas, labels=["Normal", "Fraud"])})
wandb.log({"Confusion_Matrix": wandb.plot.confusion_matrix(probs=probas, y_true=y_test_arr, class_names=["Normal", "Fraud"])})

# Log Feature Importance directly to W&B
feature_importances = model.feature_importances_
importance_data = [[name, float(imp)] for name, imp in zip(X.columns, feature_importances)]
table = wandb.Table(data=importance_data, columns=["Feature", "Importance"])
wandb.log({"Feature Importance": wandb.plot.bar(table, "Feature", "Importance", title="XGBoost Feature Importance")})

print(f"Run complete! Metrics: {metrics}")

# Save the model locally
model.save_model("xgboost_fraud_model.json")

# Log model as an artifact in Weights & Biases
artifact = wandb.Artifact("fraud_detection_model", type="model")
artifact.add_file("xgboost_fraud_model.json")
wandb.log_artifact(artifact)
wandb.finish()