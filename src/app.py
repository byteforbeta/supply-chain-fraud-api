from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import uvicorn

app = FastAPI(title="Real-Time Fraud Detection API")

print("Starting server: Loading model and baseline data...")

# 1. Load the model
model = xgb.XGBClassifier()
model.load_model("xgboost_fraud_model.json")
EXPECTED_COLS = model.feature_names_in_

# 2. Load a SINGLE valid row from the dataset to act as our "Feature Store Template"
raw_df = pd.read_csv('data/DataCoSupplyChainDataset.csv', nrows=1, encoding='ISO-8859-1')
template_df = raw_df.drop(columns=[
    'Days for shipping (real)', 'Delivery Status', 'Late_delivery_risk', 
    'Shipping date (DateOrders)', 'Customer Email', 'Customer Password', 
    'Customer Fname', 'Customer Lname', 'Product Description', 
    'Product Image', 'Latitude', 'Longitude', 'Customer Zipcode', 
    'Order Zipcode', 'Order Item Cardprod Id', 'Order Status',
    'Customer Id', 'Order Id', 'Order Customer Id'
], errors='ignore')
template_df['customer_order_frequency'] = 1

# Reorder template to match XGBoost exactly
template_df = template_df[EXPECTED_COLS]

# 3. Define the API inputs (with VALID default values from the dataset!)
class Transaction(BaseModel):
    Type: str = "DEBIT"
    Category_Name: str = "Cleats"
    Customer_Country: str = "Puerto Rico" 
    Order_Country: str = "Mexico"
    Order_Item_Total: float = 150.0
    customer_order_frequency: int = 1

@app.post("/predict_fraud")
def predict_fraud(transaction: Transaction):
    # Get the user input
    input_dict = transaction.model_dump()
    
    # Create a fresh copy of our valid template
    df_input = template_df.copy()
    
    # Update only the fields the user provided
    df_input.at[0, 'Type'] = input_dict['Type']
    df_input.at[0, 'Category Name'] = input_dict['Category_Name']
    df_input.at[0, 'Customer Country'] = input_dict['Customer_Country']
    df_input.at[0, 'Order Country'] = input_dict['Order_Country']
    df_input.at[0, 'Order Item Total'] = input_dict['Order_Item_Total']
    df_input.at[0, 'customer_order_frequency'] = input_dict['customer_order_frequency']
    
    # Convert object/strings to categorical for XGBoost
    for col in df_input.select_dtypes(include=['object', 'string']).columns:
        df_input[col] = df_input[col].astype('category')
        
    # Catch any remaining unknown categories gracefully before XGBoost crashes
    try:
        fraud_prob = model.predict_proba(df_input)[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data Error: You sent a category the model has never seen. {str(e)}")
    
    action = "BLOCK" if fraud_prob > 0.70 else "APPROVE"
    
    return {
        "fraud_probability": float(fraud_prob),
        "recommended_action": action
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)