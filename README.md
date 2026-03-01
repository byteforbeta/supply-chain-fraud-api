# Real-Time Supply Chain Fraud Detection API

An end-to-end Machine Learning operations (MLOps) pipeline and real-time REST API built to detect suspected fraud in supply chain transactions.

## Architecture & Tech Stack
* **Model:** XGBoost Classifier (handling highly imbalanced data via `scale_pos_weight`)
* **Experiment Tracking:** Weights & Biases (W&B) for logging PR-AUC, Confusion Matrices, and hyperparameter tuning.
* **Model Registry:** W&B Artifacts for versioning the `.json` model weights.
* **Serving:** FastAPI & Uvicorn for sub-millisecond, real-time inference.
* **Data Processing:** Pandas (Feature engineering, frequency encoding, dynamic categorization).

## The Machine Learning Pipeline (`src/train.py`)
Fraud detection is inherently imbalanced. Instead of relying on misleading metrics like Accuracy, this model is optimized for **Precision-Recall AUC (PR-AUC)** and **Recall**. 

The training script automatically:
1. Cleans raw tabular data and maps high-cardinality IDs using frequency encoding.
2. Trains an XGBoost tree-based model.
3. Logs interactive ROC and PR curves to a W&B dashboard.
4. Serializes the winning model and pushes it to the remote Model Registry.

## Real-Time Inference (`src/app.py`)
The model is wrapped in a robust FastAPI application designed to act as a microservice for a payment processing backend. 
* **Data Validation:** Uses Pydantic to strictly type-check incoming JSON payloads.
* **Cold-Start Protection:** Implements a fallback baseline template to prevent the model from crashing when encountering unseen categorical variables in production.

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/supply-chain-fraud-api.git](https://github.com/YOUR_USERNAME/supply-chain-fraud-api.git)
   cd supply-chain-fraud-api
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add the Data:**
   * Download the `DataCoSupplyChainDataset.csv` from [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis).
   * Place it in a `data/` folder at the root of the project. (Note: The `data/` directory is ignored by Git to keep the repository lightweight).

4. **Start the API server:**
   ```bash
   python src/app.py
   ```

5. **Test the Endpoint:**
   Navigate to `http://localhost:8000/docs` in your browser to use the interactive Swagger UI and send test JSON payloads to the `/predict_fraud` endpoint.
