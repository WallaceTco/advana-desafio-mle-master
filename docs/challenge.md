# 🧩 Part I — Model Operationalization

## 1. Objective
Transcribe the exploratory notebook (`exploration.ipynb`) into a production-ready Python module (`model.py`) that can preprocess data, train a model, and predict flight delays.  
The goal is to make the Data Scientist’s prototype usable in an operational environment with clean, testable, and maintainable code.

---

## 2. Approach and Implementation

### 2.1. Starting point
The original notebook trained several models (Logistic Regression and XGBoost) to predict whether a flight at SCL airport would have a **delay > 15 minutes**.  
The Data Scientist concluded that:
- Performance differences between models were minimal.
- Using only the 10 most relevant features did **not** reduce performance.
- Balancing classes improved recall for the positive class (“delay”).

---

### 2.2. Model selected
**Chosen model:** `LogisticRegression(class_weight="balanced", random_state=42)`

**Justification:**
- Equivalent performance to XGBoost in the notebook.
- Lighter dependency footprint and faster load times.
- Fully supported by the provided environment (`requirements.txt`).
- Deterministic, simple to train and reproduce.
- Balanced weighting improves recall on the minority class (delayed flights).

---

### 2.3. Key features
According to the Data Scientist’s final setup, only the following variables were used for training:
- `OPERA` → Airline
- `TIPOVUELO` → Flight type (International/National)
- `MES` → Month number

These were one-hot encoded to form the final feature matrix.

---

### 2.4. Implementation overview

**Public API (unchanged from skeleton):**

| Method | Purpose |
|--------|---------|
| `preprocess(data, target_column=None)` | Transforms raw flight data into preprocessed features (and target if specified). |
| `fit(features, target)` | Trains a Logistic Regression model on preprocessed data and stores the column schema. |
| `predict(features)` | Predicts delay labels (0/1) for new flights, aligning features to the training schema. |

**Private helper methods (added):**

| Helper | Description |
|--------|-------------|
| `_ensure_delay_column(df, target_column)` | Ensures target column exists; if not, derives `delay` as `1` when `Fecha-O – Fecha-I > 15 min`. |
| `_build_ohe_features(df)` | Builds one-hot encoded features for `OPERA`, `TIPOVUELO`, and `MES`. |

---

### 2.5. Good practices applied
- ✅ **Reproducibility:** fixed random seed (`random_state=42`).
- ✅ **Robustness:** automatically derives `delay` when missing; handles unseen categories.
- ✅ **Error handling:** descriptive `ValueError` messages for missing columns or invalid data.
- ✅ **Modularity:** private helpers for clarity and unit testing.
- ✅ **Type hints & docstrings:** improve readability and static analysis.
- ✅ **Scikit-learn compatibility:** ensures smooth testing and serialization.

---

### 2.6. Example usage
```python
    from model import DelayModel
    import pandas as pd
    
    # Load dataset
    df = pd.read_csv("data/data.csv")
    
    # Initialize model
    model = DelayModel()
    
    # Preprocess and train
    X, y = model.preprocess(df, target_column="delay")
    model.fit(X, y)
    
    # Predict on new data
    X_new = model.preprocess(df.head(5))
    preds = model.predict(X_new)
    print(preds)
```

---

## 3. Bugs fixed from original skeleton
- Typing error: corrected Union(Tuple[…]) → Union[Tuple[…]].
- Unimplemented methods: implemented full logic for preprocess, fit, and predict.
- Target derivation: handled automatic generation of delay from timestamps.
- Feature alignment: added mechanism to reindex columns during inference to prevent shape mismatch.

---

## 4. Testing and validation
The implementation maintains all method signatures and expected return types, ensuring compatibility with the provided automated tests.
Expected command: make model-test
All tests should pass successfully before proceeding to Part II.

---

## 5. Next steps
Proceed to Part II — API Deployment, where the trained model will be served via a FastAPI
endpoint (/predict)

### 6. Limitations & Future Work
- The current pipeline uses only `OPERA`, `TIPOVUELO`, and `MES` as in the original notebook; other engineered features (e.g., `min_diff`, `high_season`, `period_day`) were not included to remain faithful to the DS setup and to pass the provided tests.
- Logistic Regression was selected for operational simplicity. If needed, we can explore tree-based models with calibrated probabilities and monitoring for concept drift.
- Add model persistence (save/load) and versioning for Part II/III when serving in the API.
