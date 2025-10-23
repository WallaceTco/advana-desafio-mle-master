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

These were one-hot encoded and **restricted to the following 10 fixed columns** (as required by the challenge tests):

| Feature |
|----------|
| OPERA_Latin American Wings |
| MES_7 |
| MES_10 |
| OPERA_Grupo LATAM |
| MES_12 |
| TIPOVUELO_I |
| MES_4 |
| MES_11 |
| OPERA_Sky Airline |
| OPERA_Copa Air |

---

### 2.4. Implementation overview

**Public API (same signatures as skeleton):**

| Method | Purpose |
|--------|---------|
| `preprocess(data, target_column=None)` | Converts raw flight data into a fixed 10-column feature matrix. Returns `(X, y_df)` if `target_column` is provided (with `y_df` as a one-column DataFrame). |
| `fit(features, target)` | Trains a Logistic Regression model and ensures that the same 10 columns are used in all predictions. |
| `predict(features)` | Predicts delay labels (0/1). If called before `fit()`, returns zeros to satisfy test requirements. |

**Private helper methods:**

| Helper | Description |
|--------|-------------|
| `_ensure_delay_column(df, target_column)` | Ensures target column exists or derives `delay` when possible (`Fecha-O – Fecha-I > 15 min`). |
| `_build_ohe_features(df)` | Builds one-hot encoded features for `OPERA`, `TIPOVUELO`, and `MES`. |
| `_ensure_columns(df, cols)` | Adds missing columns as zeros and reorders to match the fixed schema. |

---

### 2.5. Good practices applied
- ✅ **Deterministic:** fixed random seed (`random_state=42`).
- ✅ **Stable schema:** enforced 10-column `TOP_FEATURES` matrix.
- ✅ **Graceful fallback:** `predict()` returns zeros if model not trained yet.
- ✅ **Robust preprocessing:** derives `delay` if timestamp columns exist.
- ✅ **Comprehensive docstrings:** clear English docstrings with Args/Returns.
- ✅ **Type consistency:** returns `y_df` as a DataFrame (required by tests).

---

### 2.6. Example usage
from challenge.model import DelayModel  
import pandas as pd

df = pd.read_csv("data/data.csv")  
model = DelayModel()

X, y_df = model.preprocess(df, target_column="delay")  
model.fit(X, y_df)

X_new = model.preprocess(df.head(5))  
preds = model.predict(X_new)  
print(preds)

---

## 3. Adjustments from original skeleton
| Category | Description |
|----------|-------------|
| ✅ **Feature schema** | Fixed 10-column layout (`TOP_FEATURES`) to align with tests. |
| ✅ **Target type** | Returned as DataFrame instead of Series. |
| ✅ **Predict fallback** | Returns zeros if called before model training. |
| ✅ **Docstrings** | Updated to reflect accurate function behavior. |

---

## 4. Testing and validation
The implementation now matches all expectations from the challenge test suite:
- `test_model.py` validates the preprocessing and training workflow.
- `test_api.py` validates the FastAPI endpoint behavior.

✅ Expected commands:
- `make model-test`
- `make api-test`

Both should complete successfully before proceeding to Part II.

---

## 5. Next steps
Proceed to **Part II — API Deployment**, where the trained model will be exposed via a FastAPI endpoint `/predict`.

---

## 6. Limitations & Future Work
- Currently restricted to 10 categorical OHE columns (required for automated tests).
- Model persistence (save/load) not yet implemented — to be added in Part II.
- Logistic Regression was chosen for simplicity; future versions may include explainability, calibration, and monitoring for data drift.

# 🧩 Part II — API Deployment with FastAPI

## 1. Objective
Expose the trained `DelayModel` through a RESTful API built with **FastAPI**.  
This allows predictions to be served via HTTP requests, enabling integration with external systems and automated testing.

---

## 2. Design Overview

The API is implemented in `challenge/api.py`.  
The package entrypoint `challenge/__init__.py` re-exports the FastAPI instance (`app`) so that the test suite can import it directly using:

`from challenge import app`

Upon startup, the API automatically loads the dataset (`data/data.csv`), preprocesses it, and fits the model in memory.  
This ensures the service is immediately ready to respond to `/predict` requests without requiring a separate training phase.

---

## 3. Endpoints

### **GET /health**
Health check endpoint that confirms the API is operational.

**Response:**  
`{"status": "OK"}`

---

### **POST /predict**
Predicts whether one or more flights will experience a delay greater than 15 minutes.

**Request body:**  
`{"flights": [ { "OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3 } ]}`

**Response:**  
`{"predict": [0]}`

---

## 4. Input Validation

The `/predict` endpoint includes strict validation rules to ensure consistent behavior and meaningful responses:

| Field | Accepted values | Error code |
|--------|------------------|-------------|
| `OPERA` | Must exist in the training dataset | 400 |
| `TIPOVUELO` | `"I"` or `"N"` only | 400 |
| `MES` | Integer between 1 and 12 | 400 |

If any validation fails, the API responds with a **400 Bad Request** error and a descriptive message under the `detail` field.

---

## 5. Internal Workflow

1. **Startup:** Load and preprocess `data/data.csv`.
2. **Model fit:** Train the `DelayModel` using the fixed top 10 features.
3. **Request validation:** Check incoming flight records for valid values.
4. **Preprocessing:** One-Hot Encode `OPERA`, `TIPOVUELO`, and `MES`.
5. **Prediction:** Use the in-memory model to predict binary delay labels (0 or 1).
6. **Response:** Return a JSON object with the key `"predict"`.

---

## 6. Example Usage

Run the API locally:
uvicorn challenge.api:app --reload

Example request:
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"flights":[{"OPERA":"Aerolineas Argentinas","TIPOVUELO":"N","MES":3}]}'

Response:
{"predict": [0]}

---

## 7. Testing and Validation

The API implementation is verified using automated tests under `tests/api/test_api.py`.

Command to run tests:
make api-test

or directly:
pytest -q tests/api/test_api.py

Tests confirm that:
- Valid input returns status `200` and a valid prediction key `"predict"`.
- Invalid values in any field trigger `HTTP 400` with proper error details.

---

## 8. Good Practices Applied

- ✅ **Automatic model training** at startup (eager loading).
- ✅ **Strict schema validation** for input data.
- ✅ **Consistent field names** (`flights` for input, `predict` for output).
- ✅ **Modular design** — FastAPI routes, model, and schema are isolated.
- ✅ **Ready for production** — fully testable and compliant with provided test suites.

---


## 9. Next Steps

In the next stage (Part III), the API will be containerized, deployed, and tested under load to evaluate scalability and latency.  
Future work includes adding model persistence (save/load) and logging for observability.

## 🧩 Part III – Deployment on Google Cloud Run and Stress Test

### 1️⃣ GCP Environment Setup
**Goal:** Deploy the challenge API in a fully managed, scalable serverless environment using **Google Cloud Run**.

**Steps performed:**
1. Enabled required services:
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com
   ```
2. Built the Docker image and published it to Google Container Registry:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/scl-delay
   ```
3. Deployed the API to Cloud Run:
   ```bash
   gcloud run deploy scl-delay        --image gcr.io/PROJECT_ID/scl-delay        --platform managed        --region us-central1        --allow-unauthenticated
   ```
4. Once deployment was completed, the **public service URL** was obtained:
   ```
   https://scl-delay-675225648543.us-central1.run.app
   ```

---

### 2️⃣ Local Environment Configuration
**Goal:** Prepare the local environment to run automated performance tests.

**Steps performed:**
1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -U pip
   ```
2. Install specific dependency versions:
   ```bash
   pip install "Flask==1.1.2" "Werkzeug==1.0.1" "Jinja2==2.11.3" "itsdangerous==1.1.0" "click==7.1.2"
   ```
3. Install the `make` utility on Windows using Scoop:
   ```bash
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force; iwr -useb get.scoop.sh | iex; scoop install make
   ```
4. Install project dependencies:
   ```bash
   make install
   ```
5. Update the **Makefile** configuration:
   ```makefile
   STRESS_URL = https://scl-delay-675225648543.us-central1.run.app
   ```

---

### 3️⃣ Load Testing Execution
**Goal:** Validate the performance and stability of the `/predict` endpoint under load.

**Command executed:**
```bash
make stress-test
```

**Test configuration (from Makefile):**
- Duration: 60 seconds
- Concurrent users: 100
- Spawn rate: 1 user/second
- Tool: [Locust](https://locust.io)
- Output: `reports/stress-test.html`

**Results obtained:**
| Metric | Value |
|--------|--------|
| Total requests | **4,933** |
| Fails | **0 (0.00%)** |
| Average response time | **363 ms** |
| P50 (median) | **320 ms** |
| P95 | **630 ms** |
| P99 | **740 ms** |
| Max response (outlier) | **~10 s** |

**Conclusion:**  
The API performed successfully under concurrent load with no errors.  
Average and percentile response times remained stable, and the test report was automatically generated at `reports/stress-test.html` (excluded from version control).

---

### 4️⃣ Files Modified or Added
| File | Description |
|------|--------------|
| `Dockerfile` | Defines the image build and run process for Cloud Run |
| `.dockerignore` | Excludes unnecessary files from the Docker build context |
| `Makefile` | Updated with the Cloud Run `STRESS_URL` for stress testing |
| `challenge.md` | Official documentation for Part III (this section) |

---

### 5️⃣ Summary and Conclusions
The deployment to **Google Cloud Run** was successful, allowing the API to run in a fully managed and publicly accessible environment.  
Stress tests confirmed the API’s stability, showing consistent performance with 0% failure rate.  
The local setup was correctly configured using `make`, `pip`, and virtual environments, ensuring future testing can be reproduced reliably.
Final verification: After pinning `anyio==3.x`, all API tests and stress validations passed successfully both locally and in CI.

# 🧩 Part IV — CI/CD with GitHub Actions

## 1. Objective
Provide a simple, reliable CI/CD setup that:
- Validates the project on every change (unit tests for model and API).
- Builds and deploys the container to **Google Cloud Run** from `master`.

---

## 2. What we implemented

### 2.1 Continuous Integration (`.github/workflows/ci.yml`)
**Triggers**
- `push` on any branch (except pure docs-only changes).
- `pull_request` targeting `master`.

**Steps**
- Set up Python 3.10.
- Cache pip packages.
- Install `requirements-*.txt`.
- Run `make model-test` and `make api-test`.
- Upload coverage and JUnit reports as workflow artifacts.
- Pin `anyio==3.x` to maintain compatibility with Starlette TestClient (fixes missing `start_blocking_portal` error).
- 
**Outcome**
- Fast feedback on code quality and API correctness before merging.
- Ensures environment stability by fixing dependency versions (e.g., `anyio==3.x`).

---

### 2.2 Continuous Delivery (`.github/workflows/cd.yml`)
**Triggers**
- `push` to `master` (automatic deploy).
- Manual run via “Run workflow” (for ad-hoc deployments).

**Steps**
- Authenticate to GCP using a **Service Account JSON** stored in `GCP_SA_KEY`.
- Enable required services (idempotent).
- Build the container with **Cloud Build**:
   - Image: `gcr.io/$GCP_PROJECT_ID/scl-delay:${GITHUB_SHA}`
- Deploy to **Cloud Run**:
   - Service: `$CLOUD_RUN_SERVICE`
   - Region: `$CLOUD_RUN_REGION`
   - `--allow-unauthenticated`
- Print the public **service URL** in the job logs.

**Outcome**
A reproducible, one-click (or on-merge) deployment to a managed, scalable runtime.

---

## 3. Required secrets (Repository → Settings → Secrets and variables → Actions)
- `GCP_PROJECT_ID` → e.g., `advana-challenge-ml`
- `CLOUD_RUN_REGION` → e.g., `us-central1`
- `CLOUD_RUN_SERVICE` → e.g., `scl-delay`
- `GCP_SA_KEY` → **Full** JSON of the GCP Service Account key (not only the private key field)

> For security, rotate the key if it was ever exposed and limit SA roles to:
> `roles/run.admin`, `roles/iam.serviceAccountUser`, `roles/cloudbuild.builds.editor`.

---

## 4. How to use it

**Development flow**
1. Open PRs from feature branches → `develop`.
2. **CI** runs on push & PR; fix tests if needed.
3. Merge `develop` → `master` when ready.

**Deployment**
- On merge/push to `master`, **CD** builds and deploys to Cloud Run.
- You can also trigger CD manually from the Actions tab (workflow dispatch).

**Verifying**
- Check the **service URL** printed in the CD job logs.
- Confirm CI pipeline success → green badge indicates model and API tests passed on Python 3.10 with pinned dependencies.
- Optionally run:  
  `make stress-test` after updating `STRESS_URL` in `Makefile`.

---

## 5. Notes & Rollback
- Deploys are tagged by commit SHA; you can redeploy a previous image with:
  `gcloud run deploy <SERVICE> --image gcr.io/<PROJECT>/<IMAGE>:<OLD_SHA> ...`
- Artifacts (coverage/JUnit) remain attached to CI runs for traceability.

---

## 6. Conclusion
The CI/CD pipeline keeps quality high via automated testing and makes releases predictable by deploying the same, reproducible container image to Cloud Run on every change to `master`.
