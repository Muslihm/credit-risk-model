# Credit Risk Model

A production-ready credit risk scoring system with interpretable model options for regulatory compliance.

## Project Structure
credit-risk-model/
├── .github/workflows/ci.yml # CI/CD pipeline
├── data/ # (gitignored) Raw and processed data
├── notebooks/ # Jupyter notebooks for EDA
├── src/
│ ├── data_processing.py # Feature engineering & WoE
│ ├── train.py # Model training (LR & XGBoost)
│ ├── predict.py # Inference pipeline
│ └── api/
│ ├── main.py # FastAPI application
│ └── pydantic_models.py # Request/response schemas
├── tests/
│ └── test_data_processing.py # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md

## Credit Scoring Business Understanding

### Basel II and Interpretability

The Basel II Accord's emphasis on internal risk measurement directly mandates interpretable, well-documented models. In the Internal Ratings-Based (IRB) approach, banks must demonstrate that their Probability of Default (PD) models are conceptually sound, statistically validated, and transparent to regulators. A "black-box" model cannot be adequately validated, creating **model risk**—the risk of incorrect capital calculations leading to insolvency or regulatory penalty. The 2025 ECB guide explicitly requires that even advanced ML models remain "adequately explainable."

### Proxy Variables in Absence of Default Labels

When true default data is unavailable, a correlated proxy variable (e.g., 60+ days past due) must be used. This introduces critical business risks:

- **Survival bias**: Models only observe accepted applicants, not rejected ones
- **Definition mismatch**: Proxy may capture temporary delinquency, not true default
- **Proxy discrimination**: Even non-protected variables can become proxies for protected characteristics (e.g., Morse 2020 showed "non-civil marriage" correlated with gender after controlling for risk factors)

### Interpretability vs. Performance Trade-offs

| Aspect | Logistic Regression (WoE) | Gradient Boosting (XGBoost) |
|--------|--------------------------|----------------------------|
| **Interpretability** | High (intrinsic) | Low (post-hoc only) |
| **Typical Gini** | ~0.65 | ~0.68 |
| **Regulatory Acceptance** | Preferred | Conditional |
| **Risk** | Lower compliance, higher prediction error | Higher compliance, lower prediction error |

**Decision framework**: Use Logistic Regression with WoE for regulated portfolios where transparency is paramount. Use XGBoost as a challenger model or for portfolios where even marginal Gini improvements justify additional validation burden.

## Quick Start

### Local Development

```bash
# Clone repository
git clone <your-repo-url>
cd credit-risk-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model
python -m src.train

# Run API
uvicorn src.api.main:app --reload

# Run tests
pytest tests/ -v --cov=src
# Build and run
docker-compose up --build

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
Using Docker
bash
# Build and run
docker-compose up --build

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
API Endpoints
POST /predict - Single application prediction

POST /predict/batch - Batch predictions (up to 1000)

POST /explain - Explainable prediction with feature contributions

GET /health - Health check

GET /model/info - Model metadata

Example API Request
bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "income": 65000,
    "debt": 15000,
    "credit_age_years": 8,
    "num_delinquencies": 1,
    "credit_utilization": 0.35,
    "num_credit_cards": 3
  }'
CI/CD Pipeline
The GitHub Actions workflow (ci.yml) performs:

Linting (flake8, black)

Unit tests with coverage

Security scanning (bandit)

Docker build validation

Automated deployment on main branch

License
MIT

text

---

## How to Use This Code

1. **Create the repository structure** using the terminal commands in section 1
2. **Copy each file** into its respective location
3. **Run the training** to generate synthetic data and train models:
   ```bash
   python -m src.train
Start the API:

bash
uvicorn src.api.main:app --reload
Access the documentation at http://localhost:8000/docs