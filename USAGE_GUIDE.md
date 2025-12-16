# Complete Usage Guide - Salary Prediction System 2025

## Table of Contents
1. [Quick Start](#quick-start)
2. [Detailed Workflow](#detailed-workflow)
3. [Running the Web App](#running-the-web-app)
4. [Model Training Options](#model-training-options)
5. [Making Predictions](#making-predictions)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Step 1: Installation
```bash
# Clone repository
cd salary-prediction-2025

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Generate Data
```bash
cd src
python data_generator.py
```

**Expected output:**
- `data/salary_data_2025.csv` (50,000 rows, ~15MB)
- Console output showing dataset statistics

### Step 3: Train Models
```bash
python train.py
```

**What happens:**
1. Loads dataset
2. Engineers features (15+ new features)
3. Trains 10+ models
4. Optionally runs hyperparameter optimization
5. Saves best model

**Expected time:** 5-15 minutes (depending on hardware)

**Outputs:**
- `model/best_model_XGBoost.pkl`
- `model/preprocessor.pkl`
- `model/model_comparison.csv`
- `model/model_results.json`

### Step 4: Launch Web App
```bash
cd ..
streamlit run app.py
```

**Access:** http://localhost:8501

---

## Detailed Workflow

### Data Generation

The data generator creates a realistic synthetic dataset with:
- **50,000 employees**
- **29 features** including demographics, education, skills, performance
- **Realistic correlations** (e.g., education ↔ salary, experience ↔ salary)
- **Observable biases** for fairness testing

**Customization:**
```python
# In src/data_generator.py, modify:

# Change number of samples
generator = SalaryDataGenerator(n_samples=100000)

# Adjust salary range
salary = np.clip(salary, 30000, 600000)

# Modify pay gaps
gender_mult = {'Male': 1.00, 'Female': 0.96, 'Non-binary': 0.95}  # Reduced gap
```

### Feature Engineering

The preprocessor (`src/preprocess.py`) creates 15+ engineered features:

**Example usage:**
```python
from preprocess import SalaryDataPreprocessor

# Initialize
preprocessor = SalaryDataPreprocessor()

# Load data
df = pd.read_csv('data/salary_data_2025.csv')

# Create features
df_featured = preprocessor.create_advanced_features(df)

# View new features
new_features = [
    'experience_to_age_ratio',
    'total_compensation',
    'title_seniority_score',
    'skills_diversity_index',
    'negotiation_power',
    # ... and 10 more
]
```

### Model Training

**Train all models:**
```bash
python src/train.py
```

**Train with optimization (slower but better performance):**
```bash
python src/train.py
# When prompted: y
```

**Models trained:**
1. **Baseline:** Linear Regression, Ridge, Lasso, Decision Tree
2. **Advanced:** XGBoost, LightGBM, CatBoost
3. **Ensemble:** Voting, Stacking
4. **Optimized:** XGBoost with Optuna tuning

**Evaluation metrics:**
- R² Score (variance explained)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Within ±10% (custom metric)

---

## Running the Web App

### Features

**1. Salary Prediction Tab**
- Input your profile (25+ fields)
- Get predicted salary with confidence interval
- Compare to current salary
- Receive personalized insights

**2. Negotiation Simulator**
- What-if analysis
- Test scenarios:
  - Improve education
  - Learn new skills
  - Switch to remote work
  - Better negotiation

**3. Salary Dashboard**
- Global trends
- Pay gap analysis
- Category comparisons
- Interactive visualizations

### Example: Making a Prediction

1. **Fill sidebar form:**
   - Age: 32
   - Gender: Female
   - Education: Master
   - Experience: 7 years
   - Job: Senior Data Scientist
   - Location: USA
   - Skills: 5 languages, AI proficiency 8/10

2. **Click "Predict My Salary"**

3. **Get results:**
   ```
   Predicted Salary: $145,000
   Confidence: $130,500 - $159,500 (±10%)

   Status: UNDERPAID by 12%
   (if current salary < $145k)
   ```

4. **View insights:**
   - Top factors affecting salary
   - Career advice
   - Feature importance

---

## Making Predictions Programmatically

### Option 1: Using predict.py

```python
from predict import SalaryPredictor

# Initialize
predictor = SalaryPredictor(
    model_path='model/best_model_XGBoost.pkl',
    preprocessor_path='model/preprocessor.pkl'
)

# Example person
person = {
    'age': 28,
    'gender': 'Male',
    'race': 'Asian',
    'education_level': 'Bachelor',
    'years_of_experience': 5,
    'job_title': 'Software Engineer',
    'job_category': 'Tech',
    'company_size': 'Large (1001-5000)',
    'company_location': 'USA',
    'work_mode': 'Hybrid',
    'performance_rating': 4.2,
    'manager_rating': 4.0,
    'certifications_count': 2,
    'github_portfolio_strength': 65.0,
    'linkedin_connections': 450,
    'programming_languages_known': 4,
    'ai_ml_tools_proficiency': 6.0,
    'highest_degree_university_rank': 120,
    'overtime_hours_per_month': 12,
    'economic_index_of_country': 100,
    'cost_of_living_index': 100,
    'city_tier': 1,
    'salary_negotiation_score': 6.5,
    'previous_salary_usd': 75000,
    'bonus_percentage': 10.0,
    'stock_options_value': 10000,
    'department': 'Engineering',
    'annual_salary_usd': 0  # Placeholder
}

# Predict
result = predictor.predict(person)

print(f"Predicted: ${result['predicted_salary']:,.0f}")
print(f"Range: ${result['lower_bound']:,.0f} - ${result['upper_bound']:,.0f}")

# Compare to actual
comparison = predictor.compare_to_peers(person, actual_salary=95000)
print(f"Status: {comparison['status']}")
print(f"Difference: ${comparison['difference']:,.0f}")
```

### Option 2: Batch Predictions

```python
import pandas as pd
from predict import SalaryPredictor

# Load predictor
predictor = SalaryPredictor(
    model_path='model/best_model_XGBoost.pkl',
    preprocessor_path='model/preprocessor.pkl'
)

# Load multiple people
candidates_df = pd.read_csv('candidates.csv')

# Predict for all
predictions = []
for idx, row in candidates_df.iterrows():
    pred = predictor.predict(row.to_dict(), return_confidence=False)
    predictions.append(pred)

candidates_df['predicted_salary'] = predictions

# Save results
candidates_df.to_csv('predictions_output.csv', index=False)
```

---

## Model Training Options

### Basic Training (No Optimization)
```bash
python src/train.py
# When asked about optimization, type: n
```

**Time:** ~5 minutes
**Models:** 9 models trained
**Best R²:** ~0.93-0.94

### With Hyperparameter Optimization
```bash
python src/train.py
# When asked about optimization, type: y
```

**Time:** ~15-30 minutes
**Models:** 10 models (including optimized XGBoost)
**Best R²:** ~0.94-0.95

### Custom Training

Edit `src/train.py`:

```python
# Change train-test split
X_train, X_test, y_train, y_test = trainer.prepare_data(df, test_size=0.3)

# Adjust XGBoost parameters
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,  # More trees
    learning_rate=0.03,  # Slower learning
    max_depth=10,        # Deeper trees
    # ... other params
)

# More Optuna trials
trainer.optimize_xgboost(n_trials=100)  # Default: 50
```

---

## Exploratory Data Analysis

### Run EDA Notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

**What's included:**
- 20+ visualizations
- Statistical hypothesis tests
- Pay gap analysis
- Correlation heatmaps
- Distribution plots

**Key sections:**
1. Data quality check
2. Target variable analysis
3. Demographics (gender, race, age)
4. Education analysis
5. Job category & titles
6. Experience curves
7. Company & location
8. Skills & performance
9. Correlation analysis
10. Key insights summary

---

## Troubleshooting

### Issue: Model files not found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'model/best_model_XGBoost.pkl'
```

**Solution:**
```bash
# Train the model first
python src/train.py
```

### Issue: Import errors

**Error:**
```
ModuleNotFoundError: No module named 'xgboost'
```

**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: Dataset not generated

**Error:**
```
FileNotFoundError: data/salary_data_2025.csv
```

**Solution:**
```bash
cd src
python data_generator.py
```

### Issue: Streamlit app slow

**Possible causes:**
1. Large dataset loaded multiple times
2. Model predictions not cached

**Solution:**
- App already uses `@st.cache_resource` and `@st.cache_data`
- If still slow, reduce dataset size in data_generator.py

### Issue: Poor model performance

**Check:**
1. Data quality: `df.isnull().sum()`
2. Feature engineering: Verify all 15+ features created
3. Scaling: Ensure preprocessor is fitted correctly
4. Outliers: Check for extreme values

**Debug:**
```python
# In train.py, add:
print(f"Training samples: {len(X_train)}")
print(f"Features: {X_train.columns.tolist()}")
print(f"Target range: {y_train.min()} - {y_train.max()}")
```

---

## Advanced Usage

### Custom Feature Engineering

Add your own features in `src/preprocess.py`:

```python
def create_advanced_features(self, df):
    # ... existing features ...

    # Add your custom feature
    df['your_custom_feature'] = (
        df['some_column'] * df['other_column']
    )

    return df
```

### Export Model for Production

```python
import joblib

# Save with joblib (better for sklearn-compatible models)
joblib.dump(model, 'production_model.joblib')

# Load
loaded_model = joblib.load('production_model.joblib')
```

### REST API (Future Enhancement)

```python
from fastapi import FastAPI
from predict import SalaryPredictor

app = FastAPI()
predictor = SalaryPredictor(...)

@app.post("/predict")
def predict_salary(person: dict):
    result = predictor.predict(person)
    return result
```

---

## Performance Benchmarks

**Hardware:** Apple M1 MacBook Pro, 16GB RAM

| Task | Time | Output Size |
|------|------|-------------|
| Data Generation | 15 seconds | 15 MB |
| Feature Engineering | 8 seconds | 25 MB |
| Training (no opt) | 5 minutes | 50 MB |
| Training (with opt) | 20 minutes | 50 MB |
| Single Prediction | <100ms | - |
| Batch 1000 | 3 seconds | - |

---

## Next Steps

1. **Experiment with features**: Try creating domain-specific features
2. **Try TabNet**: Implement deep learning model (see requirements.txt)
3. **Add SHAP**: Implement detailed SHAP explanations in app
4. **Deploy**: Host on Streamlit Cloud or Heroku
5. **API**: Build FastAPI REST endpoint
6. **Mobile**: Create mobile-friendly UI

---

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues
- Review code comments in modules

## License

MIT License - See LICENSE file for details

---

**Happy Predicting! 💰📊🚀**
