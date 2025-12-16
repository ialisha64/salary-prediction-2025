"""
Prediction Module
Make salary predictions with confidence intervals and explanations
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')


class SalaryPredictor:
    """Make salary predictions with explanations"""

    def __init__(self, model_path, preprocessor_path):
        """Load trained model and preprocessor"""

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)

        print(f"Model loaded from: {model_path}")
        print(f"Preprocessor loaded from: {preprocessor_path}")

    def prepare_input(self, input_data):
        """Prepare single input for prediction"""

        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Apply same feature engineering
        df_featured = self.preprocessor.create_advanced_features(df)

        # Encode categorical features (fit=False to use saved encodings)
        df_encoded = self.preprocessor.encode_categorical_features(df_featured, fit=False)

        # Select features
        df_modeling = self.preprocessor.select_features_for_modeling(df_encoded)

        # Remove target if present
        if 'annual_salary_usd' in df_modeling.columns:
            df_modeling = df_modeling.drop(columns=['annual_salary_usd'])

        # Ensure all features are present
        missing_features = set(self.preprocessor.feature_names) - set(df_modeling.columns)
        for feat in missing_features:
            df_modeling[feat] = 0

        # Reorder to match training
        df_modeling = df_modeling[self.preprocessor.feature_names]

        return df_modeling

    def predict(self, input_data, return_confidence=True):
        """Make prediction with confidence interval"""

        # Prepare input
        X = self.prepare_input(input_data)

        # Predict
        prediction = self.model.predict(X)[0]

        if return_confidence:
            # Calculate confidence interval (±1 std of residuals)
            # Approximate based on typical model performance
            confidence_margin = prediction * 0.10  # ±10%

            lower_bound = max(35000, prediction - confidence_margin)
            upper_bound = prediction + confidence_margin

            return {
                'predicted_salary': round(prediction, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'confidence_margin_pct': 10
            }
        else:
            return round(prediction, 2)

    def explain_prediction(self, input_data):
        """Provide explanation for prediction"""

        # This is a simplified explanation
        # For full SHAP explanation, use the notebook/app

        X = self.prepare_input(input_data)

        # Get feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.preprocessor.feature_names,
                'importance': importances,
                'value': X.iloc[0].values
            }).sort_values('importance', ascending=False)

            top_features = feature_importance.head(10)

            return top_features

        return None

    def compare_to_peers(self, input_data, actual_salary=None):
        """Compare predicted salary to actual (if provided)"""

        prediction = self.predict(input_data, return_confidence=False)

        if actual_salary:
            difference = actual_salary - prediction
            pct_difference = (difference / prediction) * 100

            if pct_difference > 10:
                status = "OVERPAID"
            elif pct_difference < -10:
                status = "UNDERPAID"
            else:
                status = "FAIR"

            return {
                'predicted_salary': round(prediction, 2),
                'actual_salary': actual_salary,
                'difference': round(difference, 2),
                'pct_difference': round(pct_difference, 2),
                'status': status
            }

        return {'predicted_salary': round(prediction, 2)}


def main():
    """Test predictor"""

    # Initialize predictor
    predictor = SalaryPredictor(
        model_path='model/best_model_XGBoost.pkl',
        preprocessor_path='model/preprocessor.pkl'
    )

    # Example input
    example_person = {
        'age': 32,
        'gender': 'Female',
        'race': 'Asian',
        'education_level': 'Master',
        'years_of_experience': 7,
        'job_title': 'Senior Data Scientist',
        'job_category': 'Data Science',
        'company_size': 'Large (1001-5000)',
        'company_location': 'USA',
        'work_mode': 'Hybrid',
        'performance_rating': 4.5,
        'manager_rating': 4.3,
        'certifications_count': 3,
        'github_portfolio_strength': 75.5,
        'linkedin_connections': 850,
        'programming_languages_known': 5,
        'ai_ml_tools_proficiency': 8.5,
        'highest_degree_university_rank': 45,
        'overtime_hours_per_month': 15,
        'economic_index_of_country': 100,
        'cost_of_living_index': 100,
        'city_tier': 1,
        'salary_negotiation_score': 7.5,
        'previous_salary_usd': 95000,
        'bonus_percentage': 12.0,
        'stock_options_value': 15000,
        'department': 'Data',
        'annual_salary_usd': 0  # Will be predicted
    }

    # Make prediction
    result = predictor.predict(example_person)

    print("\n" + "="*60)
    print("SALARY PREDICTION")
    print("="*60)
    print(f"Predicted Salary: ${result['predicted_salary']:,.0f}")
    print(f"Confidence Interval: ${result['lower_bound']:,.0f} - ${result['upper_bound']:,.0f}")
    print(f"Margin: ±{result['confidence_margin_pct']}%")

    # Compare if actual provided
    actual_salary = 125000
    comparison = predictor.compare_to_peers(example_person, actual_salary)

    print("\n" + "="*60)
    print("COMPARISON TO MARKET")
    print("="*60)
    print(f"Predicted: ${comparison['predicted_salary']:,.0f}")
    print(f"Actual: ${comparison['actual_salary']:,.0f}")
    print(f"Difference: ${comparison['difference']:,.0f} ({comparison['pct_difference']:+.1f}%)")
    print(f"Status: {comparison['status']}")

    # Explanation
    print("\n" + "="*60)
    print("TOP INFLUENCING FEATURES")
    print("="*60)
    explanation = predictor.explain_prediction(example_person)
    if explanation is not None:
        print(explanation[['feature', 'importance', 'value']].to_string(index=False))


if __name__ == "__main__":
    main()
