"""
Preprocessing and Feature Engineering Module
Advanced feature engineering for salary prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')


class SalaryDataPreprocessor:
    """Advanced preprocessing and feature engineering for salary data"""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []

    def create_advanced_features(self, df):
        """Create 15+ advanced engineered features"""

        df = df.copy()

        print("Creating advanced features...")

        # 1. Experience to age ratio (career intensity)
        df['experience_to_age_ratio'] = df['years_of_experience'] / (df['age'] + 1)

        # 2. Total compensation
        df['total_compensation'] = (
            df['annual_salary_usd'] +
            (df['annual_salary_usd'] * df['bonus_percentage'] / 100) +
            df['stock_options_value']
        )

        # 3. Seniority score from job title
        df['title_seniority_score'] = df['job_title'].apply(self._extract_seniority_score)

        # 4. Career progression velocity (experience gained per year of age)
        df['promotion_velocity'] = df['years_of_experience'] / (df['age'] - 21)
        df['promotion_velocity'] = df['promotion_velocity'].replace([np.inf, -np.inf], 0).fillna(0)

        # 5. Education ROI score
        edu_years = {'High School': 0, 'Associate': 2, 'Bachelor': 4, 'Master': 6, 'PhD': 10}
        df['education_years'] = df['education_level'].map(edu_years)
        df['education_roi'] = df['annual_salary_usd'] / (df['education_years'] + 1)

        # 6. Skills diversity index
        df['skills_diversity_index'] = (
            df['programming_languages_known'] +
            df['ai_ml_tools_proficiency'] +
            df['certifications_count']
        ) / 3

        # 7. Digital presence score
        df['digital_presence_score'] = (
            (df['github_portfolio_strength'] / 100) * 0.5 +
            (np.log1p(df['linkedin_connections']) / 10) * 0.5
        )

        # 8. Work-life balance score (inverse of overtime)
        df['work_life_balance_score'] = 100 - (df['overtime_hours_per_month'] / 80 * 100)
        df['work_life_balance_score'] = df['work_life_balance_score'].clip(0, 100)

        # 9. Performance composite score
        df['performance_composite'] = (
            df['performance_rating'] * 0.6 +
            df['manager_rating'] * 0.4
        )

        # 10. Location advantage score
        df['location_advantage'] = (
            df['economic_index_of_country'] / df['cost_of_living_index']
        )

        # 11. Remote work premium indicator
        df['is_remote'] = (df['work_mode'] == 'Remote').astype(int)
        df['is_hybrid'] = (df['work_mode'] == 'Hybrid').astype(int)

        # 12. Tech role indicator
        df['is_tech_role'] = df['job_category'].isin(['Tech', 'Data Science', 'Engineering']).astype(int)

        # 13. Elite university flag
        df['elite_university'] = (df['highest_degree_university_rank'] <= 50).astype(int)
        df['elite_university'] = df['elite_university'] * (df['highest_degree_university_rank'] > 0).astype(int)

        # 14. Company size score (numerical)
        size_mapping = {
            'Startup (1-50)': 1,
            'Small (51-200)': 2,
            'Medium (201-1000)': 3,
            'Large (1001-5000)': 4,
            'Enterprise (5000+)': 5
        }
        df['company_size_score'] = df['company_size'].map(size_mapping)

        # 15. Negotiation power index
        df['negotiation_power'] = (
            df['salary_negotiation_score'] * 0.3 +
            df['years_of_experience'] * 0.3 +
            df['performance_rating'] * 2 * 0.4
        )

        # 16. Career stage
        df['career_stage'] = pd.cut(
            df['years_of_experience'],
            bins=[-1, 2, 5, 10, 20, 100],
            labels=['Entry', 'Junior', 'Mid', 'Senior', 'Executive']
        )

        # 17. Skill rarity index (high skills in non-tech = rare = valuable)
        df['skill_rarity_index'] = (
            df['programming_languages_known'] * (1 - df['is_tech_role']) * 2 +
            df['programming_languages_known'] * df['is_tech_role']
        )

        # 18. Previous salary growth potential
        df['salary_growth_potential'] = np.where(
            df['previous_salary_usd'] > 0,
            (df['annual_salary_usd'] - df['previous_salary_usd']) / (df['previous_salary_usd'] + 1),
            0
        )

        # 19. Overqualification score
        df['overqualification_score'] = (
            df['education_years'] - (df['years_of_experience'] * 0.5)
        ).clip(lower=0)

        # 20. City tier economic multiplier
        df['city_tier_multiplier'] = df['city_tier'].map({1: 1.2, 2: 1.0, 3: 0.85})

        print(f"Created {20} advanced features")

        return df

    def _extract_seniority_score(self, title):
        """Extract numerical seniority score from job title"""
        title_lower = str(title).lower()

        if any(word in title_lower for word in ['cto', 'cfo', 'cmo', 'cpo', 'coo', 'ceo']):
            return 10
        elif any(word in title_lower for word in ['vp', 'vice president']):
            return 9
        elif any(word in title_lower for word in ['director', 'head of']):
            return 8
        elif any(word in title_lower for word in ['principal', 'distinguished']):
            return 7
        elif any(word in title_lower for word in ['staff', 'lead']):
            return 6
        elif any(word in title_lower for word in ['manager']):
            return 5
        elif any(word in title_lower for word in ['senior', 'sr']):
            return 4
        elif any(word in title_lower for word in ['junior', 'jr', 'associate']):
            return 2
        else:
            return 3

    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features with multiple strategies"""

        df = df.copy()

        print("Encoding categorical features...")

        # Target encoding for high cardinality features (job_title, company_location)
        if fit:
            self.target_means = {}

        # For job_title - use target encoding
        if 'job_title' in df.columns:
            if fit:
                self.target_means['job_title'] = df.groupby('job_title')['annual_salary_usd'].mean().to_dict()
            df['job_title_encoded'] = df['job_title'].map(self.target_means['job_title'])
            df['job_title_encoded'] = df['job_title_encoded'].fillna(df['annual_salary_usd'].mean())

        # For company_location - use target encoding
        if 'company_location' in df.columns:
            if fit:
                self.target_means['company_location'] = df.groupby('company_location')['annual_salary_usd'].mean().to_dict()
            df['location_encoded'] = df['company_location'].map(self.target_means['company_location'])
            df['location_encoded'] = df['location_encoded'].fillna(df['annual_salary_usd'].mean())

        # Frequency encoding for department
        if 'department' in df.columns:
            if fit:
                self.freq_encoding = df['department'].value_counts(normalize=True).to_dict()
            df['department_frequency'] = df['department'].map(self.freq_encoding)
            df['department_frequency'] = df['department_frequency'].fillna(0)

        # Label encoding for ordinal features
        ordinal_features = ['education_level', 'career_stage']

        for feature in ordinal_features:
            if feature in df.columns:
                if fit:
                    self.label_encoders[feature] = LabelEncoder()
                    df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
                else:
                    # Handle unseen categories
                    df[f'{feature}_encoded'] = df[feature].astype(str).apply(
                        lambda x: self.label_encoders[feature].transform([x])[0]
                        if x in self.label_encoders[feature].classes_
                        else -1
                    )

        # One-hot encoding for low cardinality categoricals
        one_hot_features = ['gender', 'race', 'work_mode', 'job_category']

        for feature in one_hot_features:
            if feature in df.columns:
                dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
                df = pd.concat([df, dummies], axis=1)

        print(f"Encoded categorical features")

        return df

    def select_features_for_modeling(self, df):
        """Select final features for modeling"""

        # Features to drop
        drop_features = [
            'employee_id',
            'job_title',
            'company_location',
            'department',
            'education_level',
            'career_stage',
            'gender',
            'race',
            'work_mode',
            'job_category',
            'company_size',
            'total_compensation',  # This includes target variable
            'previous_salary_usd',  # Can cause data leakage
            'bonus_percentage',  # Part of compensation
            'stock_options_value',  # Part of compensation
        ]

        # Keep only features that exist
        drop_features = [f for f in drop_features if f in df.columns]

        modeling_df = df.drop(columns=drop_features, errors='ignore')

        return modeling_df

    def prepare_data(self, df, target_col='annual_salary_usd', test_size=0.2, random_state=42):
        """Complete data preparation pipeline"""

        print("\n" + "="*60)
        print("DATA PREPARATION PIPELINE")
        print("="*60)

        # Create advanced features
        df_featured = self.create_advanced_features(df)

        # Encode categorical features
        df_encoded = self.encode_categorical_features(df_featured, fit=True)

        # Select features for modeling
        df_modeling = self.select_features_for_modeling(df_encoded)

        # Separate features and target
        if target_col in df_modeling.columns:
            y = df_modeling[target_col]
            X = df_modeling.drop(columns=[target_col])
        else:
            raise ValueError(f"Target column {target_col} not found")

        # Store feature names
        self.feature_names = X.columns.tolist()
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

        print(f"\nFinal feature count: {len(self.feature_names)}")
        print(f"Target variable: {target_col}")
        print(f"Dataset shape: {X.shape}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale numerical features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[self.numerical_features] = self.scaler.fit_transform(
            X_train[self.numerical_features]
        )
        X_test_scaled[self.numerical_features] = self.scaler.transform(
            X_test[self.numerical_features]
        )

        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print("\nData preparation complete!")

        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

    def save_preprocessor(self, filepath='model/preprocessor.pkl'):
        """Save the preprocessor for later use"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load_preprocessor(filepath='model/preprocessor.pkl'):
        """Load a saved preprocessor"""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def main():
    """Test the preprocessor"""

    # Load data
    df = pd.read_csv('../data/salary_data_2025.csv')
    print(f"Loaded dataset: {df.shape}")

    # Initialize preprocessor
    preprocessor = SalaryDataPreprocessor()

    # Prepare data
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocessor.prepare_data(df)

    # Print feature importance preview
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Total features created: {len(preprocessor.feature_names)}")
    print(f"\nFeature list:")
    for i, feat in enumerate(preprocessor.feature_names[:20], 1):
        print(f"{i}. {feat}")
    if len(preprocessor.feature_names) > 20:
        print(f"... and {len(preprocessor.feature_names) - 20} more features")

    # Save preprocessor
    preprocessor.save_preprocessor()

    return preprocessor, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocessor, X_train, X_test, y_train, y_test = main()
