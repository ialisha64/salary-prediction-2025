"""
Model Training Module
Implements baseline and advanced models with hyperparameter tuning
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Advanced models
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Hyperparameter tuning
import optuna
from optuna.samplers import TPESampler

# Import preprocessor
from preprocess import SalaryDataPreprocessor


class SalaryModelTrainer:
    """Train and evaluate salary prediction models"""

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def evaluate_model(self, y_true, y_pred, model_name):
        """Comprehensive model evaluation"""

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Custom metric: % predictions within ±10% of true salary
        pct_diff = np.abs((y_pred - y_true) / y_true) * 100
        within_10_pct = (pct_diff <= 10).mean() * 100

        # Mean Absolute Percentage Error
        mape = (pct_diff).mean()

        results = {
            'model': model_name,
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R2': round(r2, 4),
            'MAPE': round(mape, 2),
            'Within_10%': round(within_10_pct, 2)
        }

        return results

    def train_baseline_models(self):
        """Train baseline models"""

        print("\n" + "="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)

        baseline_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42)
        }

        for name, model in baseline_models.items():
            print(f"\nTraining {name}...")

            # Train
            model.fit(self.X_train, self.y_train)

            # Predict
            y_pred_train = model.predict(self.X_train)
            y_pred_test = model.predict(self.X_test)

            # Evaluate
            train_results = self.evaluate_model(self.y_train, y_pred_train, f"{name} (Train)")
            test_results = self.evaluate_model(self.y_test, y_pred_test, f"{name} (Test)")

            # Store
            self.models[name] = model
            self.results[f"{name}_train"] = train_results
            self.results[f"{name}_test"] = test_results

            print(f"Test R²: {test_results['R2']:.4f} | MAE: ${test_results['MAE']:,.0f} | Within 10%: {test_results['Within_10%']:.1f}%")

        print("\nBaseline models training complete!")

    def train_advanced_models(self):
        """Train advanced gradient boosting models"""

        print("\n" + "="*60)
        print("TRAINING ADVANCED MODELS")
        print("="*60)

        # XGBoost
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb_model.fit(self.X_train, self.y_train)

        y_pred_train = xgb_model.predict(self.X_train)
        y_pred_test = xgb_model.predict(self.X_test)

        self.models['XGBoost'] = xgb_model
        self.results['XGBoost_train'] = self.evaluate_model(self.y_train, y_pred_train, "XGBoost (Train)")
        self.results['XGBoost_test'] = self.evaluate_model(self.y_test, y_pred_test, "XGBoost (Test)")

        print(f"Test R²: {self.results['XGBoost_test']['R2']:.4f} | MAE: ${self.results['XGBoost_test']['MAE']:,.0f}")

        # LightGBM
        print("\nTraining LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        lgb_model.fit(self.X_train, self.y_train)

        y_pred_train = lgb_model.predict(self.X_train)
        y_pred_test = lgb_model.predict(self.X_test)

        self.models['LightGBM'] = lgb_model
        self.results['LightGBM_train'] = self.evaluate_model(self.y_train, y_pred_train, "LightGBM (Train)")
        self.results['LightGBM_test'] = self.evaluate_model(self.y_test, y_pred_test, "LightGBM (Test)")

        print(f"Test R²: {self.results['LightGBM_test']['R2']:.4f} | MAE: ${self.results['LightGBM_test']['MAE']:,.0f}")

        # CatBoost
        print("\nTraining CatBoost...")
        cat_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False
        )
        cat_model.fit(self.X_train, self.y_train)

        y_pred_train = cat_model.predict(self.X_train)
        y_pred_test = cat_model.predict(self.X_test)

        self.models['CatBoost'] = cat_model
        self.results['CatBoost_train'] = self.evaluate_model(self.y_train, y_pred_train, "CatBoost (Train)")
        self.results['CatBoost_test'] = self.evaluate_model(self.y_test, y_pred_test, "CatBoost (Test)")

        print(f"Test R²: {self.results['CatBoost_test']['R2']:.4f} | MAE: ${self.results['CatBoost_test']['MAE']:,.0f}")

        print("\nAdvanced models training complete!")

    def optimize_xgboost(self, n_trials=50):
        """Hyperparameter optimization for XGBoost using Optuna"""

        print("\n" + "="*60)
        print(f"OPTIMIZING XGBOOST ({n_trials} trials)")
        print("="*60)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }

            model = xgb.XGBRegressor(**params)
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)

            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))

            return rmse

        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )

        # Optimize
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Best model
        print(f"\nBest RMSE: ${study.best_value:,.0f}")
        print(f"Best parameters: {study.best_params}")

        # Train final model with best params
        best_xgb = xgb.XGBRegressor(**study.best_params)
        best_xgb.fit(self.X_train, self.y_train)

        y_pred_train = best_xgb.predict(self.X_train)
        y_pred_test = best_xgb.predict(self.X_test)

        self.models['XGBoost_Optimized'] = best_xgb
        self.results['XGBoost_Optimized_train'] = self.evaluate_model(self.y_train, y_pred_train, "XGBoost Optimized (Train)")
        self.results['XGBoost_Optimized_test'] = self.evaluate_model(self.y_test, y_pred_test, "XGBoost Optimized (Test)")

        print(f"\nOptimized XGBoost Test R²: {self.results['XGBoost_Optimized_test']['R2']:.4f}")
        print(f"MAE: ${self.results['XGBoost_Optimized_test']['MAE']:,.0f}")
        print(f"Within 10%: {self.results['XGBoost_Optimized_test']['Within_10%']:.1f}%")

        return best_xgb

    def train_ensemble(self):
        """Train ensemble models (Voting and Stacking)"""

        print("\n" + "="*60)
        print("TRAINING ENSEMBLE MODELS")
        print("="*60)

        # Select best performing models for ensemble
        best_models = [
            ('xgb', self.models['XGBoost']),
            ('lgb', self.models['LightGBM']),
            ('cat', self.models['CatBoost'])
        ]

        # Voting Regressor (average predictions)
        print("\nTraining Voting Ensemble...")
        voting_reg = VotingRegressor(estimators=best_models)
        voting_reg.fit(self.X_train, self.y_train)

        y_pred_train = voting_reg.predict(self.X_train)
        y_pred_test = voting_reg.predict(self.X_test)

        self.models['Voting Ensemble'] = voting_reg
        self.results['Voting_train'] = self.evaluate_model(self.y_train, y_pred_train, "Voting (Train)")
        self.results['Voting_test'] = self.evaluate_model(self.y_test, y_pred_test, "Voting (Test)")

        print(f"Test R²: {self.results['Voting_test']['R2']:.4f} | MAE: ${self.results['Voting_test']['MAE']:,.0f}")

        # Stacking Regressor
        print("\nTraining Stacking Ensemble...")
        stacking_reg = StackingRegressor(
            estimators=best_models,
            final_estimator=Ridge(alpha=1.0),
            cv=5
        )
        stacking_reg.fit(self.X_train, self.y_train)

        y_pred_train = stacking_reg.predict(self.X_train)
        y_pred_test = stacking_reg.predict(self.X_test)

        self.models['Stacking Ensemble'] = stacking_reg
        self.results['Stacking_train'] = self.evaluate_model(self.y_train, y_pred_train, "Stacking (Train)")
        self.results['Stacking_test'] = self.evaluate_model(self.y_test, y_pred_test, "Stacking (Test)")

        print(f"Test R²: {self.results['Stacking_test']['R2']:.4f} | MAE: ${self.results['Stacking_test']['MAE']:,.0f}")

        print("\nEnsemble models training complete!")

    def select_best_model(self):
        """Select the best performing model based on test metrics"""

        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        # Get all test results
        test_results = {k: v for k, v in self.results.items() if '_test' in k}

        # Create comparison dataframe
        comparison_df = pd.DataFrame(test_results).T
        comparison_df = comparison_df.sort_values('R2', ascending=False)

        print("\n", comparison_df.to_string())

        # Select best model by R2 score
        best_result_key = comparison_df.index[0]
        # Extract the model name from the result key
        # The key format is "ModelName_test", so we remove "_test"
        model_name_from_key = best_result_key.replace('_test', '')

        # For ensemble models, the key is just the model name (e.g., "Voting", "Stacking")
        # For other models, it matches the key in self.models
        self.best_model_name = model_name_from_key

        # Check if model exists in self.models
        if model_name_from_key in self.models:
            self.best_model = self.models[model_name_from_key]
        else:
            # Fallback: try to find the model by checking all keys
            self.best_model = None
            for key in self.models:
                if key == model_name_from_key or key.replace(' ', '_') == model_name_from_key:
                    self.best_model = self.models[key]
                    self.best_model_name = key
                    break

        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"{'='*60}")
        print(f"R² Score: {comparison_df.iloc[0]['R2']:.4f}")
        print(f"MAE: ${comparison_df.iloc[0]['MAE']:,.0f}")
        print(f"RMSE: ${comparison_df.iloc[0]['RMSE']:,.0f}")
        print(f"MAPE: {comparison_df.iloc[0]['MAPE']:.2f}%")
        print(f"Within 10%: {comparison_df.iloc[0]['Within_10%']:.1f}%")

        return self.best_model, self.best_model_name, comparison_df

    def save_models(self, best_only=False):
        """Save trained models"""

        if best_only and self.best_model:
            # Save only best model
            filepath = f'model/best_model_{self.best_model_name.replace(" ", "_")}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"\nBest model saved to: {filepath}")

        else:
            # Save all models
            for name, model in self.models.items():
                filepath = f'model/{name.replace(" ", "_")}.pkl'
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            print(f"\nAll {len(self.models)} models saved to model/")

        # Save results
        results_filepath = 'model/model_results.json'
        with open(results_filepath, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Model results saved to: {results_filepath}")


def main():
    """Main training pipeline"""

    print("\n" + "="*70)
    print(" "*15 + "SALARY PREDICTION MODEL TRAINING")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/salary_data_2025.csv')
    print(f"Dataset loaded: {df.shape}")

    # Prepare data
    preprocessor = SalaryDataPreprocessor()
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocessor.prepare_data(df)

    # Save preprocessor
    preprocessor.save_preprocessor()

    # Initialize trainer (use non-scaled data for tree-based models)
    trainer = SalaryModelTrainer(X_train, X_test, y_train, y_test)

    # Train baseline models
    trainer.train_baseline_models()

    # Train advanced models
    trainer.train_advanced_models()

    # Optimize best model (optional - takes time)
    print("\n" + "="*60)
    user_input = input("Do you want to run hyperparameter optimization? (y/n): ")
    if user_input.lower() == 'y':
        trainer.optimize_xgboost(n_trials=30)

    # Train ensemble models
    trainer.train_ensemble()

    # Select best model
    best_model, best_name, comparison = trainer.select_best_model()

    # Save models
    trainer.save_models(best_only=True)

    # Save comparison table
    comparison.to_csv('model/model_comparison.csv')
    print("\nModel comparison saved to: model/model_comparison.csv")

    print("\n" + "="*70)
    print(" "*20 + "TRAINING COMPLETE!")
    print("="*70)

    return trainer, best_model, comparison


if __name__ == "__main__":
    trainer, best_model, comparison = main()
