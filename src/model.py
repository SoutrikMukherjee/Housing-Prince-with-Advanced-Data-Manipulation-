import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load


class AdvancedHousingModel:
    """Advanced model class for housing price prediction."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the model.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.feature_importances = None
        self.scaler = StandardScaler()
    
    def train_multiple_models(self, 
                             X_train: pd.DataFrame, 
                             y_train: pd.Series,
                             X_val: pd.DataFrame = None,
                             y_val: pd.Series = None,
                             cv: int = 5) -> Dict[str, Any]:
        """Train multiple models and evaluate their performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with model evaluation results
        """
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Define models to train
        model_params = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
                }
            },
            'Lasso': {
                'model': Lasso(random_state=self.random_state),
                'params': {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                    'max_iter': [10000]
                }
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=self.random_state),
                'params': {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_iter': [10000]
                }
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                    'subsample': [0.7, 0.8, 0.9]
                }
            }
        }
        
        # Dictionary to store results
        results = {
            'model_names': [],
            'train_rmse': [],
            'val_rmse': [],
            'cv_rmse': [],
            'r2_score': [],
            'mae': [],
            'best_params': []
        }
        
        # Loop through models
        best_val_rmse = float('inf')
        
        for model_name, model_info in model_params.items():
            print(f"Training {model_name}...")
            
            # Set up cross-validation
            kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            
            # Check if we need to perform hyperparameter tuning
            if model_info['params']:
                # Use RandomizedSearchCV for more efficient search
                model = RandomizedSearchCV(
                    estimator=model_info['model'],
                    param_distributions=model_info['params'],
                    n_iter=10,  # Number of parameter settings sampled
                    cv=kf,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    random_state=self.random_state
                )
                
                # Fit the model
                model.fit(X_train_scaled, y_train)
                
                # Get the best model and parameters
                best_model = model.best_estimator_
                best_params = model.best_params_
                
                # Get CV score
                cv_rmse = -model.best_score_
            else:
                # Fit the model directly
                model_info['model'].fit(X_train_scaled, y_train)
                best_model = model_info['model']
                best_params = {}
                
                # Calculate CV score manually
                cv_scores = cross_val_score(
                    best_model,
                    X_train_scaled,
                    y_train,
                    cv=kf,
                    scoring='neg_root_mean_squared_error'
                )
                cv_rmse = -np.mean(cv_scores)
            
            # Store the trained model
            self.models[model_name] = best_model
            
            # Make predictions on training set
            y_train_pred = best_model.predict(X_train_scaled)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            
            # Validation metrics
            val_rmse = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                y_val_pred = best_model.predict(X_val_scaled)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                
                # Update best model if this one is better
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    self.best_model = best_model
                    self.best_model_name = model_name
            
            # Store results
            results['model_names'].append(model_name)
            results['train_rmse'].append(train_rmse)
            results['val_rmse'].append(val_rmse)
            results['cv_rmse'].append(cv_rmse)
            results['r2_score'].append(train_r2)
            results['mae'].append(train_mae)
            results['best_params'].append(best_params)
            
            print(f"  Train RMSE: {train_rmse:.4f}")
            if val_rmse:
                print(f"  Validation RMSE: {val_rmse:.4f}")
            print(f"  CV RMSE: {cv_rmse:.4f}")
            print(f"  R² Score: {train_r2:.4f}")
            print(f"  MAE: {train_mae:.4f}")
            print(f"  Best params: {best_params}")
            print()
        
        # Calculate feature importances for the best model
        self._calculate_feature_importances(X_train)
        
        return results
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Make predictions using a trained model.
        
        Args:
            X: Feature DataFrame
            model_name: Name of the model to use (uses best model if None)
            
        Returns:
            Array of predictions
        """
        # Check if we have a trained model
        if not self.models and not self.best_model:
            raise ValueError("No trained models available. Call train_multiple_models() first.")
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Select the model to use
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            model = self.models[model_name]
        else:
            model = self.best_model if self.best_model is not None else self.models[list(self.models.keys())[0]]
        
        # Make predictions
        return model.predict(X_scaled)
    
    def evaluate_model(self, 
                      X: pd.DataFrame, 
                      y: pd.Series, 
                      model_name: str = None) -> Dict[str, float]:
        """Evaluate a trained model on a dataset.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_name: Name of the model to evaluate (uses best model if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X, model_name)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y - y_pred) / y)) * 100  # Mean Absolute Percentage Error
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Calculate metrics
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'Mean Residual': np.mean(residuals),
            'Residual Std': np.std(residuals)
        }
        
        return metrics
    
    def plot_residuals(self, 
                      X: pd.DataFrame, 
                      y: pd.Series, 
                      model_name: str = None,
                      figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot residuals analysis for a trained model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_name: Name of the model to use (uses best model if None)
            figsize: Figure size for the plots
        """
        # Make predictions
        y_pred = self.predict(X, model_name)
        
        # Calculate residuals
        residuals = y - y_pred
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title('Residuals vs Predicted Values')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Residuals')
        
        # Plot 2: Histogram of Residuals
        axes[0, 1].hist(residuals, bins=30, edgecolor='black')
        axes[0, 1].set_title('Histogram of Residuals')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Actual vs Predicted
        axes[1, 0].scatter(y, y_pred, alpha=0.5)
        
        # Add diagonal line (perfect predictions)
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        axes[1, 0].set_title('Actual vs Predicted Values')
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        
        # Plot 4: QQ Plot of Residuals
        from scipy import stats
        stats.probplot(residuals, plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str, model_name: str = None) -> None:
        """Save a trained model to disk.
        
        Args:
            filepath: Path to save the model
            model_name: Name of the model to save (saves best model if None)
        """
        # Select the model to save
        if model_name is not None:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found. Available models: {list(self.models.keys())}")
            model = self.models[model_name]
        else:
            model = self.best_model if self.best_model is not None else self.models[list(self.models.keys())[0]]
        
        # Create a dictionary with all necessary components
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'feature_importances': self.feature_importances,
            'best_model_name': getattr(self, 'best_model_name', None)
        }
        
        # Save the model data
        dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        # Load the model data
        model_data = load(filepath)
        
        # Extract components
        model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_importances = model_data['feature_importances']
        
        # Store as best model
        self.best_model = model
        
        # If available, use the best model name
        if 'best_model_name' in model_data and model_data['best_model_name'] is not None:
            self.best_model_name = model_data['best_model_name']
            # Also store in models dictionary
            self.models = {self.best_model_name: model}
        else:
            # Generate a generic name
            generic_name = model.__class__.__name__
            self.models = {generic_name: model}
            self.best_model_name = generic_name
        
        print(f"Model loaded from {filepath}")
    
    def _calculate_feature_importances(self, X: pd.DataFrame) -> None:
        """Calculate feature importances for the best model.
        
        Args:
            X: Feature DataFrame
        """
        if self.best_model is None:
            return
        
        # Check if the model has feature_importances_ attribute
        if hasattr(self.best_model, 'feature_importances_'):
            # Get feature importances
            importances = self.best_model.feature_importances_
            
            # Create a DataFrame of feature importances
            self.feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        # For linear models, use coefficients
        elif hasattr(self.best_model, 'coef_'):
            # Get coefficients
            coefficients = self.best_model.coef_
            
            # For multi-output models, take the mean of coefficients
            if coefficients.ndim > 1:
                coefficients = coefficients.mean(axis=0)
            
            # Create a DataFrame of feature importances (absolute coefficients)
            self.feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': np.abs(coefficients)
            }).sort_values('Importance', ascending=False).reset_index(drop=True)
        else:
            # For models without native feature importance
            self.feature_importances = None
    
    def plot_feature_importances(self, top_n: int = 20, figsize: Tuple[int, int] = (12, 8)) -> None:
        """Plot feature importances for the best model.
        
        Args:
            top_n: Number of top features to plot
            figsize: Figure size for the plot
        """
        if self.feature_importances is None:
            print("Feature importances not available for the current model.")
            return
        
        # Select top N features
        top_features = self.feature_importances.head(top_n)
        
        # Create a horizontal bar plot
        plt.figure(figsize=figsize)
        ax = sns.barplot(x='Importance', y='Feature', data=top_features)
        
        # Add labels and title
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    def ensemble_predictions(self, 
                            X: pd.DataFrame, 
                            weights: Dict[str, float] = None) -> np.ndarray:
        """Make ensemble predictions using multiple trained models.
        
        Args:
            X: Feature DataFrame
            weights: Dictionary mapping model names to weights (uses equal weights if None)
            
        Returns:
            Array of ensemble predictions
        """
        if not self.models:
            raise ValueError("No trained models available. Call train_multiple_models() first.")
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Get available model names
        model_names = list(self.models.keys())
        
        # If weights are not provided, use equal weights
        if weights is None:
            weights = {model_name: 1.0 / len(model_names) for model_name in model_names}
        else:
            # Validate weights
            missing_models = [model_name for model_name in weights.keys() if model_name not in model_names]
            if missing_models:
                raise ValueError(f"Models not found: {missing_models}. Available models: {model_names}")
            
            # Normalize weights
            weight_sum = sum(weights.values())
            weights = {model_name: weight / weight_sum for model_name, weight in weights.items()}
        
        # Make predictions for each model
        predictions = {}
        for model_name, model in self.models.items():
            if model_name in weights:
                predictions[model_name] = model.predict(X_scaled)
        
        # Compute weighted average of predictions
        ensemble_pred = np.zeros(X.shape[0])
        for model_name, pred in predictions.items():
            ensemble_pred += weights[model_name] * pred
        
        return ensemble_pred


class StochasticEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Custom ensemble regressor that combines multiple models using stochastic weights."""
    
    def __init__(self, base_models=None, random_state=42):
        """Initialize the ensemble regressor.
        
        Args:
            base_models: List of base models to ensemble
            random_state: Random seed for reproducibility
        """
        self.base_models = base_models if base_models is not None else []
        self.random_state = random_state
        self.models_ = []
        self.weights_ = None
    
    def fit(self, X, y):
        """Fit the ensemble model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Self
        """
        # Fit each base model
        self.models_ = []
        for base_model in self.base_models:
            model = base_model.fit(X, y)
            self.models_.append(model)
        
        # Calculate weights based on validation performance
        # For demonstration, we use cross-validation scores
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        scores = []
        
        for model in self.models_:
            cv_scores = cross_val_score(
                model, X, y, cv=kf, scoring='neg_root_mean_squared_error'
            )
            scores.append(-np.mean(cv_scores))  # Convert to positive RMSE
        
        # Inverse weights (lower RMSE → higher weight)
        self.weights_ = 1.0 / np.array(scores)
        # Normalize weights to sum to 1
        self.weights_ = self.weights_ / np.sum(self.weights_)
        
        return self
    
    def predict(self, X):
        """Make predictions with the ensemble model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.models_:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Get predictions from all models
        predictions = np.array([model.predict(X) for model in self.models_])
        
        # Apply weights and sum
        ensemble_pred = np.zeros(X.shape[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights_[i] * pred
        
        return ensemble_pred
    
    def get_model_weights(self):
        """Get the weights assigned to each model.
        
        Returns:
            Dictionary mapping model names to weights
        """
        if self.weights_ is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        return {
            f"Model_{i}_{type(model).__name__}": weight 
            for i, (model, weight) in enumerate(zip(self.models_, self.weights_))
        }
