import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Advanced feature engineering class that implements the scikit-learn transformer interface."""
    
    def __init__(self, 
                polynomial_degree: int = 2,
                interaction_features: bool = True,
                binning_features: bool = False,
                cyclical_features: bool = True,
                outlier_features: bool = True,
                transformation_features: bool = True):
        """Initialize the feature engineer.
        
        Args:
            polynomial_degree: Degree of polynomial features
            interaction_features: Whether to create interaction features
            binning_features: Whether to create binned features
            cyclical_features: Whether to create cyclical features for coordinates
            outlier_features: Whether to create outlier indicator features
            transformation_features: Whether to create transformed features (log, sqrt, etc.)
        """
        self.polynomial_degree = polynomial_degree
        self.interaction_features = interaction_features
        self.binning_features = binning_features
        self.cyclical_features = cyclical_features
        self.outlier_features = outlier_features
        self.transformation_features = transformation_features
        
        # These will be set during fit
        self.numerical_features = None
        self.feature_medians = {}
        self.feature_means = {}
        self.feature_stds = {}
        self.feature_min = {}
        self.feature_max = {}
        self.feature_q1 = {}
        self.feature_q3 = {}
    
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the feature engineer.
        
        Args:
            X: Feature DataFrame
            y: Target (not used)
            
        Returns:
            Self
        """
        # Store list of numerical features
        self.numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        
        # Calculate and store statistics for each numerical feature
        for col in self.numerical_features:
            self.feature_medians[col] = X[col].median()
            self.feature_means[col] = X[col].mean()
            self.feature_stds[col] = X[col].std()
            self.feature_min[col] = X[col].min()
            self.feature_max[col] = X[col].max()
            self.feature_q1[col] = X[col].quantile(0.25)
            self.feature_q3[col] = X[col].quantile(0.75)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame by adding engineered features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed DataFrame with additional features
        """
        # Create a copy to avoid modifying the original
        X_transformed = X.copy()
        
        # Apply different types of feature engineering based on init parameters
        if self.transformation_features:
            X_transformed = self._add_transformation_features(X_transformed)
            
        if self.interaction_features:
            X_transformed = self._add_interaction_features(X_transformed)
            
        if self.binning_features:
            X_transformed = self._add_binning_features(X_transformed)
            
        if self.cyclical_features:
            X_transformed = self._add_cyclical_features(X_transformed)
            
        if self.outlier_features:
            X_transformed = self._add_outlier_features(X_transformed)
        
        return X_transformed
    
    def _add_transformation_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add transformed versions of numerical features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with added transformation features
        """
        for col in self.numerical_features:
            if col not in X.columns:
                continue
                
            # Apply logarithmic transformation (with offset for non-positive values)
            min_val = self.feature_min[col]
            if min_val <= 0:
                offset = abs(min_val) + 1
                X[f"{col}_log"] = np.log(X[col] + offset)
            else:
                X[f"{col}_log"] = np.log(X[col])
                
            # Square root transformation (with offset for negative values)
            if min_val < 0:
                offset = abs(min_val) + 1
                X[f"{col}_sqrt"] = np.sqrt(X[col] + offset)
            else:
                X[f"{col}_sqrt"] = np.sqrt(X[col])
                
            # Square transformation
            X[f"{col}_squared"] = np.square(X[col])
            
            # Cube transformation
            X[f"{col}_cubed"] = np.power(X[col], 3)
            
            # Inverse transformation (with safeguard against division by zero)
            X[f"{col}_inverse"] = np.where(X[col] != 0, 1 / X[col], np.nan)
            
            # Z-score (standardized) feature
            X[f"{col}_zscore"] = (X[col] - self.feature_means[col]) / self.feature_stds[col]
            
            # Min-max scaled feature
            range_val = self.feature_max[col] - self.feature_min[col]
            if range_val > 0:
                X[f"{col}_minmax"] = (X[col] - self.feature_min[col]) / range_val
                
            # Sigmoid transformation
            X[f"{col}_sigmoid"] = 1 / (1 + np.exp(-X[col]))
            
            # Yeo-Johnson transformation (handles negative values better than Box-Cox)
            X[f"{col}_yeojohnson"], _ = stats.yeojohnson(X[col])
        
        return X
    
    def _add_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between pairs of numerical features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with added interaction features
        """
        # Use PolynomialFeatures for interaction terms
        if len(self.numerical_features) >= 2:
            # Create multiplicative interactions
            for i, col1 in enumerate(self.numerical_features):
                if col1 not in X.columns:
