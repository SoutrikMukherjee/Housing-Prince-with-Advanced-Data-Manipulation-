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
                    continue
                    
                for j, col2 in enumerate(self.numerical_features[i+1:], i+1):
                    if col2 not in X.columns:
                        continue
                        
                    # Multiplicative interaction
                    X[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                    
                    # Additive interaction
                    X[f"{col1}_plus_{col2}"] = X[col1] + X[col2]
                    
                    # Difference interaction
                    X[f"{col1}_minus_{col2}"] = X[col1] - X[col2]
                    
                    # Ratio interaction (with safeguard against division by zero)
                    X[f"{col1}_div_{col2}"] = np.where(X[col2] != 0, X[col1] / X[col2], np.nan)
                    
                    # Average
                    X[f"{col1}_{col2}_avg"] = (X[col1] + X[col2]) / 2
                    
                    # Geometric mean (for positive values)
                    if (X[col1] > 0).all() and (X[col2] > 0).all():
                        X[f"{col1}_{col2}_geom_mean"] = np.sqrt(X[col1] * X[col2])
        
        return X
    
    def _add_binning_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add binned versions of numerical features.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with added binning features
        """
        for col in self.numerical_features:
            if col not in X.columns:
                continue
                
            # Equal-width binning (10 bins)
            X[f"{col}_bin10"] = pd.cut(X[col], bins=10, labels=False)
            
            # Equal-frequency binning (quartiles)
            X[f"{col}_quartile"] = pd.qcut(X[col], q=4, labels=False, duplicates='drop')
            
            # Custom binning based on domain knowledge
            # Example: Bin house age into categories
            if col == 'HouseAge':
                bins = [0, 5, 10, 20, 40, 100]  # Custom bin edges
                labels = [0, 1, 2, 3, 4]  # Custom bin labels
                X[f"{col}_custom_bin"] = pd.cut(X[col], bins=bins, labels=labels, include_lowest=True)
                
            # Z-score based binning
            z_scores = (X[col] - self.feature_means[col]) / self.feature_stds[col]
            X[f"{col}_zscore_bin"] = pd.cut(z_scores, bins=[-np.inf, -2, -1, 0, 1, 2, np.inf], labels=False)
            
            # IQR-based binning
            iqr = self.feature_q3[col] - self.feature_q1[col]
            lower_bound = self.feature_q1[col] - 1.5 * iqr
            upper_bound = self.feature_q3[col] + 1.5 * iqr
            X[f"{col}_iqr_bin"] = pd.cut(
                X[col], 
                bins=[
                    -np.inf,
                    lower_bound,
                    self.feature_q1[col],
                    self.feature_median[col],
                    self.feature_q3[col],
                    upper_bound,
                    np.inf
                ],
                labels=False
            )
        
        return X
    
    def _add_cyclical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical features for coordinates (latitude, longitude).
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with added cyclical features
        """
        # Check if we have coordinates in the dataset
        coord_features = [col for col in self.numerical_features if col.lower() in [
            'latitude', 'longitude', 'lat', 'lon', 'x_coord', 'y_coord'
        ]]
        
        for col in coord_features:
            if col not in X.columns:
                continue
                
            # Normalize to [0, 2π]
            min_val = self.feature_min[col]
            max_val = self.feature_max[col]
            range_val = max_val - min_val
            
            if range_val > 0:
                # Scale to [0, 2π]
                normalized = 2 * np.pi * (X[col] - min_val) / range_val
                
                # Create sine and cosine features
                X[f"{col}_sin"] = np.sin(normalized)
                X[f"{col}_cos"] = np.cos(normalized)
                
                # Create tangent feature
                X[f"{col}_tan"] = np.tan(normalized)
        
        # Special handling for latitude and longitude pairs
        if 'Latitude' in X.columns and 'Longitude' in X.columns:
            # Calculate Haversine distance to city center or reference point
            # (assuming reference point might be median values)
            ref_lat = self.feature_medians.get('Latitude', X['Latitude'].median())
            ref_lon = self.feature_medians.get('Longitude', X['Longitude'].median())
            
            # Convert to radians
            lat_rad = np.radians(X['Latitude'])
            lon_rad = np.radians(X['Longitude'])
            ref_lat_rad = np.radians(ref_lat)
            ref_lon_rad = np.radians(ref_lon)
            
            # Haversine formula components
            dlon = lon_rad - ref_lon_rad
            dlat = lat_rad - ref_lat_rad
            a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(ref_lat_rad) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            # Earth radius in kilometers
            R = 6371.0
            
            # Distance in kilometers
            X['distance_to_ref'] = R * c
            
            # Direction (bearing) to reference point
            y = np.sin(dlon) * np.cos(lat_rad)
            x = np.cos(ref_lat_rad) * np.sin(lat_rad) - np.sin(ref_lat_rad) * np.cos(lat_rad) * np.cos(dlon)
            theta = np.arctan2(y, x)
            bearing = (np.degrees(theta) + 360) % 360
            X['bearing_to_ref'] = bearing
            
            # Create sine and cosine of bearing for cyclical representation
            X['bearing_sin'] = np.sin(np.radians(bearing))
            X['bearing_cos'] = np.cos(np.radians(bearing))
        
        return X
    
    def _add_outlier_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add indicator features for outliers.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with added outlier features
        """
        for col in self.numerical_features:
            if col not in X.columns:
                continue
                
            # Z-score based outliers
            z_scores = np.abs((X[col] - self.feature_means[col]) / self.feature_stds[col])
            X[f"{col}_is_outlier_z3"] = (z_scores > 3).astype(int)  # Beyond 3 standard deviations
            X[f"{col}_is_outlier_z2"] = (z_scores > 2).astype(int)  # Beyond 2 standard deviations
            
            # IQR-based outliers
            iqr = self.feature_q3[col] - self.feature_q1[col]
            lower_bound = self.feature_q1[col] - 1.5 * iqr
            upper_bound = self.feature_q3[col] + 1.5 * iqr
            X[f"{col}_is_outlier_iqr"] = (
                (X[col] < lower_bound) | (X[col] > upper_bound)
            ).astype(int)
            
            # Distance from median
            X[f"{col}_dist_from_median"] = np.abs(X[col] - self.feature_medians[col])
            
            # Percentile-based feature
            percentiles = pd.qcut(X[col], q=100, labels=False, duplicates='drop')
            X[f"{col}_percentile"] = percentiles
        
        return X
