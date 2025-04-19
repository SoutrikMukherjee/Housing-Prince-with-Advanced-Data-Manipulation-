import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

class DataProcessor:
    """Advanced data processing class for the housing dataset."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the data processor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.numerical_features = None
        self.categorical_features = None
        self.preprocessor = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file with optimized settings.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame containing the loaded data
        """
        # Using optimized Pandas read_csv parameters
        df = pd.read_csv(
            filepath,
            # Optimize memory usage by specifying dtypes
            dtype={
                'MedInc': np.float32,
                'HouseAge': np.float32,
                'AveRooms': np.float32,
                'AveBedrms': np.float32,
                'Population': np.float32,
                'AveOccup': np.float32,
                'Latitude': np.float32,
                'Longitude': np.float32,
            },
            # Use efficient parsing for dates if present
            parse_dates=False,
            # Low memory mode for large files
            low_memory=True
        )
        
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    
    def detect_feature_types(self, df: pd.DataFrame, threshold: int = 10) -> Tuple[List[str], List[str]]:
        """Automatically detect numerical and categorical features.
        
        Args:
            df: Input DataFrame
            threshold: Maximum number of unique values for a column to be considered categorical
            
        Returns:
            Tuple of (numerical_features, categorical_features)
        """
        # Intelligent feature type detection using Pandas APIs
        numerical_features = []
        categorical_features = []
        
        for col in df.columns:
            # Skip target column if present
            if col == 'target' or col == 'median_house_value':
                continue
                
            # Check if column has numeric dtype
            if np.issubdtype(df[col].dtype, np.number):
                # Check if the number of unique values suggests a categorical variable
                if df[col].nunique() < threshold:
                    categorical_features.append(col)
                else:
                    numerical_features.append(col)
            else:
                categorical_features.append(col)
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        print(f"Detected {len(numerical_features)} numerical features: {numerical_features}")
        print(f"Detected {len(categorical_features)} categorical features: {categorical_features}")
        
        return numerical_features, categorical_features
    
    def analyze_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing value statistics
        """
        # Advanced missing value analysis with Pandas
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        
        # Create summary DataFrame with statistics
        missing_info = pd.DataFrame({
            'Missing Values': missing_values,
            'Missing Percentage': missing_percentage,
            'Data Type': df.dtypes
        }).sort_values('Missing Percentage', ascending=False)
        
        # Add correlation of missingness with target if available
        if 'target' in df.columns or 'median_house_value' in df.columns:
            target_col = 'target' if 'target' in df.columns else 'median_house_value'
            
            # Advanced technique: Correlation of missingness with target
            missingness_correlation = {}
            for col in df.columns:
                if missing_values[col] > 0:
                    missingness = df[col].isnull().astype(int)
                    missingness_correlation[col] = np.corrcoef(missingness, df[target_col])[0, 1]
            
            missing_info['Correlation with Target'] = missing_info.index.map(
                lambda x: missingness_correlation.get(x, np.nan)
            )
        
        return missing_info
    
    def create_preprocessor(self, 
                           impute_strategy: str = 'median',
                           categorical_strategy: str = 'onehot',
                           handle_outliers: bool = True,
                           knn_impute: bool = False) -> ColumnTransformer:
        """Create a preprocessing pipeline for the data.
        
        Args:
            impute_strategy: Strategy for imputing missing values ('mean', 'median', 'most_frequent')
            categorical_strategy: Strategy for handling categorical features ('onehot', 'ordinal')
            handle_outliers: Whether to handle outliers using winsorization
            knn_impute: Whether to use KNN imputation for numerical features
            
        Returns:
            ColumnTransformer preprocessing pipeline
        """
        if self.numerical_features is None or self.categorical_features is None:
            raise ValueError("Feature types not detected. Call detect_feature_types() first.")
        
        # Advanced numerical preprocessing pipeline with NumPy operations
        if knn_impute:
            num_pipeline = Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler())
            ])
        else:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy=impute_strategy)),
                ('scaler', StandardScaler())
            ])
        
        # Categorical preprocessing pipeline
        if categorical_strategy == 'onehot':
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        else:
            # For ordinal encoding, we would implement a custom ordinal encoder
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                # Implement custom ordinal encoder if needed
            ])
        
        # Combine the pipelines with ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.numerical_features),
                ('cat', cat_pipeline, self.categorical_features)
            ],
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def create_train_val_test_split(self, 
                                   df: pd.DataFrame, 
                                   target_col: str = 'median_house_value',
                                   test_size: float = 0.2,
                                   val_size: float = 0.25) -> Tuple:
        """Split the data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame
            target_col: Name of the target column
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Extract features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # First split: training + validation vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Second split: training vs validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=self.random_state
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def handle_outliers(self, 
                       df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = 'winsorize',
                       threshold: float = 3.0) -> pd.DataFrame:
        """Handle outliers in the specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check for outliers (defaults to numerical_features)
            method: Method to handle outliers ('winsorize', 'clip', 'remove')
            threshold: Z-score threshold for identifying outliers
            
        Returns:
            DataFrame with outliers handled
        """
        if columns is None:
            if self.numerical_features is None:
                raise ValueError("Feature types not detected. Call detect_feature_types() first.")
            columns = self.numerical_features
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_clean = df.copy()
        
        outlier_stats = {}
        
        for col in columns:
            # Skip columns with non-numeric dtype
            if not np.issubdtype(df[col].dtype, np.number):
                continue
                
            # Calculate z-scores using NumPy vectorized operations
            mean = df[col].mean()
            std = df[col].std()
            z_scores = np.abs((df[col] - mean) / std)
            
            # Identify outliers
            outliers = (z_scores > threshold)
            outlier_count = outliers.sum()
            
            outlier_stats[col] = {
                'outlier_count': outlier_count,
                'outlier_percentage': (outlier_count / len(df)) * 100
            }
            
            # Handle outliers based on the chosen method
            if method == 'winsorize':
                # Winsorization: Cap outliers at threshold
                upper_bound = mean + threshold * std
                lower_bound = mean - threshold * std
                df_clean[col] = df[col].clip(lower_bound, upper_bound)
                
            elif method == 'clip':
                # Clip at percentiles
                lower_bound = df[col].quantile(0.01)
                upper_bound = df[col].quantile(0.99)
                df_clean[col] = df[col].clip(lower_bound, upper_bound)
                
            elif method == 'remove':
                # Only for educational purposes - in practice, we would avoid removing rows
                df_clean = df_clean[~outliers]
        
        # Create a summary of outlier handling
        outlier_summary = pd.DataFrame.from_dict(outlier_stats, orient='index')
        print("Outlier handling summary:")
        print(outlier_summary)
        
        return df_clean
    
    def create_interaction_features(self, 
                                  df: pd.DataFrame, 
                                  interaction_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """Create interaction features from pairs of numerical features.
        
        Args:
            df: Input DataFrame
            interaction_pairs: List of (feature1, feature2) tuples to create interactions for
            
        Returns:
            DataFrame with added interaction features
        """
        if self.numerical_features is None:
            raise ValueError("Feature types not detected. Call detect_feature_types() first.")
            
        # If no specific pairs are provided, create all possible pairs from numerical features
        if interaction_pairs is None:
            if len(self.numerical_features) >= 2:
                import itertools
                interaction_pairs = list(itertools.combinations(self.numerical_features, 2))
            else:
                return df  # Not enough numerical features for interactions
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_with_interactions = df.copy()
        
        for feature1, feature2 in interaction_pairs:
            # Skip if either feature is not in the DataFrame
            if feature1 not in df.columns or feature2 not in df.columns:
                continue
                
            # Create multiplicative interaction
            interaction_name = f"{feature1}_x_{feature2}"
            df_with_interactions[interaction_name] = df[feature1] * df[feature2]
            
            # Create ratio interaction (with safeguards against division by zero)
            ratio_name = f"{feature1}_div_{feature2}"
            df_with_interactions[ratio_name] = np.where(
                df[feature2] != 0,
                df[feature1] / df[feature2],
                np.nan
            )
            
            # Create sum interaction
            sum_name = f"{feature1}_plus_{feature2}"
            df_with_interactions[sum_name] = df[feature1] + df[feature2]
            
            # Create difference interaction
            diff_name = f"{feature1}_minus_{feature2}"
            df_with_interactions[diff_name] = df[feature1] - df[feature2]
        
        return df_with_interactions
    
    def create_polynomial_features(self, 
                                  df: pd.DataFrame, 
                                  features: Optional[List[str]] = None,
                                  degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for the specified numerical features.
        
        Args:
            df: Input DataFrame
            features: List of features to create polynomials for (defaults to numerical_features)
            degree: Degree of the polynomial features
            
        Returns:
            DataFrame with added polynomial features
        """
        if features is None:
            if self.numerical_features is None:
                raise ValueError("Feature types not detected. Call detect_feature_types() first.")
            features = self.numerical_features
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_poly = df.copy()
        
        for feature in features:
            # Skip if feature is not in the DataFrame
            if feature not in df.columns:
                continue
                
            # Create polynomial features of specified degree
            for d in range(2, degree + 1):
                poly_name = f"{feature}_pow{d}"
                df_poly[poly_name] = np.power(df[feature], d)
            
            # Create log feature (with safeguards)
            log_name = f"{feature}_log"
            # Ensure values are positive before taking log
            min_val = df[feature].min()
            if min_val <= 0:
                offset = abs(min_val) + 1  # Add offset to make all values positive
                df_poly[log_name] = np.log(df[feature] + offset)
            else:
                df_poly[log_name] = np.log(df[feature])
            
            # Create sqrt feature (with safeguards)
            sqrt_name = f"{feature}_sqrt"
            # Ensure values are positive before taking sqrt
            if min_val < 0:
                offset = abs(min_val) + 1  # Add offset to make all values positive
                df_poly[sqrt_name] = np.sqrt(df[feature] + offset)
            else:
                df_poly[sqrt_name] = np.sqrt(df[feature])
        
        return df_poly
    
    def create_group_statistics(self, 
                               df: pd.DataFrame,
                               group_col: str,
                               agg_cols: List[str],
                               statistics: List[str] = ['mean', 'median', 'std', 'min', 'max']) -> pd.DataFrame:
        """Create aggregate statistics grouped by a categorical column.
        
        Args:
            df: Input DataFrame
            group_col: Column to group by
            agg_cols: Columns to aggregate
            statistics: List of statistics to compute
            
        Returns:
            DataFrame with added group statistics
        """
        # Validate input columns
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in DataFrame")
            
        valid_agg_cols = [col for col in agg_cols if col in df.columns]
        if not valid_agg_cols:
            raise ValueError(f"None of the aggregate columns {agg_cols} found in DataFrame")
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_grouped = df.copy()
        
        # Dictionary to map statistic names to functions
        stat_funcs = {
            'mean': np.mean,
            'median': np.median,
            'std': lambda x: np.std(x) if len(x) > 1 else 0,
            'min': np.min,
            'max': np.max,
            'range': lambda x: np.max(x) - np.min(x) if len(x) > 0 else 0,
            'q25': lambda x: np.percentile(x, 25),
            'q75': lambda x: np.percentile(x, 75),
            'iqr': lambda x: np.percentile(x, 75) - np.percentile(x, 25),
            'count': len,
            'sum': np.sum
        }
        
        # Filter statistics based on available functions
        valid_stats = [s for s in statistics if s in stat_funcs]
        
        # Compute group statistics efficiently using Pandas groupby
        for col in valid_agg_cols:
            for stat in valid_stats:
                # Create a dictionary for aggregation with selected statistic
                agg_dict = {col: stat_funcs[stat]}
                
                # Compute the aggregate statistic
                grouped_stat = df_grouped.groupby(group_col)[col].transform(stat_funcs[stat])
                
                # Add as a new feature
                stat_name = f"{group_col}_{col}_{stat}"
                df_grouped[stat_name] = grouped_stat
                
                # Compute difference from group statistic (useful feature)
                diff_name = f"{col}_diff_from_{group_col}_{stat}"
                df_grouped[diff_name] = df_grouped[col] - grouped_stat
                
                # Compute ratio to group statistic
                ratio_name = f"{col}_ratio_to_{group_col}_{stat}"
                # Prevent division by zero
                df_grouped[ratio_name] = np.where(
                    grouped_stat != 0,
                    df_grouped[col] / grouped_stat,
                    np.nan
                )
        
        return df_grouped

    def efficient_feature_importance(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series,
                                    method: str = 'f_regression') -> pd.DataFrame:
        """Calculate feature importance scores for numerical features.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Method to calculate feature importance ('f_regression', 'mutual_info')
            
        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.feature_selection import f_regression, mutual_info_regression
        
        # Select only numerical columns
        num_cols = X.select_dtypes(include=np.number).columns.tolist()
        X_num = X[num_cols]
        
        # Calculate feature importance based on the chosen method
        if method == 'f_regression':
            # F-test for regression
            f_values, p_values = f_regression(X_num, y)
            importance_values = f_values
            importance_type = 'F-Value'
            p_values_present = True
        elif method == 'mutual_info':
            # Mutual information regression
            importance_values = mutual_info_regression(X_num, y)
            importance_type = 'Mutual Information'
            p_values_present = False
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create a DataFrame with importance scores
        feature_importance = pd.DataFrame({
            'Feature': num_cols,
            importance_type: importance_values
        })
        
        # Add p-values if available
        if p_values_present:
            feature_importance['P-Value'] = p_values
        
        # Sort by importance score in descending order
        feature_importance = feature_importance.sort_values(
            importance_type, ascending=False
        ).reset_index(drop=True)
        
        return feature_importance
