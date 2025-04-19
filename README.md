# Housing Price Prediction: Advanced Data Analysis & ML Project

A comprehensive machine learning project showcasing advanced data manipulation, exploratory data analysis, and machine learning techniques for housing price prediction.

## Project Overview

This project demonstrates professional-level implementation of the entire machine learning workflow, with a special focus on advanced Pandas and NumPy operations for data manipulation and exploratory data analysis. It involves predicting housing prices using the California Housing Dataset.

## Key Features

- **Advanced Pandas & NumPy Operations**: Demonstrates complex data manipulation techniques
- **Comprehensive Data Analysis**: In-depth EDA with statistical analysis and visualizations
- **Feature Engineering Pipeline**: Creates useful features from raw data
- **Model Development**: Implements and compares multiple ML algorithms
- **Performance Evaluation**: Rigorous model evaluation across different metrics
- **Ensemble Methods**: Combines multiple models for improved performance

## Project Structure

```
housing-price-prediction/
│
├── data/
│   ├── housing.csv                  # Dataset (will be downloaded)
│   └── processed/                   # Preprocessed data
│       ├── train_data.csv           # Training data
│       ├── val_data.csv             # Validation data
│       ├── test_data.csv            # Test data
│       └── selected_features.txt    # List of selected features
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA and visualizations
│   ├── 02_data_preprocessing.ipynb  # Advanced preprocessing
│   └── 03_model_training.ipynb      # Model implementation and evaluation
│
├── src/
│   ├── __init__.py
│   ├── data_processor.py            # Data processing utilities
│   ├── feature_engineering.py       # Feature creation and transformation
│   ├── model.py                     # ML model implementation
│   └── visualization.py             # Custom visualization functions
│
├── models/
│   ├── best_single_model.joblib     # Saved best single model
│   └── ensemble_model.joblib        # Saved ensemble model
│
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

## Technical Highlights

### Advanced Data Processing

- **Custom DataProcessor Class**: Implements efficient data loading, cleaning, and transformation
- **Automated Feature Type Detection**: Intelligently identifies numerical and categorical features
- **Comprehensive Missing Value Analysis**: Advanced missing value analysis with correlation to target
- **Advanced Outlier Detection & Handling**: Multiple methods for identifying and handling outliers
- **Feature Creation Strategies**: Generates interaction, polynomial, and domain-specific features

### Feature Engineering

- **AdvancedFeatureEngineer Class**: Implements scikit-learn transformer interface
- **Transformation Features**: Logarithmic, square root, power transformations
- **Interaction Features**: Pairwise multiplicative, additive, and ratio interactions
- **Binning Strategies**: Equal-width, equal-frequency, and custom binning
- **Cyclical Features**: Sine/cosine encoding for cyclical variables
- **Outlier Features**: Z-score and IQR-based outlier indicators

### Visualization Capabilities

- **AdvancedVisualizer Class**: Comprehensive suite of visualization techniques
- **Distribution Analysis**: Histograms, KDE plots, and boxplots with statistics
- **Correlation Analysis**: Advanced correlation matrices and feature-target correlations
- **Geographic Analysis**: Interactive scatter plots and heatmaps for spatial data
- **Dimensionality Reduction Visualization**: PCA and t-SNE projections
- **Model Evaluation Plots**: Residual analysis, feature importance, and learning curves

### Model Implementation

- **AdvancedHousingModel Class**: Trains and evaluates multiple model types
- **Model Comparison Framework**: Comprehensive metrics for model selection
- **Automated Hyperparameter Tuning**: RandomizedSearchCV for efficient parameter optimization
- **Custom StochasticEnsembleRegressor**: Weights models based on cross-validation performance
- **Market Segment Analysis**: Evaluates model performance across different housing markets

## Installation & Usage

1. Clone this repository:
```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

3. Run the notebooks in sequence:
```bash
jupyter notebook notebooks/
```

## Results

The final model achieved:
- **RMSE**: ~$45,000 on test data
- **R²**: ~0.83 on test data
- **MAE**: ~$32,000 on test data

Performance analysis showed that:
- Medium-priced homes were predicted most accurately
- Location (latitude/longitude) and median income were the strongest predictors
- The ensemble model outperformed individual models by ~5% in RMSE

## Future Improvements

- Incorporate external data sources (schools, crime rates, amenities)
- Implement neural network models for comparison
- Add time-series analysis for temporal price trends
- Deploy model as a web application

## License

MIT

## Author

Your Name - [your.email@example.com](mailto:soutrik.viratech@gmail.com)
