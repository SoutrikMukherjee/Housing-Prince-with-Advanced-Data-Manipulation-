import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class AdvancedVisualizer:
    """Advanced data visualization class for exploratory data analysis."""
    
    def __init__(self, style: str = 'whitegrid', context: str = 'notebook', 
                 palette: str = 'viridis', figsize: Tuple[int, int] = (12, 8)):
        """Initialize the visualizer.
        
        Args:
            style: Seaborn style
            context: Seaborn context
            palette: Color palette
            figsize: Default figure size
        """
        # Set visualization style
        sns.set_style(style)
        sns.set_context(context)
        
        self.palette = palette
        self.figsize = figsize
    
    def plot_correlation_matrix(self, 
                               df: pd.DataFrame, 
                               method: str = 'pearson',
                               annot: bool = True,
                               cmap: str = 'coolwarm',
                               figsize: Optional[Tuple[int, int]] = None) -> None:
        """Plot correlation matrix heatmap.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            annot: Whether to annotate cells
            cmap: Colormap
            figsize: Figure size
        """
        # Calculate correlation matrix
        corr = df.corr(method=method)
        
        # Set up figure size
        if figsize is None:
            # Adjust figure size based on number of features
            n_features = len(corr)
            figsize = (max(8, n_features * 0.5), max(6, n_features * 0.5))
        
        # Create heatmap
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Mask upper triangle
        
        sns.heatmap(
            corr, 
            mask=mask,
            annot=annot, 
            cmap=cmap, 
            vmin=-1, 
            vmax=1, 
            center=0,
            square=True, 
            linewidths=.5, 
            fmt='.2f'
        )
        
        plt.title(f'{method.capitalize()} Correlation Matrix', fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_distributions(self, 
                                  df: pd.DataFrame, 
                                  features: Optional[List[str]] = None,
                                  n_cols: int = 3,
                                  figsize: Optional[Tuple[int, int]] = None,
                                  kde: bool = True) -> None:
        """Plot histograms of feature distributions.
        
        Args:
            df: Input DataFrame
            features: List of features to plot (uses all numeric features if None)
            n_cols: Number of columns in the subplot grid
            figsize: Figure size
            kde: Whether to overlay KDE plot
        """
        # Select features to plot
        if features is None:
            features = df.select_dtypes(include=np.number).columns.tolist()
        else:
            # Filter out non-existent features
            features = [f for f in features if f in df.columns]
        
        if not features:
            print("No numeric features to plot.")
            return
        
        # Calculate grid dimensions
        n_features = len(features)
        n_rows = int(np.ceil(n_features / n_cols))
        
        # Set up figure size
        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Plot histograms
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                sns.histplot(data=df, x=feature, kde=kde, ax=ax, color=sns.color_palette(self.palette)[i % 10])
                ax.set_title(f'Distribution of {feature}')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
                
                # Add summary statistics
                mean = df[feature].mean()
                median = df[feature].median()
                mode = df[feature].mode().iloc[0] if not df[feature].mode().empty else np.nan
                
                stats_text = f'Mean: {mean:.2f}\nMedian: {median:.2f}\nMode: {mode:.2f}'
                ax.text(
                    0.95, 0.95, stats_text, 
                    transform=ax.transAxes, 
                    verticalalignment='top', 
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_scatter_matrix(self, 
                           df: pd.DataFrame, 
                           features: Optional[List[str]] = None,
                           target: Optional[str] = None,
                           samples: int = 1000,
                           figsize: Optional[Tuple[int, int]] = None) -> None:
        """Plot scatter matrix of features.
        
        Args:
            df: Input DataFrame
            features: List of features to plot (uses up to 5 numeric features if None)
            target: Target feature for color-coding points
            samples: Number of samples to use (for large datasets)
            figsize: Figure size
        """
        # Select features to plot
        if features is None:
            # Select up to 5 numeric features
            features = df.select_dtypes(include=np.number).columns[:5].tolist()
        else:
            # Filter out non-existent features
            features = [f for f in features if f in df.columns]
        
        if target is not None and target in df.columns:
            features = [f for f in features if f != target]
            plot_df = df[features + [target]]
            hue = target
        else:
            plot_df = df[features]
            hue = None
        
        # Sample if dataset is large
        if len(plot_df) > samples:
            plot_df = plot_df.sample(samples, random_state=42)
        
        # Set up figure size
        if figsize is None:
            n_features = len(features)
            figsize = (n_features * 3, n_features * 3)
        
        # Create scatter matrix
        sns.set(style="ticks")
        scatter_matrix = sns.pairplot(
            plot_df, 
            hue=hue, 
            diag_kind='kde', 
            plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5},
            palette=self.palette if hue is not None else None
        )
        
        # Set title
        plt.suptitle('Scatter Matrix of Features', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_boxplots(self, 
                     df: pd.DataFrame, 
                     features: Optional[List[str]] = None,
                     n_cols: int = 3,
                     figsize: Optional[Tuple[int, int]] = None,
                     show_outliers: bool = True) -> None:
        """Plot boxplots of feature distributions.
        
        Args:
            df: Input DataFrame
            features: List of features to plot (uses all numeric features if None)
            n_cols: Number of columns in the subplot grid
            figsize: Figure size
            show_outliers: Whether to show outliers in the boxplots
        """
        # Select features to plot
        if features is None:
            features = df.select_dtypes(include=np.number).columns.tolist()
        else:
            # Filter out non-existent features
            features = [f for f in features if f in df.columns]
        
        if not features:
            print("No numeric features to plot.")
            return
        
        # Calculate grid dimensions
        n_features = len(features)
        n_rows = int(np.ceil(n_features / n_cols))
        
        # Set up figure size
        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        # Plot boxplots
        for i, feature in enumerate(features):
            if i < len(axes):
                ax = axes[i]
                sns.boxplot(
                    x=df[feature], 
                    ax=ax, 
                    color=sns.color_palette(self.palette)[i % 10],
                    showfliers=show_outliers
                )
                
                # Add stripplot to show individual data points
                if show_outliers:
                    sns.stripplot(
                        x=df[feature], 
                        ax=ax, 
                        color='black', 
                        size=3, 
                        alpha=0.3,
                        jitter=True
                    )
                
                ax.set_title(f'Boxplot of {feature}')
                ax.set_xlabel(feature)
                
                # Add summary statistics
                q1 = df[feature].quantile(0.25)
                median = df[feature].median()
                q3 = df[feature].quantile(0.75)
                iqr = q3 - q1
                
                stats_text = f'Q1: {q1:.2f}\nMedian: {median:.2f}\nQ3: {q3:.2f}\nIQR: {iqr:.2f}'
                ax.text(
                    0.95, 0.95, stats_text, 
                    transform=ax.transAxes, 
                    verticalalignment='top', 
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_target_correlations(self, 
                               df: pd.DataFrame, 
                               target: str,
                               top_n: int = 10,
                               method: str = 'pearson',
                               figsize: Optional[Tuple[int, int]] = None) -> None:
        """Plot features with highest correlation to target.
        
        Args:
            df: Input DataFrame
            target: Target feature
            top_n: Number of top correlated features to show
            method: Correlation method ('pearson', 'spearman', 'kendall')
            figsize: Figure size
        """
        if target not in df.columns:
            print(f"Target feature '{target}' not found in DataFrame.")
            return
        
        # Calculate correlations with target
        corr = df.corr(method=method)[target].drop(target)
        
        # Sort by absolute correlation and get top N
        top_corr = corr.abs().sort_values(ascending=False).head(top_n)
        top_features = top_corr.index.tolist()
        
        # Set up figure size
        if figsize is None:
            figsize = (10, 6)
        
        # Create bar plot
        plt.figure(figsize=figsize)
        ax = sns.barplot(
            x=corr[top_features].values, 
            y=top_features,
            palette=[
                sns.color_palette('RdBu_r', n_colors=100)[int((x + 1) * 49)]
                for x in corr[top_features].values
            ]
        )
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(corr[top_features].values):
            ax.text(
                v + (0.01 if v >= 0 else -0.01), 
                i, 
                f'{v:.3f}', 
                va='center', 
                ha='left' if v >= 0 else 'right',
                fontweight='bold'
            )
        
        # Set title and labels
        plt.title(f'Top {top_n} Features Correlated with {target}', fontsize=14, pad=20)
        plt.xlabel(f'{method.capitalize()} Correlation Coefficient')
        plt.ylabel('Features')
        
        plt.tight_layout()
        plt.show()
    
    def plot_dimensionality_reduction(self, 
                                    df: pd.DataFrame, 
                                    target: Optional[str] = None,
                                    method: str = 'pca',
                                    n_components: int = 2,
                                    figsize: Optional[Tuple[int, int]] = None) -> None:
        """Plot dimensionality reduction visualization.
        
        Args:
            df: Input DataFrame
            target: Target feature for color-coding points
            method: Dimensionality reduction method ('pca', 'tsne')
            n_components: Number of components to reduce to
            figsize: Figure size
        """
        # Select features and target
        if target is not None and target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]
        else:
            X = df
            y = None
        
        # Select only numeric features
        X = X.select_dtypes(include=np.number)
        
        if X.empty:
            print("No numeric features available for dimensionality reduction.")
            return
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            title = 'PCA Projection'
            comp_names = [f'Principal Component {i+1}' for i in range(n_components)]
        
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(X) - 1))
            title = 't-SNE Projection'
            comp_names = [f't-SNE Component {i+1}' for i in range(n_components)]
        
        else:
            print(f"Unsupported method: {method}")
            return
        
        # Apply dimensionality reduction
        X_reduced = reducer.fit_transform(X_scaled)
        
        # Set up figure size
        if figsize is None:
            figsize = (10, 8)
        
        # Create scatter plot
        plt.figure(figsize=figsize)
        
        if n_components == 2:
            # 2D scatter plot
            if y is not None:
                # Check if target is categorical or continuous
                if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                    # Continuous target
                    scatter = plt.scatter(
                        X_reduced[:, 0], 
                        X_reduced[:, 1],
                        c=y, 
                        cmap='viridis', 
                        alpha=0.8,
                        edgecolor='k',
                        s=50
                    )
                    plt.colorbar(scatter, label=target)
                else:
                    # Categorical target
                    for i, label in enumerate(sorted(y.unique())):
                        mask = (y == label)
                        plt.scatter(
                            X_reduced[mask, 0], 
                            X_reduced[mask, 1],
                            label=label,
                            alpha=0.8,
                            edgecolor='k',
                            s=50
                        )
                    plt.legend()
            else:
                plt.scatter(
                    X_reduced[:, 0], 
                    X_reduced[:, 1],
                    alpha=0.8,
                    edgecolor='k',
                    s=50
                )
            
            plt.xlabel(comp_names[0])
            plt.ylabel(comp_names[1])
        
        elif n_components == 3:
            # 3D scatter plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            if y is not None:
                # Check if target is categorical or continuous
                if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
                    # Continuous target
                    scatter = ax.scatter(
                        X_reduced[:, 0], 
                        X_reduced[:, 1], 
                        X_reduced[:, 2],
                        c=y, 
                        cmap='viridis', 
                        alpha=0.8,
                        edgecolor='k',
                        s=50
                    )
                    plt.colorbar(scatter, label=target)
                else:
                    # Categorical target
                    for i, label in enumerate(sorted(y.unique())):
                        mask = (y == label)
                        ax.scatter(
                            X_reduced[mask, 0], 
                            X_reduced[mask, 1], 
                            X_reduced[mask, 2],
                            label=label,
                            alpha=0.8,
                            edgecolor='k',
                            s=50
                        )
                    plt.legend()
            else:
                ax.scatter(
                    X_reduced[:, 0], 
                    X_reduced[:, 1], 
                    X_reduced[:, 2],
                    alpha=0.8,
                    edgecolor='k',
                    s=50
                )
            
            ax.set_xlabel(comp_names[0])
            ax.set_ylabel(comp_names[1])
            ax.set_zlabel(comp_names[2])
        
        plt.title(title, fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
        
        # If PCA, plot explained variance
        if method.lower() == 'pca' and hasattr(reducer, 'explained_variance_ratio_'):
            plt.figure(figsize=(10, 6))
            
            # Cumulative explained variance
            cum_var = np.cumsum(reducer.explained_variance_ratio_)
            
            plt.bar(
                range(1, n_components + 1), 
                reducer.explained_variance_ratio_,
                alpha=0.7, 
                color='skyblue'
            )
            plt.step(
                range(1, n_components + 1), 
                cum_var,
                where='mid', 
                color='red', 
                marker='o', 
                linestyle='-', 
                linewidth=2
            )
            
            plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='95% Threshold')
            plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.5, label='90% Threshold')
            
            plt.title('Explained Variance by Principal Components')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.xticks(range(1, n_components + 1))
            plt.legend()
            plt.tight_layout()
            plt.show()
    
    def plot_feature_against_target(self, 
                                   df: pd.DataFrame, 
                                   feature: str,
                                   target: str,
                                   bins: int = 30,
                                   figsize: Optional[Tuple[int, int]] = None) -> None:
        """Plot a feature against the target variable.
        
        Args:
            df: Input DataFrame
            feature: Feature to plot
            target: Target feature
            bins: Number of bins for histograms
            figsize: Figure size
        """
        if feature not in df.columns or target not in df.columns:
            print(f"Feature '{feature}' or target '{target}' not found in DataFrame.")
            return
        
        # Set up figure size
        if figsize is None:
            figsize = (14, 6)
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Scatter plot with regression line
        sns.regplot(
            x=feature, 
            y=target, 
            data=df, 
            scatter_kws={'alpha': 0.5}, 
            line_kws={'color': 'red'},
            ax=ax1
        )
        ax1.set_title(f'Relationship between {feature} and {target}')
        ax1.set_xlabel(feature)
        ax1.set_ylabel(target)
        
        # Calculate and display correlation
        corr = df[[feature, target]].corr().iloc[0, 1]
        ax1.text(
            0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax1.transAxes, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        # Plot 2: Joint histogram
        sns.histplot(
            df, 
            x=feature, 
            y=target, 
            bins=bins,
            cmap='viridis',
            cbar=True,
            ax=ax2
        )
        ax2.set_title(f'Joint Distribution of {feature} and {target}')
        ax2.set_xlabel(feature)
        ax2.set_ylabel(target)
        
        plt.tight_layout()
        plt.show()
    
    def plot_interactive_scatter(self, 
                               df: pd.DataFrame, 
                               x: str,
                               y: str,
                               color: Optional[str] = None,
                               size: Optional[str] = None,
                               hover_data: Optional[List[str]] = None,
                               title: str = 'Interactive Scatter Plot') -> None:
        """Create an interactive scatter plot using Plotly.
        
        Args:
            df: Input DataFrame
            x: Feature for x-axis
            y: Feature for y-axis
            color: Feature for color-coding points
            size: Feature for sizing points
            hover_data: Additional features to show on hover
            title: Plot title
        """
        try:
            # Create interactive scatter plot
            fig = px.scatter(
                df, 
                x=x, 
                y=y,
                color=color,
                size=size,
                hover_data=hover_data,
                title=title,
                template='plotly_white',
                height=600
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 20}
                },
                xaxis_title=x,
                yaxis_title=y,
                legend_title=color if color else '',
                coloraxis_colorbar=dict(title=color) if color else None,
                hovermode='closest'
            )
            
            # Show plot
            fig.show()
        
        except Exception as e:
            print(f"Error creating interactive plot: {e}")
            print("Falling back to static plot...")
            
            # Fallback to static plot
            plt.figure(figsize=self.figsize)
            sns.scatterplot(x=x, y=y, hue=color, size=size, data=df)
            plt.title(title)
            plt.show()
    
    def plot_3d_scatter(self, 
                       df: pd.DataFrame, 
                       x: str,
                       y: str,
                       z: str,
                       color: Optional[str] = None,
                       size: Optional[str] = None,
                       hover_data: Optional[List[str]] = None,
                       title: str = '3D Scatter Plot') -> None:
        """Create a 3D scatter plot using Plotly.
        
        Args:
            df: Input DataFrame
            x: Feature for x-axis
            y: Feature for y-axis
            z: Feature for z-axis
            color: Feature for color-coding points
            size: Feature for sizing points
            hover_data: Additional features to show on hover
            title: Plot title
        """
        try:
            # Create 3D scatter plot
            fig = px.scatter_3d(
                df, 
                x=x, 
                y=y, 
                z=z,
                color=color,
                size=size,
                hover_data=hover_data,
                title=title,
                height=800
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 20}
                },
                scene=dict(
                    xaxis_title=x,
                    yaxis_title=y,
                    zaxis_title=z
                ),
                legend_title=color if color else ''
            )
            
            # Show plot
            fig.show()
        
        except Exception as e:
            print(f"Error creating 3D plot: {e}")
            print("Falling back to static plot...")
            
            # Fallback to static 2D plot matrix
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            if color is not None and color in df.columns:
                scatter = ax.scatter(
                    df[x], 
                    df[y], 
                    df[z],
                    c=df[color], 
                    cmap='viridis', 
                    alpha=0.7,
                    edgecolor='k',
                    s=50 if size is None else df[size]*20
                )
                plt.colorbar(scatter, label=color)
            else:
                ax.scatter(
                    df[x], 
                    df[y], 
                    df[z],
                    alpha=0.7,
                    edgecolor='k',
                    s=50 if size is None else df[size]*20
                )
            
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(z)
            ax.set_title(title)
            
            plt.show()
    
    def plot_geographic_heatmap(self, 
                              df: pd.DataFrame, 
                              lat_col: str,
                              lon_col: str,
                              value_col: str,
                              zoom: int = 12,
                              radius: int = 10,
                              title: str = 'Geographic Heatmap') -> None:
        """Create a geographic heatmap using Plotly.
        
        Args:
            df: Input DataFrame
            lat_col: Column with latitude values
            lon_col: Column with longitude values
            value_col: Column with values for heatmap intensity
            zoom: Initial zoom level
            radius: Radius for heatmap points
            title: Plot title
        """
        try:
            # Create a map centered on the mean coordinates
            center_lat = df[lat_col].mean()
            center_lon = df[lon_col].mean()
            
            fig = px.density_mapbox(
                df, 
                lat=lat_col, 
                lon=lon_col, 
                z=value_col,
                radius=radius,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom,
                mapbox_style="open-street-map",
                title=title,
                height=700
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': title,
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 20}
                },
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            # Show plot
            fig.show()
        
        except Exception as e:
            print(f"Error creating geographic heatmap: {e}")
            print("Falling back to static scatter plot...")
            
            # Fallback to
