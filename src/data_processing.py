"""
Feature engineering and data preprocessing for credit risk modeling.
Includes Weight of Evidence (WoE) transformation and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class WeightOfEvidenceEncoder(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence (WoE) transformer for categorical/binned variables.
    WoE = ln(% of non-events / % of events)
    Positive WoE = Lower risk (more non-events)
    Negative WoE = Higher risk (more events)
    """
    
    def __init__(self, eps: float = 0.5):
        """
        Args:
            eps: Small value to avoid log(0) or division by zero
        """
        self.eps = eps
        self.woe_maps = {}
        self.iv_values = {}  # Information Value per feature
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Calculate WoE mappings from training data."""
        for col in X.columns:
            woe_map = {}
            total_events = y.sum()
            total_non_events = len(y) - total_events
            
            for category in X[col].unique():
                mask = X[col] == category
                n_events = y[mask].sum()
                n_non_events = mask.sum() - n_events
                
                # Apply epsilon smoothing
                event_dist = (n_events + self.eps) / (total_events + self.eps)
                non_event_dist = (n_non_events + self.eps) / (total_non_events + self.eps)
                
                woe = np.log(non_event_dist / event_dist)
                woe_map[category] = woe
            
            self.woe_maps[col] = woe_map
            
            # Calculate Information Value
            iv = 0
            for category, woe in woe_map.items():
                mask = X[col] == category
                n_events = y[mask].sum()
                n_non_events = mask.sum() - n_events
                
                event_pct = (n_events + self.eps) / (total_events + self.eps)
                non_event_pct = (n_non_events + self.eps) / (total_non_events + self.eps)
                
                iv += (non_event_pct - event_pct) * woe
            
            self.iv_values[col] = iv
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply WoE transformation."""
        X_transformed = X.copy()
        for col in X.columns:
            if col in self.woe_maps:
                X_transformed[col] = X[col].map(self.woe_maps[col]).fillna(0)
        return X_transformed


def calculate_iv_woe(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> pd.DataFrame:
    """
    Calculate Information Value and Weight of Evidence for all features.
    
    Args:
        df: Input dataframe
        target_col: Name of target column (0 = non-default, 1 = default)
        feature_cols: List of feature column names
    
    Returns:
        DataFrame with IV scores for each feature
    """
    iv_results = []
    
    for col in feature_cols:
        # Bin continuous variables
        if df[col].dtype in ['float64', 'int64'] and df[col].nunique() > 10:
            df[f'{col}_binned'] = pd.qcut(df[col], q=10, duplicates='drop')
            col_to_use = f'{col}_binned'
        else:
            col_to_use = col
        
        # Calculate distribution
        grouped = df.groupby(col_to_use)[target_col].agg(['sum', 'count'])
        grouped.columns = ['events', 'total']
        grouped['non_events'] = grouped['total'] - grouped['events']
        
        total_events = df[target_col].sum()
        total_non_events = len(df) - total_events
        
        # Add smoothing
        grouped['event_pct'] = (grouped['events'] + 0.5) / (total_events + 0.5)
        grouped['non_event_pct'] = (grouped['non_events'] + 0.5) / (total_non_events + 0.5)
        
        # Calculate WoE and IV
        grouped['woe'] = np.log(grouped['non_event_pct'] / grouped['event_pct'])
        grouped['iv_contrib'] = (grouped['non_event_pct'] - grouped['event_pct']) * grouped['woe']
        
        iv = grouped['iv_contrib'].sum()
        
        iv_results.append({
            'feature': col,
            'information_value': iv,
            'predictive_power': _iv_strength(iv)
        })
        
        # Clean up temporary column
        if f'{col}_binned' in df.columns:
            df.drop(columns=[f'{col}_binned'], inplace=True)
    
    return pd.DataFrame(iv_results).sort_values('information_value', ascending=False)


def _iv_strength(iv: float) -> str:
    """Classify IV strength based on industry standards."""
    if iv < 0.02:
        return 'Not predictive'
    elif iv < 0.1:
        return 'Weak'
    elif iv < 0.3:
        return 'Medium'
    elif iv < 0.5:
        return 'Strong'
    else:
        return 'Suspiciously high (overfitting)'


def select_features_by_iv(df: pd.DataFrame, target_col: str, features: List[str], 
                          iv_threshold: float = 0.02) -> List[str]:
    """
    Select features based on Information Value threshold.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        features: List of candidate features
        iv_threshold: Minimum IV to keep feature (default 0.02)
    
    Returns:
        List of selected feature names
    """
    iv_df = calculate_iv_woe(df, target_col, features)
    selected = iv_df[iv_df['information_value'] >= iv_threshold]['feature'].tolist()
    
    print(f"Selected {len(selected)} features out of {len(features)} (IV >= {iv_threshold})")
    print("\nTop 5 features by IV:")
    print(iv_df.head())
    
    return selected


def handle_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle missing values by dropping columns with > threshold missing and
    imputing remaining with median for numerical, mode for categorical.
    
    Args:
        df: Input dataframe
        threshold: Maximum allowed missing fraction (default 0.5 = 50%)
    
    Returns:
        Cleaned dataframe
    """
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    
    if cols_to_drop:
        print(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Impute remaining
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'MISSING', inplace=True)
    
    return df


def create_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features commonly used in credit scoring.
    
    Args:
        df: Input dataframe with base credit variables
    
    Returns:
        Dataframe with additional engineered features
    """
    df = df.copy()
    
    # Debt-to-Income ratio (if columns exist)
    if 'debt' in df.columns and 'income' in df.columns:
        df['debt_to_income'] = df['debt'] / (df['income'] + 1)
    
    # Credit utilization (if columns exist)
    if 'credit_balance' in df.columns and 'credit_limit' in df.columns:
        df['utilization_rate'] = df['credit_balance'] / (df['credit_limit'] + 1)
    
    # Average age of accounts (if age columns exist)
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    if len(age_cols) > 0:
        df['avg_account_age'] = df[age_cols].mean(axis=1)
    
    return df


def prepare_training_data(df: pd.DataFrame, target_col: str = 'default') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete data preparation pipeline for credit risk modeling.
    
    Args:
        df: Raw dataframe
        target_col: Target column name
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    print("Starting data preparation pipeline...")
    
    # Separate target
    if target_col in df.columns:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Handle missing values
    X = handle_missing_values(X)
    
    # Create derived features
    X = create_credit_features(X)
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Default rate: {y.mean():.2%}")
    
    return X, y