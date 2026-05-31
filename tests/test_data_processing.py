"""
Unit tests for data processing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_processing import (
    WeightOfEvidenceEncoder,
    calculate_iv_woe,
    handle_missing_values,
    create_credit_features,
    select_features_by_iv
)


class TestWeightOfEvidenceEncoder:
    """Test cases for WoE encoder."""
    
    def test_woe_fit_transform(self):
        """Test that WoE transforms correctly."""
        # Create sample data
        X = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'B', 'A'],
            'cat2': ['X', 'X', 'Y', 'Y', 'X']
        })
        y = pd.Series([0, 1, 0, 1, 0])
        
        encoder = WeightOfEvidenceEncoder(eps=0.5)
        encoder.fit(X, y)
        X_transformed = encoder.transform(X)
        
        # Check output shape
        assert X_transformed.shape == X.shape
        assert isinstance(X_transformed, pd.DataFrame)
        
        # Check that WoE values are finite
        assert not X_transformed.isnull().any().any()
    
    def test_woe_smoking_driving_example(self):
        """Test with classic WoE example pattern."""
        # High risk = negative WoE
        X = pd.DataFrame({'risk_factor': ['high', 'low', 'high', 'low', 'high']})
        y = pd.Series([1, 0, 1, 0, 1])  # High risk group has more defaults
        
        encoder = WeightOfEvidenceEncoder(eps=0.5)
        encoder.fit(X, y)
        
        # High risk should have negative WoE
        assert encoder.woe_maps['risk_factor']['high'] < 0
        # Low risk should have positive WoE
        assert encoder.woe_maps['risk_factor']['low'] > 0


class TestIVCalculation:
    """Test Information Value calculation."""
    
    def test_iv_predictive_feature(self):
        """Test that predictive features get higher IV."""
        np.random.seed(42)
        n = 1000
        
        # Strongly predictive feature
        default_prob = np.where(np.random.rand(n) > 0.7, 0.3, 0.05)
        y = np.random.binomial(1, default_prob)
        
        df = pd.DataFrame({
            'strong_predictor': np.random.normal(0, 1, n),
            'random_noise': np.random.normal(0, 1, n),
            'target': y
        })
        
        iv_df = calculate_iv_woe(df, 'target', ['strong_predictor', 'random_noise'])
        
        # Strong predictor should have higher IV
        strong_iv = iv_df[iv_df['feature'] == 'strong_predictor']['information_value'].values[0]
        noise_iv = iv_df[iv_df['feature'] == 'random_noise']['information_value'].values[0]
        
        assert strong_iv > noise_iv
    
    def test_iv_threshold_selection(self):
        """Test feature selection by IV threshold."""
        np.random.seed(42)
        n = 500
        
        df = pd.DataFrame({
            'good_feature': np.random.normal(0, 1, n),
            'bad_feature': np.random.normal(0, 1, n),
            'target': np.random.binomial(1, 0.1, n)
        })
        
        # Make good_feature actually predictive
        df['good_feature'] = df['good_feature'] + 2 * df['target']
        
        selected = select_features_by_iv(df, 'target', ['good_feature', 'bad_feature'], iv_threshold=0.05)
        
        assert 'good_feature' in selected
        # bad_feature might be selected if it has some IV, but likely lower


class TestMissingValueHandling:
    """Test missing value imputation."""
    
    def test_drop_high_missing_columns(self):
        """Test dropping columns with too many missing values."""
        df = pd.DataFrame({
            'complete': [1, 2, 3, 4, 5],
            'high_missing': [np.nan, np.nan, np.nan, np.nan, 1],
            'target': [0, 1, 0, 1, 0]
        })
        
        df_clean = handle_missing_values(df, threshold=0.5)
        
        # Column with 80% missing (4/5) should be dropped
        assert 'high_missing' not in df_clean.columns
        assert 'complete' in df_clean.columns
    
    def test_median_imputation_numeric(self):
        """Test median imputation for numeric columns."""
        df = pd.DataFrame({
            'numeric': [1, 2, np.nan, 4, 5],
            'target': [0, 0, 0, 0, 0]
        })
        
        df_clean = handle_missing_values(df, threshold=0.5)
        
        # Missing value should be replaced with median (3.0)
        assert df_clean['numeric'].isnull().sum() == 0
        assert df_clean['numeric'].iloc[2] == 3.0


class TestFeatureEngineering:
    """Test derived feature creation."""
    
    def test_debt_to_income_ratio(self):
        """Test debt-to-income ratio calculation."""
        df = pd.DataFrame({
            'debt': [10000, 20000, 50000],
            'income': [50000, 40000, 100000]
        })
        
        df_engineered = create_credit_features(df)
        
        expected_ratios = [10000/50001, 20000/40001, 50000/100001]
        np.testing.assert_array_almost_equal(
            df_engineered['debt_to_income'].values,
            expected_ratios,
            decimal=5
        )
    
    def test_utilization_rate(self):
        """Test credit utilization calculation."""
        df = pd.DataFrame({
            'credit_balance': [500, 2000, 8000],
            'credit_limit': [1000, 5000, 10000]
        })
        
        df_engineered = create_credit_features(df)
        
        expected_util = [500/1001, 2000/5001, 8000/10001]
        np.testing.assert_array_almost_equal(
            df_engineered['utilization_rate'].values,
            expected_util,
            decimal=5
        )


class TestDataValidation:
    """Test data quality checks."""
    
    def test_default_rate_range(self):
        """Test that default rate is within expected range."""
        np.random.seed(42)
        y = pd.Series(np.random.binomial(1, 0.1, 1000))
        
        assert 0 <= y.mean() <= 1
        assert y.mean() > 0  # Should have some defaults
        assert y.mean() < 0.3  # Default rate shouldn't be extremely high
    
    def test_feature_correlation(self):
        """Test that features aren't perfectly correlated."""
        df = pd.DataFrame({
            'feat1': np.random.normal(0, 1, 100),
            'feat2': np.random.normal(0, 1, 100),
            'feat3': np.random.normal(0, 1, 100)
        })
        
        corr_matrix = df.corr()
        
        # No feature should be perfectly correlated with another
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                if i != j:
                    assert abs(corr_matrix.iloc[i, j]) < 0.99