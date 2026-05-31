"""
Inference module for credit risk predictions.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict
from pathlib import Path

from src.train import CreditRiskModel


class CreditRiskPredictor:
    """
    Production-ready predictor for credit risk scoring.
    Handles input validation, feature engineering, and probability calibration.
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to saved model .joblib file
        """
        self.model = CreditRiskModel()
        self.model.load(model_path)
        
    def predict_proba(self, applicant_data: Union[pd.DataFrame, Dict, List[Dict]]) -> np.ndarray:
        """
        Predict default probability for one or more applicants.
        
        Args:
            applicant_data: Single applicant dict, list of dicts, or DataFrame
        
        Returns:
            Array of default probabilities (0 to 1)
        """
        # Convert to DataFrame
        if isinstance(applicant_data, dict):
            df = pd.DataFrame([applicant_data])
        elif isinstance(applicant_data, list):
            df = pd.DataFrame(applicant_data)
        else:
            df = applicant_data.copy()
        
        # Ensure all required features are present
        missing_features = set(self.model.selected_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Apply same feature engineering as training
        from src.data_processing import create_credit_features
        df = create_credit_features(df)
        
        # Make prediction
        probabilities = self.model.predict_proba(df)
        
        return probabilities
    
    def predict_risk_category(self, probabilities: np.ndarray, 
                              thresholds: Dict[str, float] = None) -> List[str]:
        """
        Convert probabilities to risk categories.
        
        Args:
            probabilities: Array of default probabilities
            thresholds: Dict with 'low', 'medium', 'high' thresholds
        
        Returns:
            List of risk categories
        """
        if thresholds is None:
            thresholds = {
                'low': 0.02,    # 0-2% = Low risk
                'medium': 0.05,  # 2-5% = Medium risk
                'high': 0.10     # 5-10% = High risk
                # >10% = Very High risk
            }
        
        categories = []
        for prob in probabilities:
            if prob < thresholds['low']:
                categories.append('Low Risk')
            elif prob < thresholds['medium']:
                categories.append('Medium Risk')
            elif prob < thresholds['high']:
                categories.append('High Risk')
            else:
                categories.append('Very High Risk')
        
        return categories
    
    def explain_prediction(self, applicant_data: Dict, feature_contributions: bool = True) -> Dict:
        """
        Provide explanation for a single prediction (for regulatory compliance).
        
        Args:
            applicant_data: Single applicant dictionary
            feature_contributions: Whether to show per-feature contributions
        
        Returns:
            Dictionary with explanation, probability, and feature impacts
        """
        df = pd.DataFrame([applicant_data])
        proba = self.predict_proba(df)[0]
        
        explanation = {
            'default_probability': float(proba),
            'risk_category': self.predict_risk_category(np.array([proba]))[0],
            'model_type': self.model.model_type,
            'is_interpretable': self.model.model_type == 'logistic_regression'
        }
        
        if feature_contributions and self.model.model_type == 'logistic_regression':
            # For logistic regression, we can show feature contributions
            X_filtered = df[self.model.selected_features].copy()
            from src.data_processing import create_credit_features
            X_filtered = create_credit_features(X_filtered)
            
            if self.model.woe_encoder:
                X_woe = self.model.woe_encoder.transform(X_filtered)
                X_scaled = self.model.scaler.transform(X_woe)
                
                # Calculate contribution per feature
                coefficients = self.model.model.coef_[0]
                contributions = X_scaled[0] * coefficients
                
                feature_contrib = dict(zip(self.model.selected_features, contributions))
                # Sort by absolute contribution
                feature_contrib = dict(sorted(feature_contrib.items(), 
                                             key=lambda x: abs(x[1]), 
                                             reverse=True))
                
                explanation['feature_contributions'] = feature_contrib
                explanation['intercept'] = float(self.model.model.intercept_[0])
        
        return explanation


def main():
    """Example inference pipeline."""
    print("Credit Risk Prediction Demo")
    print("=" * 40)
    
    # Load trained model
    predictor = CreditRiskPredictor('models/logistic_regression_model.joblib')
    
    # Single applicant
    applicant = {
        'income': 65000,
        'debt': 15000,
        'credit_age_years': 8,
        'num_delinquencies': 1,
        'credit_utilization': 0.35,
        'num_credit_cards': 3
    }
    
    print("\nApplicant 1:")
    for k, v in applicant.items():
        print(f"  {k}: {v}")
    
    proba = predictor.predict_proba(applicant)[0]
    category = predictor.predict_risk_category([proba])[0]
    
    print(f"\nPrediction:")
    print(f"  Default Probability: {proba:.2%}")
    print(f"  Risk Category: {category}")
    
    # Explain prediction (for regulator/audit)
    print("\nExplanation (for compliance):")
    explanation = predictor.explain_prediction(applicant)
    print(f"  Model Type: {explanation['model_type']}")
    print(f"  Is Interpretable: {explanation['is_interpretable']}")
    
    if 'feature_contributions' in explanation:
        print("\n  Top feature contributions:")
        for feature, contrib in list(explanation['feature_contributions'].items())[:3]:
            print(f"    {feature}: {contrib:+.4f}")
    
    # Batch prediction
    applicants = [
        {'income': 120000, 'debt': 5000, 'credit_age_years': 15, 
         'num_delinquencies': 0, 'credit_utilization': 0.10, 'num_credit_cards': 2},
        {'income': 30000, 'debt': 35000, 'credit_age_years': 2, 
         'num_delinquencies': 3, 'credit_utilization': 0.85, 'num_credit_cards': 1},
    ]
    
    print("\n" + "=" * 40)
    print("Batch Prediction:")
    probabilities = predictor.predict_proba(applicants)
    categories = predictor.predict_risk_category(probabilities)
    
    for i, (app, prob, cat) in enumerate(zip(applicants, probabilities, categories), 1):
        print(f"\nApplicant {i}:")
        print(f"  Income: ${app['income']:,}")
        print(f"  Default Probability: {prob:.2%}")
        print(f"  Risk Category: {cat}")


if __name__ == "__main__":
    main()