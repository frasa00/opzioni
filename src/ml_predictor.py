# src/ml_predictor.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class OptionsMLPredictor:
    """Machine Learning predictor for options market signals."""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the ML predictor.
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        
    def prepare_features_target(self, historical_df, target_col='target_direction', forecast_horizon=1):
        """
        Prepares features (X) and target (y) from historical indicator DataFrame.
        'target_col' should be pre-calculated in the historical_df (e.g., 1 if next day return > 0).
        'forecast_horizon' is how many days ahead to predict.
        """
        if target_col not in historical_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        
        # Use all indicator columns as features, exclude the target and date
        feature_cols = [col for col in historical_df.columns if col not in [target_col, 'timestamp', 'date']]
        X = historical_df[feature_cols].shift(forecast_horizon).iloc[forecast_horizon:]  # Lag features
        y = historical_df[target_col].iloc[forecast_horizon:]  # Align target
        
        # Drop rows with NaN values created by shifting
        valid_indices = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train the selected model on the prepared data.
        """
        # Split data - use TimeSeriesSplit for financial data to avoid lookahead bias
        tscv = TimeSeriesSplit(n_splits=5)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, 
                                                max_depth=10,
                                                random_state=random_state,
                                                n_jobs=-1)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100,
                                                    max_depth=6,
                                                    random_state=random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Simple train/test split for demonstration (prefer cross-validation)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=random_state
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained: {self.model_type}")
        print(f"Test Accuracy: {accuracy:.2%}")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return accuracy
    
    def predict(self, current_features):
        """
        Make a prediction for the current market state.
        
        Args:
            current_features: DataFrame or dict with the same features used in training.
            
        Returns:
            prediction (int/float), probability (dict)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        if isinstance(current_features, dict):
            current_features = pd.DataFrame([current_features])
        
        prediction = self.model.predict(current_features)[0]
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(current_features)[0]
            proba_dict = {f'class_{i}': prob for i, prob in enumerate(proba)}
        else:
            proba_dict = {'confidence': None}
        
        return prediction, proba_dict
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        if self.model is not None:
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save.")
    
    def load_model(self, filepath):
        """Load a trained model from disk."""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
