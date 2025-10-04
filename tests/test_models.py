"""Tests for models module."""

import pytest
import pandas as pd
import numpy as np

from ml_microstructure.models import (
    ModelConfig,
    LogisticRegressionPredictor,
    RandomForestPredictor,
    LightGBMPredictor,
    ModelFactory,
)


class TestModelConfig:
    """Test ModelConfig class."""
    
    def test_model_config_creation(self):
        """Test ModelConfig creation."""
        config = ModelConfig(
            model_type="lightgbm",
            random_state=42,
            test_size=0.2,
            validation_size=0.2,
            class_weight="balanced"
        )
        
        assert config.model_type == "lightgbm"
        assert config.random_state == 42
        assert config.test_size == 0.2
        assert config.validation_size == 0.2
        assert config.class_weight == "balanced"
    
    def test_model_config_defaults(self):
        """Test ModelConfig with defaults."""
        config = ModelConfig(model_type="lightgbm")
        
        assert config.model_type == "lightgbm"
        assert config.random_state == 42
        assert config.test_size == 0.2
        assert config.validation_size == 0.2
        assert config.class_weight == "balanced"


class TestLogisticRegressionPredictor:
    """Test LogisticRegressionPredictor class."""
    
    def test_logistic_regression_creation(self):
        """Test LogisticRegressionPredictor creation."""
        config = ModelConfig(model_type="logistic_regression")
        predictor = LogisticRegressionPredictor(config)
        
        assert predictor.config == config
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert not predictor.is_fitted
    
    def test_logistic_regression_fit_predict(self):
        """Test LogisticRegressionPredictor fit and predict."""
        # Create test data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
        })
        y = pd.Series(np.random.randint(0, 3, 100))
        
        config = ModelConfig(model_type="logistic_regression")
        predictor = LogisticRegressionPredictor(config)
        
        # Fit model
        predictor.fit(X, y)
        
        assert predictor.is_fitted
        assert len(predictor.feature_names) == 3
        
        # Make predictions
        predictions = predictor.predict(X)
        probabilities = predictor.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_logistic_regression_evaluate(self):
        """Test LogisticRegressionPredictor evaluation."""
        # Create test data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })
        y = pd.Series(np.random.randint(0, 3, 100))
        
        config = ModelConfig(model_type="logistic_regression")
        predictor = LogisticRegressionPredictor(config)
        predictor.fit(X, y)
        
        # Evaluate model
        metrics = predictor.evaluate(X, y)
        
        assert "accuracy" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics
        assert metrics["accuracy"] >= 0.0
        assert metrics["accuracy"] <= 1.0


class TestRandomForestPredictor:
    """Test RandomForestPredictor class."""
    
    def test_random_forest_creation(self):
        """Test RandomForestPredictor creation."""
        config = ModelConfig(model_type="random_forest")
        predictor = RandomForestPredictor(config)
        
        assert predictor.config == config
        assert predictor.model is not None
        assert not predictor.is_fitted
    
    def test_random_forest_fit_predict(self):
        """Test RandomForestPredictor fit and predict."""
        # Create test data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })
        y = pd.Series(np.random.randint(0, 3, 100))
        
        config = ModelConfig(model_type="random_forest")
        predictor = RandomForestPredictor(config)
        
        # Fit model
        predictor.fit(X, y)
        
        assert predictor.is_fitted
        assert len(predictor.feature_names) == 2
        
        # Make predictions
        predictions = predictor.predict(X)
        probabilities = predictor.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_random_forest_feature_importance(self):
        """Test RandomForestPredictor feature importance."""
        # Create test data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })
        y = pd.Series(np.random.randint(0, 3, 100))
        
        config = ModelConfig(model_type="random_forest")
        predictor = RandomForestPredictor(config)
        predictor.fit(X, y)
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == 2
        assert all(imp >= 0.0 for imp in importance.values())
        assert all(imp <= 1.0 for imp in importance.values())


class TestLightGBMPredictor:
    """Test LightGBMPredictor class."""
    
    def test_lightgbm_creation(self):
        """Test LightGBMPredictor creation."""
        config = ModelConfig(model_type="lightgbm")
        predictor = LightGBMPredictor(config)
        
        assert predictor.config == config
        assert predictor.model is not None
        assert not predictor.is_fitted
    
    def test_lightgbm_fit_predict(self):
        """Test LightGBMPredictor fit and predict."""
        # Create test data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })
        y = pd.Series(np.random.randint(0, 3, 100))
        
        config = ModelConfig(model_type="lightgbm")
        predictor = LightGBMPredictor(config)
        
        # Fit model
        predictor.fit(X, y)
        
        assert predictor.is_fitted
        assert len(predictor.feature_names) == 2
        
        # Make predictions
        predictions = predictor.predict(X)
        probabilities = predictor.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_lightgbm_feature_importance(self):
        """Test LightGBMPredictor feature importance."""
        # Create test data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })
        y = pd.Series(np.random.randint(0, 3, 100))
        
        config = ModelConfig(model_type="lightgbm")
        predictor = LightGBMPredictor(config)
        predictor.fit(X, y)
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == 2
        assert all(imp >= 0.0 for imp in importance.values())


class TestModelFactory:
    """Test ModelFactory class."""
    
    def test_create_model(self):
        """Test model creation."""
        config = ModelConfig(model_type="lightgbm")
        model = ModelFactory.create_model(config)
        
        assert isinstance(model, LightGBMPredictor)
        assert model.config == config
    
    def test_create_model_unknown_type(self):
        """Test model creation with unknown type."""
        config = ModelConfig(model_type="unknown_model")
        
        with pytest.raises(ValueError):
            ModelFactory.create_model(config)
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = ModelFactory.get_available_models()
        
        assert isinstance(models, list)
        assert "logistic_regression" in models
        assert "random_forest" in models
        assert "lightgbm" in models
        assert "lstm" in models
        assert "transformer" in models



