# ML Microstructure Signals

A machine learning system for predicting short-term mid-price moves from order book features and backtesting trading signals.

**Note**: This is a student project - some features are experimental and results may vary.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/IsmailMoudden/ML-Microstructure-Signals.git
cd ML-Microstructure-Signals

# Install in development mode
pip install -e ".[dev]"

# Or install production dependencies only
pip install -e .
```

### Basic Usage

```bash
# Train a model with synthetic data
python -m ml_microstructure.pipeline.train config=configs/model/lgbm.yaml

# Make predictions
python -m ml_microstructure.pipeline.predict run_id=<mlflow_run_id>

# Evaluate model performance
python -m ml_microstructure.pipeline.evaluate run_id=<mlflow_run_id>

# Run backtest
python -m ml_microstructure.backtest.run run_id=<mlflow_run_id>

# Launch dashboard
streamlit run ml_microstructure/dashboards/streamlit_app.py
```

## 📊 Features

### Data Sources
- **Synthetic LOB**: Poisson arrival generator for testing
- **LOBSTER**: High-frequency order book data (config available)
- **Kaggle Crypto**: Cryptocurrency order book data (config available)

### Feature Engineering
- **Order Flow Imbalance (OFI)**: Multi-level order flow analysis
- **Spread Features**: Bid-ask spread dynamics
- **Depth Features**: Order book depth analysis
- **Imbalance Features**: Queue imbalance metrics
- **VWAP Features**: Volume-weighted average price
- **Rolling Returns**: Multi-horizon return features
- **Microprice**: Weighted mid-price calculation

### Models
- **Baseline Models**: Logistic Regression, Random Forest, LightGBM
- **Sequence Models**: LSTM, Transformer (available in code)
- **Hyperparameter Optimization**: Optuna integration
- **Model Persistence**: MLflow tracking

### Backtesting
- **Signal Generation**: Probability-to-signal mapping
- **Execution Engine**: Transaction costs, slippage, position sizing
- **Performance Metrics**: Sharpe, Sortino, Calmar ratios, drawdown analysis
- **Walk-Forward Analysis**: Out-of-sample testing

### Dashboard
- **Live Replay**: Real-time feature visualization
- **Model Analysis**: Feature importance, prediction confidence
- **Backtest Results**: Equity curves, drawdown analysis
- **Interactive Charts**: Plotly-based visualizations

## 🏗️ Architecture

```
ml_microstructure/
├── data/           # Data loaders and processors
├── features/       # Feature extraction pipeline
├── models/         # ML model implementations
├── pipeline/       # Training, prediction, evaluation
├── backtest/       # Signal generation and backtesting
├── dashboards/     # Streamlit dashboard
└── utils/          # Utilities and helpers

configs/            # Hydra configuration files
tests/              # Unit and integration tests
notebooks/          # Jupyter notebooks for EDA
reports/            # LaTeX research reports
```

## 📈 Example Workflow

### 1. Data Preparation

```python
from ml_microstructure.data import SyntheticLOBGenerator, OrderBookProcessor
from ml_microstructure.features import FeaturePipeline

# Generate synthetic data
generator = SyntheticLOBGenerator(
    initial_price=100.0,
    tick_size=0.01,
    max_levels=10,
    arrival_rate=100.0,
    duration_seconds=3600
)
snapshots = generator.generate_data()

# Process into DataFrame
processor = OrderBookProcessor(max_levels=10)
df = processor.process_snapshots(snapshots)

# Extract features
pipeline = FeaturePipeline()
df_features = pipeline.extract_features(df)
```

### 2. Model Training

```python
from ml_microstructure.models import ModelFactory, ModelConfig
from ml_microstructure.utils.labeling import LabelGenerator

# Generate labels
label_generator = LabelGenerator(horizon=1, threshold=0.001)
labels = label_generator.generate_labels(df_features)

# Prepare training data
X = df_features.drop(['timestamp'], axis=1)
y = labels

# Train model
config = ModelConfig(model_type="lightgbm")
model = ModelFactory.create_model(config)
model.fit(X, y)
```

### 3. Backtesting

```python
from ml_microstructure.backtest import BacktestRunner

# Run backtest
runner = BacktestRunner(config)
results = runner.run(run_id="your_mlflow_run_id")

# View results
print(results["report"])
```

## 🔧 Configuration

The system uses Hydra for configuration management. Key configuration files:

- `configs/train.yaml`: Training pipeline configuration
- `configs/predict.yaml`: Prediction pipeline configuration
- `configs/backtest.yaml`: Backtesting configuration
- `configs/model/*.yaml`: Model-specific parameters

### Example Configuration

```yaml
# configs/train.yaml
data:
  type: synthetic
  synthetic:
    initial_price: 100.0
    tick_size: 0.01
    max_levels: 10
    arrival_rate: 100.0
    duration_seconds: 3600

model:
  type: lightgbm
  params:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

labeling:
  horizon: 1
  threshold: 0.001
  method: ternary
```

## 📊 Performance Metrics

The backtesting system calculates comprehensive performance metrics:

- **Return Metrics**: Annualized return, Sharpe ratio, Sortino ratio
- **Risk Metrics**: Maximum drawdown, Value at Risk (VaR)
- **Trade Metrics**: Hit rate, profit factor, turnover
- **Risk-Adjusted Metrics**: Calmar ratio, information ratio

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ml_microstructure --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Integration tests only
```

## 📚 Documentation

- **Research Paper**: [ML_Microstructure_Signals_Research_Paper.pdf](ML_Microstructure_Signals_Research_Paper.pdf) - Complete academic paper with results and analysis
- **Paper Sources**: LaTeX source files in `reports/paper/sources/`
- **API Documentation**: Available in `docs/` (generated with Sphinx)
- **Jupyter Notebooks**: Examples in `notebooks/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run linting
ruff check .
black --check .

# Run type checking
mypy ml_microstructure/
```

## ⚠️ Limitations & Development Challenges

### Technical Limitations
- **Synthetic data only**: Models tested mainly on generated data, no validation on real high-frequency data
- **Simplified costs**: Transaction costs and slippage modeled in a basic way, no signal→order latency
- **No baseline**: Missing comparison with Buy&Hold or simple strategies (SMA)
- **Basic ML metrics**: Focus on financial metrics, missing AUC-PR, calibration, Brier score

### Development Challenges
- **PyTorch complexity**: LSTM/Transformer implementation difficult, memory issues with large datasets
- **Hydra configuration**: Initially confusing setup, complex hierarchy to master
- **Unit tests**: Tedious to write, difficult edge cases to cover
- **Realistic backtest**: Balancing realism and simplicity is complicated

### Unresolved Issues
- No walk-forward analysis enabled by default
- Basic missing data handling
- No operational risk management
- Manual hyperparameter optimization

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LOBSTER**: High-frequency order book data
- **Kaggle**: Cryptocurrency datasets
- **MLflow**: Experiment tracking
- **Hydra**: Configuration management
- **Streamlit**: Dashboard framework

## 📞 Support

For questions and support:

- 📧 Email: ismail.moudden1@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/IsmailMoudden/ML-Microstructure-Signals/issues)
- 📖 Documentation: [Wiki](https://github.com/IsmailMoudden/ML-Microstructure-Signals/wiki)

## Roadmap

- [ ] Real-time data streaming integration
- [ ] Advanced sequence models (Transformer variants)
- [ ] Multi-asset backtesting
- [ ] Risk management modules
- [ ] Cloud deployment templates
- [ ] Additional data sources (Binance, Coinbase)

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. It is not intended for live trading without proper risk management and testing.



