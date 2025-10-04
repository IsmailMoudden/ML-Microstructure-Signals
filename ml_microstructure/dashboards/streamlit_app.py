"""Streamlit dashboard for microstructure signals analysis."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ml_microstructure.data import SyntheticLOBGenerator, OrderBookProcessor
from ml_microstructure.features import FeaturePipeline
from ml_microstructure.models import ModelFactory, ModelConfig
from ml_microstructure.backtest import BacktestRunner
from ml_microstructure.utils.labeling import LabelGenerator

logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="ML Microstructure Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">ML Microstructure Signals Dashboard</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Configuration")

# Data source selection
data_source = st.sidebar.selectbox(
    "Data Source",
    ["Synthetic", "LOBSTER", "Kaggle Crypto"],
    help="Select the data source for analysis"
)

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["logistic_regression", "random_forest", "lightgbm", "lstm", "transformer"],
    help="Select the model type for prediction"
)

# MLflow configuration
st.sidebar.subheader("MLflow Configuration")
mlflow_tracking_uri = st.sidebar.text_input(
    "MLflow Tracking URI",
    value="file:./mlruns",
    help="MLflow tracking URI"
)

mlflow_experiment = st.sidebar.text_input(
    "MLflow Experiment",
    value="microstructure_signals",
    help="MLflow experiment name"
)

# Backtest configuration
st.sidebar.subheader("Backtest Configuration")
long_threshold = st.sidebar.slider(
    "Long Threshold",
    min_value=0.5,
    max_value=0.9,
    value=0.6,
    step=0.05,
    help="Probability threshold for long signals"
)

short_threshold = st.sidebar.slider(
    "Short Threshold",
    min_value=0.1,
    max_value=0.5,
    value=0.4,
    step=0.05,
    help="Probability threshold for short signals"
)

transaction_cost = st.sidebar.slider(
    "Transaction Cost",
    min_value=0.0,
    max_value=0.01,
    value=0.001,
    step=0.0001,
    help="Transaction cost per trade"
)

slippage = st.sidebar.slider(
    "Slippage",
    min_value=0.0,
    max_value=0.01,
    value=0.0005,
    step=0.0001,
    help="Slippage per trade"
)

# Main content
@st.cache_data
def load_synthetic_data():
    """Load synthetic data."""
    generator = SyntheticLOBGenerator(
        initial_price=100.0,
        tick_size=0.01,
        max_levels=10,
        arrival_rate=100.0,
        duration_seconds=3600
    )
    snapshots = generator.generate_data()
    
    processor = OrderBookProcessor(max_levels=10)
    df = processor.process_snapshots(snapshots)
    
    return df

@st.cache_data
def extract_features(df):
    """Extract features from data."""
    feature_pipeline = FeaturePipeline()
    df_features = feature_pipeline.extract_features(df)
    return df_features

@st.cache_data
def generate_labels(df):
    """Generate labels for data."""
    label_generator = LabelGenerator(horizon=1, threshold=0.001)
    labels = label_generator.generate_labels(df)
    df_labeled = df.copy()
    df_labeled['label'] = labels
    return df_labeled

def create_price_chart(df):
    """Create price chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['mid_price'],
        mode='lines',
        name='Mid Price',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Mid Price Over Time",
        xaxis_title="Time",
        yaxis_title="Price",
        height=400
    )
    
    return fig

def create_spread_chart(df):
    """Create spread chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['spread'],
        mode='lines',
        name='Spread',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Bid-Ask Spread Over Time",
        xaxis_title="Time",
        yaxis_title="Spread",
        height=400
    )
    
    return fig

def create_imbalance_chart(df):
    """Create imbalance chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['imbalance'],
        mode='lines',
        name='Imbalance',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title="Order Book Imbalance Over Time",
        xaxis_title="Time",
        yaxis_title="Imbalance",
        height=400
    )
    
    return fig

def create_feature_importance_chart(feature_importance):
    """Create feature importance chart."""
    if not feature_importance:
        return None
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*sorted_features[:20])  # Top 20 features
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(importance),
        y=list(features),
        orientation='h',
        marker=dict(color='lightblue')
    ))
    
    fig.update_layout(
        title="Feature Importance (Top 20)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=600
    )
    
    return fig

def create_equity_curve_chart(df):
    """Create equity curve chart."""
    if 'cumulative_pnl' not in df.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['cumulative_pnl'],
        mode='lines',
        name='Cumulative PnL',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Cumulative PnL",
        height=400
    )
    
    return fig

def create_drawdown_chart(df):
    """Create drawdown chart."""
    if 'cumulative_pnl' not in df.columns:
        return None
    
    # Calculate drawdown
    cumulative_pnl = df['cumulative_pnl']
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=drawdown,
        mode='lines',
        name='Drawdown',
        line=dict(color='red', width=2),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Time",
        yaxis_title="Drawdown",
        height=400
    )
    
    return fig

def create_signal_chart(df):
    """Create signal chart."""
    if 'signal' not in df.columns:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price and Signals', 'Signal Distribution'),
        vertical_spacing=0.1
    )
    
    # Price and signals
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['mid_price'],
        mode='lines',
        name='Mid Price',
        line=dict(color='blue', width=2)
    ), row=1, col=1)
    
    # Add signal markers
    long_signals = df[df['signal'] == 1]
    short_signals = df[df['signal'] == -1]
    
    if len(long_signals) > 0:
        fig.add_trace(go.Scatter(
            x=long_signals['timestamp'],
            y=long_signals['mid_price'],
            mode='markers',
            name='Long Signals',
            marker=dict(color='green', size=8, symbol='triangle-up')
        ), row=1, col=1)
    
    if len(short_signals) > 0:
        fig.add_trace(go.Scatter(
            x=short_signals['timestamp'],
            y=short_signals['mid_price'],
            mode='markers',
            name='Short Signals',
            marker=dict(color='red', size=8, symbol='triangle-down')
        ), row=1, col=1)
    
    # Signal distribution
    signal_counts = df['signal'].value_counts().sort_index()
    fig.add_trace(go.Bar(
        x=signal_counts.index,
        y=signal_counts.values,
        name='Signal Count',
        marker=dict(color=['red', 'gray', 'green'])
    ), row=2, col=1)
    
    fig.update_layout(
        title="Trading Signals Analysis",
        height=800
    )
    
    return fig

# Main dashboard
if st.button("Run Analysis", type="primary"):
    with st.spinner("Loading data and running analysis..."):
        # Load data
        if data_source == "Synthetic":
            df = load_synthetic_data()
        else:
            st.error("Only synthetic data is supported in this demo")
            st.stop()
        
        # Extract features
        df_features = extract_features(df)
        
        # Generate labels
        df_labeled = generate_labels(df_features)
        
        # Display basic statistics
        st.subheader("Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df_labeled))
        
        with col2:
            st.metric("Features", len(df_features.columns))
        
        with col3:
            st.metric("Label Distribution", df_labeled['label'].value_counts().to_dict())
        
        with col4:
            st.metric("Data Range", f"{df_labeled['timestamp'].min()} to {df_labeled['timestamp'].max()}")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Data Analysis", "Feature Analysis", "Model Analysis", "Backtest Results"])
        
        with tab1:
            st.subheader("Data Analysis")
            
            # Price chart
            st.plotly_chart(create_price_chart(df_labeled), use_container_width=True)
            
            # Spread and imbalance charts
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_spread_chart(df_labeled), use_container_width=True)
            with col2:
                st.plotly_chart(create_imbalance_chart(df_labeled), use_container_width=True)
        
        with tab2:
            st.subheader("Feature Analysis")
            
            # Feature correlation heatmap
            numeric_features = df_features.select_dtypes(include=[np.number]).columns
            if len(numeric_features) > 1:
                corr_matrix = df_features[numeric_features].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature statistics
            st.subheader("Feature Statistics")
            st.dataframe(df_features[numeric_features].describe())
        
        with tab3:
            st.subheader("Model Analysis")
            
            # Model training simulation
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    # Prepare data
                    feature_cols = [col for col in df_labeled.columns if col not in ['timestamp', 'label']]
                    X = df_labeled[feature_cols]
                    y = df_labeled['label']
                    
                    # Split data
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                    
                    # Train model
                    model_config = ModelConfig(
                        model_type=model_type,
                        random_state=42,
                        class_weight="balanced"
                    )
                    
                    model = ModelFactory.create_model(model_config)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    predictions = model.predict(X_test)
                    probabilities = model.predict_proba(X_test)
                    
                    # Display results
                    st.success("Model trained successfully!")
                    
                    # Accuracy
                    accuracy = (predictions == y_test).mean()
                    st.metric("Test Accuracy", f"{accuracy:.2%}")
                    
                    # Feature importance
                    if hasattr(model, 'get_feature_importance'):
                        feature_importance = model.get_feature_importance()
                        fig = create_feature_importance_chart(feature_importance)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Backtest Results")
            
            # Backtest simulation
            if st.button("Run Backtest"):
                with st.spinner("Running backtest..."):
                    # Generate synthetic predictions for demo
                    np.random.seed(42)
                    n_samples = len(df_labeled)
                    
                    # Create synthetic predictions
                    prob_up = np.random.beta(2, 2, n_samples)
                    prob_down = np.random.beta(2, 2, n_samples)
                    prob_flat = 1 - prob_up - prob_down
                    
                    # Normalize probabilities
                    total_prob = prob_up + prob_down + prob_flat
                    prob_up /= total_prob
                    prob_down /= total_prob
                    prob_flat /= total_prob
                    
                    # Create predictions DataFrame
                    df_pred = df_labeled.copy()
                    df_pred['prob_up'] = prob_up
                    df_pred['prob_down'] = prob_down
                    df_pred['prob_flat'] = prob_flat
                    
                    # Generate signals
                    from ml_microstructure.backtest.signals import SignalConfig, SignalGenerator
                    
                    signal_config = SignalConfig(
                        long_threshold=long_threshold,
                        short_threshold=short_threshold,
                        position_sizing="fixed",
                        position_size=1.0
                    )
                    
                    signal_generator = SignalGenerator(signal_config)
                    df_signals = signal_generator.generate_signals(df_pred)
                    
                    # Execute trades
                    from ml_microstructure.backtest.execution import ExecutionConfig, ExecutionEngine
                    
                    execution_config = ExecutionConfig(
                        transaction_cost=transaction_cost,
                        slippage=slippage,
                        max_position=10.0
                    )
                    
                    execution_engine = ExecutionEngine(execution_config)
                    df_executed = execution_engine.execute_trades(df_signals)
                    
                    # Calculate metrics
                    from ml_microstructure.backtest.metrics import MetricsConfig, BacktestMetrics
                    
                    metrics_config = MetricsConfig()
                    metrics_calculator = BacktestMetrics(metrics_config)
                    metrics = metrics_calculator.calculate_metrics(df_executed)
                    
                    # Display results
                    st.success("Backtest completed successfully!")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total PnL", f"{metrics.get('total_pnl', 0):.2f}")
                    
                    with col2:
                        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    
                    with col3:
                        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}")
                    
                    with col4:
                        st.metric("Hit Rate", f"{metrics.get('hit_rate', 0):.2%}")
                    
                    # Charts
                    st.plotly_chart(create_equity_curve_chart(df_executed), use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_drawdown_chart(df_executed), use_container_width=True)
                    with col2:
                        st.plotly_chart(create_signal_chart(df_executed), use_container_width=True)
                    
                    # Detailed metrics
                    st.subheader("Detailed Metrics")
                    st.json(metrics)

# Footer
st.markdown("---")
st.markdown(
    "**ML Microstructure Signals Dashboard** | "
    "Built with Streamlit and Plotly | "
    "For educational and research purposes"
)



