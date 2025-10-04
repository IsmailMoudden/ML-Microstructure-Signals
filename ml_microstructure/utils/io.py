"""I/O utilities for microstructure signals."""

import logging
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

logger = logging.getLogger(__name__)


def save_data(data: Union[pd.DataFrame, Dict[str, Any]], filepath: Union[str, Path], format: str = "auto") -> None:
    """Save data to file.
    
    Args:
        data: Data to save (DataFrame or dictionary)
        filepath: Path to save file
        format: File format ('auto', 'parquet', 'csv', 'json', 'pickle')
    """
    filepath = Path(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "auto":
        format = filepath.suffix[1:]  # Remove the dot
    
    logger.info(f"Saving data to {filepath} in {format} format")
    
    if isinstance(data, pd.DataFrame):
        if format == "parquet":
            data.to_parquet(filepath, index=False)
        elif format == "csv":
            data.to_csv(filepath, index=False)
        elif format == "pickle":
            data.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format for DataFrame: {format}")
    
    elif isinstance(data, dict):
        if format == "json":
            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "pickle":
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format for dictionary: {format}")
    
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
    
    logger.info(f"Data saved successfully")


def load_data(filepath: Union[str, Path], format: str = "auto") -> Union[pd.DataFrame, Dict[str, Any]]:
    """Load data from file.
    
    Args:
        filepath: Path to load file from
        format: File format ('auto', 'parquet', 'csv', 'json', 'pickle')
        
    Returns:
        Loaded data (DataFrame or dictionary)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == "auto":
        format = filepath.suffix[1:]  # Remove the dot
    
    logger.info(f"Loading data from {filepath} in {format} format")
    
    if format == "parquet":
        data = pd.read_parquet(filepath)
    elif format == "csv":
        data = pd.read_csv(filepath)
    elif format == "json":
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
    elif format == "pickle":
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Data loaded successfully")
    return data



