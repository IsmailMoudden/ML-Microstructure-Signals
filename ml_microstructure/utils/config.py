"""Configuration utilities for microstructure signals."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manager for configuration files."""
    
    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path("configs")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_name: Name of configuration file (without extension)
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded successfully")
        return config
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_name: Name of configuration file (without extension)
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving configuration to {config_path}")
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved successfully")
    
    def get_config_path(self, config_name: str) -> Path:
        """Get path to configuration file.
        
        Args:
            config_name: Name of configuration file (without extension)
            
        Returns:
            Path to configuration file
        """
        return self.config_dir / f"{config_name}.yaml"



