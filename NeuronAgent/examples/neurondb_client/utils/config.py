"""
Configuration management utilities
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """
    Load and manage configuration from files and environment
    
    Usage:
        loader = ConfigLoader()
        config = loader.load_from_file("config.json")
        api_key = loader.get_env("NEURONAGENT_API_KEY")
    """
    
    @staticmethod
    def load_from_file(filepath: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            Configuration dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get environment variable
        
        Args:
            key: Environment variable name
            default: Default value if not set
        
        Returns:
            Environment variable value or default
        """
        return os.getenv(key, default)
    
    @staticmethod
    def load_agent_config(filepath: str, profile_name: str) -> Dict[str, Any]:
        """
        Load agent configuration from file
        
        Args:
            filepath: Path to agent configs JSON file
            profile_name: Name of the profile to load
        
        Returns:
            Agent configuration dictionary
        """
        configs = ConfigLoader.load_from_file(filepath)
        profiles = configs.get('agent_configurations', {})
        
        if profile_name not in profiles:
            raise ValueError(f"Profile '{profile_name}' not found in {filepath}")
        
        return profiles[profile_name]
    
    @staticmethod
    def find_config_file(filename: str, search_paths: Optional[list] = None) -> Optional[str]:
        """
        Find configuration file in common locations
        
        Args:
            filename: Configuration file name
            search_paths: Optional list of paths to search
        
        Returns:
            Path to file or None if not found
        """
        if search_paths is None:
            search_paths = [
                '.',
                '~/.neurondb',
                '/etc/neurondb',
                os.path.expanduser('~/.config/neurondb')
            ]
        
        for path in search_paths:
            expanded = os.path.expanduser(path)
            filepath = os.path.join(expanded, filename)
            if os.path.exists(filepath):
                return filepath
        
        return None

