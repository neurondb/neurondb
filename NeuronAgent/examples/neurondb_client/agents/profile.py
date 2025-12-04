"""
Agent profile definitions and management
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class AgentProfile:
    """
    Agent profile configuration
    
    Attributes:
        name: Unique profile name
        description: Profile description
        system_prompt: System prompt for the agent
        model_name: LLM model to use
        enabled_tools: List of enabled tools
        config: Model configuration (temperature, max_tokens, etc.)
        memory_table: Optional memory table name
    """
    name: str
    system_prompt: str
    model_name: str = "gpt-4"
    description: Optional[str] = None
    enabled_tools: List[str] = None
    config: Dict = None
    memory_table: Optional[str] = None
    
    def __post_init__(self):
        """Set defaults"""
        if self.enabled_tools is None:
            self.enabled_tools = ['sql', 'http']
        if self.config is None:
            self.config = {
                'temperature': 0.7,
                'max_tokens': 2000,
                'top_p': 0.9
            }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API"""
        return {
            'name': self.name,
            'system_prompt': self.system_prompt,
            'model_name': self.model_name,
            'description': self.description,
            'enabled_tools': self.enabled_tools,
            'config': self.config,
            'memory_table': self.memory_table
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentProfile':
        """Create from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_file(cls, filepath: str, profile_name: str) -> 'AgentProfile':
        """Load profile from JSON file"""
        with open(filepath, 'r') as f:
            configs = json.load(f)
        
        profile_data = configs['agent_configurations'].get(profile_name)
        if not profile_data:
            raise ValueError(f"Profile '{profile_name}' not found in {filepath}")
        
        return cls.from_dict(profile_data)


def load_profiles_from_file(filepath: str) -> Dict[str, AgentProfile]:
    """
    Load all profiles from a JSON file
    
    Args:
        filepath: Path to JSON configuration file
    
    Returns:
        Dictionary mapping profile names to AgentProfile objects
    """
    with open(filepath, 'r') as f:
        configs = json.load(f)
    
    profiles = {}
    for name, data in configs.get('agent_configurations', {}).items():
        profiles[name] = AgentProfile.from_dict(data)
    
    return profiles


# Predefined profiles
DEFAULT_PROFILES = {
    'general_assistant': AgentProfile(
        name='general-assistant',
        description='General purpose assistant',
        system_prompt='You are a helpful, harmless, and honest assistant.',
        model_name='gpt-4',
        enabled_tools=['sql', 'http'],
        config={'temperature': 0.7, 'max_tokens': 1000}
    ),
    'code_assistant': AgentProfile(
        name='code-assistant',
        description='Code analysis and programming',
        system_prompt='You are an expert programmer. Help with code analysis and writing.',
        model_name='gpt-4',
        enabled_tools=['code', 'sql'],
        config={'temperature': 0.3, 'max_tokens': 2000}
    ),
    'data_analyst': AgentProfile(
        name='data-analyst',
        description='Data analysis and SQL',
        system_prompt='You are a data analyst. Help with SQL queries and data analysis.',
        model_name='gpt-4',
        enabled_tools=['sql'],
        config={'temperature': 0.2, 'max_tokens': 1500}
    ),
    'research_assistant': AgentProfile(
        name='research-assistant',
        description='Research and information gathering',
        system_prompt='You are a research assistant. Help gather and synthesize information.',
        model_name='gpt-4',
        enabled_tools=['http', 'sql'],
        config={'temperature': 0.5, 'max_tokens': 2000}
    ),
}


def get_default_profile(name: str) -> Optional[AgentProfile]:
    """Get a default profile by name"""
    return DEFAULT_PROFILES.get(name)

