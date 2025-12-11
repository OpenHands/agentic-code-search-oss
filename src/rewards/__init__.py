import importlib
import pkgutil
from pathlib import Path

REWARD_REGISTRY = {}

def get_reward_function(reward_name: str):
    """Get a reward function by name from the registry."""
    if reward_name not in REWARD_REGISTRY:
        raise ValueError(f"Reward function '{reward_name}' not found in registry.")
    return REWARD_REGISTRY[reward_name]

def reward(name: str):
    """Decorator to register a new reward function."""
    def decorator(func):
        REWARD_REGISTRY[name] = func
        return func
    return decorator

def _auto_load_rewards():
    """Automatically discover and import all reward modules to register functions."""
    current_dir = Path(__file__).parent
    
    # Recursively import all Python modules
    def _import_submodules(path, package_name):
        # Import all Python modules in this directory
        for importer, modname, ispkg in pkgutil.iter_modules([str(path)]):
            if modname != '__init__':
                try:
                    importlib.import_module(f'.{modname}', package=package_name)
                except ImportError:
                    pass
        
        # Recursively process subdirectories
        for item in path.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                try:
                    # Import the package (runs __init__.py if it exists)
                    importlib.import_module(f'.{item.name}', package=package_name)
                except ImportError:
                    pass
                # Recursively import modules from subdirectories
                _import_submodules(item, f'{package_name}.{item.name}')
    
    _import_submodules(current_dir, __name__)

# Auto-load all reward functions on import
_auto_load_rewards()