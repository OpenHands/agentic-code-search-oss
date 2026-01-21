import importlib
import pkgutil
from pathlib import Path

TOOL_REGISTRY = {}

DEFAULT_OPENHANDS_TOOLS = [
    "apply_patch",
    "browser_use",
    "delegate",
    "file_editor",
    "glob",
    "grep",
    "planning_file_editor",
    "preset",
    "task_tracker",
    "terminal",
    "tom_consult"
]

def tool_exists(tool_name: str):
    """Check if a tool exists in the registry."""
    return tool_name in DEFAULT_OPENHANDS_TOOLS or tool_name in TOOL_REGISTRY

def tool(name: str):
    """Decorator to register a new tool function."""
    def decorator(func):
        if name in DEFAULT_OPENHANDS_TOOLS:
            raise ValueError(f"Tool name '{name}' is an in-built openhands tool and cannot be overridden.")

        # Track the tool in local registry for run-time validation
        TOOL_REGISTRY[name] = func
        return func
    return decorator

def _auto_load_tools():
    """Automatically discover and import all tool modules to register functions."""
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

# Auto-load all tool functions on import
_auto_load_tools()
